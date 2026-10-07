"""Build a tiny stand-in for Xenova/finbert to exercise the inference code path:
- tokenizer.json: BERT-uncased style WordPiece (BertNormalizer lowercase, BertPreTokenizer,
  [CLS] $A [SEP] post-processor), truncation/padding left unset (like HF exports usually are).
- model.onnx: inputs input_ids / attention_mask / token_type_ids (int64, [batch, seq]),
  output logits float32 [batch, 3]. Logic: logits = masked-mean of an embedding table.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

vocab_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
              "shares", "of", "apple", "jump", "after", "record", "profit",
              "company", "files", "for", "bankruptcy", "the", "results", "were", "in", "line",
              "##s", "beat", "estimates", "miss", "loss", "."]
vocab = {t: i for i, t in enumerate(vocab_list)}

tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
tok.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                            strip_accents=None, lowercase=True)
tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A:0 [SEP]:0", pair="[CLS] $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
tok.decoder = decoders.WordPiece(prefix="##")
tok.save(str(out / "tokenizer.json"))

# Embedding table [V, 3]: positive words push col 0, negative col 1, others col 2
V = len(vocab_list)
emb = np.zeros((V, 3), dtype=np.float32)
emb[:, 2] = 0.5
for w in ("jump", "record", "profit", "beat"):
    emb[vocab[w]] = [3.0, 0.0, 0.0]
for w in ("bankruptcy", "miss", "loss"):
    emb[vocab[w]] = [0.0, 3.0, 0.0]

nodes = [
    helper.make_node("Gather", ["emb", "input_ids"], ["e"], axis=0),            # [B,S,3]
    helper.make_node("Cast", ["attention_mask"], ["m_f"], to=TensorProto.FLOAT),  # [B,S]
    helper.make_node("Unsqueeze", ["m_f", "ax2"], ["m3"]),                      # [B,S,1]
    helper.make_node("Mul", ["e", "m3"], ["em"]),
    helper.make_node("ReduceSum", ["em", "ax1"], ["s"], keepdims=0),            # [B,3]
    helper.make_node("ReduceSum", ["m3", "ax1"], ["n"], keepdims=0),            # [B,1]
    helper.make_node("Div", ["s", "n"], ["mean"]),
    # consume token_type_ids so it is a real graph input (as in BERT exports)
    helper.make_node("Cast", ["token_type_ids"], ["tt_f"], to=TensorProto.FLOAT),
    helper.make_node("ReduceSum", ["tt_f", "ax1"], ["tt_s"], keepdims=1),       # [B,1]
    helper.make_node("Mul", ["tt_s", "zero"], ["tt0"]),
    helper.make_node("Add", ["mean", "tt0"], ["logits"]),
]
inits = [numpy_helper.from_array(emb, "emb"),
         numpy_helper.from_array(np.array([2], dtype=np.int64), "ax2"),
         numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
         numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")]
inputs = [helper.make_tensor_value_info(n, TensorProto.INT64, ["batch_size", "sequence_length"])
          for n in ("input_ids", "attention_mask", "token_type_ids")]
outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 3])]
graph = helper.make_graph(nodes, "fake_finbert", inputs, outputs, inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, str(out / "model.onnx"))
print("wrote", out)
