"""Download the real models the app uses and check they behave. Needs internet; used by CI.

    python scripts/check_models.py            # FinBERT: download, inspect, score known headlines
    python scripts/check_models.py --mlx      # also check the Apple-GPU Whisper repos exist (and, on an M-series
                                              # Mac with mlx-whisper installed, transcribe a test clip)
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newstrader.ml.sentiment import (  # noqa: E402
    FINBERT_FILES,
    FINBERT_REPO,
    FINBERT_REVISION,
    FinbertOnnx,
    download_finbert,
    finbert_dir,
)

HEADLINES = [
    ("Nvidia beats revenue estimates and raises its full-year outlook", "positive"),
    ("Apple shares jump after record quarterly profit", "positive"),
    ("Company files for bankruptcy protection after missing debt payment", "negative"),
    ("Tesla shares plunge after regulators open a probe into its self-driving software", "negative"),
    ("The company will hold its annual shareholder meeting on Tuesday", "neutral"),
    ("Ford to report third-quarter results on October 28", "neutral"),
]


def check_finbert() -> int:
    from huggingface_hub import HfApi

    info = HfApi().model_info(FINBERT_REPO, revision=FINBERT_REVISION, files_metadata=True)
    print(f"{FINBERT_REPO} @ {FINBERT_REVISION}: commit {info.sha}")
    sizes = {s.rfilename: s.size for s in info.siblings or []}
    for name in FINBERT_FILES:
        print(f"  {name}: {sizes.get(name)} bytes")
        if name not in sizes:
            print(f"FAILED: {name} missing from the repo")
            return 1
    models = Path(tempfile.mkdtemp(prefix="nt-models-"))
    t0 = time.time()
    download_finbert(models)
    print(f"downloaded in {time.time() - t0:.0f}s")
    m = FinbertOnnx(finbert_dir(models))
    print("inputs:", [(i.name, i.type) for i in m.sess.get_inputs()], "labels:", m.id2label)
    t0 = time.time()
    scores = m.predict([h for h, _ in HEADLINES])
    print(f"scored {len(HEADLINES)} headlines in {(time.time() - t0) * 1000:.0f} ms")
    wrong = 0
    for (text, want), s in zip(HEADLINES, scores, strict=True):
        ok = s.label == want
        wrong += not ok
        print(f"  {'ok ' if ok else 'BAD'} {s.label:8} {s.positive:.2f}/{s.negative:.2f}/{s.neutral:.2f}  {text}")
    if wrong > 1:
        print(f"FAILED: {wrong} of {len(HEADLINES)} headlines scored the wrong way")
        return 1
    return 0


def check_mlx() -> int:
    from huggingface_hub import HfApi

    from newstrader.audio.transcriber import MLX_REPOS, MlxWhisper, mlx_available

    api = HfApi()
    bad = 0
    for name, repo in MLX_REPOS.items():
        try:
            files = [s.rfilename for s in api.model_info(repo).siblings or []]
            weights = [f for f in files if f.startswith("weights") or f.endswith(".safetensors")]
            print(f"  {name:16} {repo}: {'ok' if weights else 'NO WEIGHTS'} ({', '.join(weights[:2])})")
            bad += not weights
        except Exception as exc:
            print(f"  {name:16} {repo}: MISSING ({type(exc).__name__}: {exc})")
            bad += 1
    if not mlx_available():
        print("mlx-whisper not installed here (or not an Apple Silicon Mac) - skipped the transcription test")
        return 1 if bad else 0
    import numpy as np

    try:
        model = MlxWhisper("tiny")
        model.warm_up()
        segs = model.transcribe(np.zeros(16000 * 3, dtype=np.float32), "en", None, 500)
        print(f"mlx-whisper ran on silence: {len(segs)} segments (expected 0)")
    except Exception as exc:  # GitHub's Mac VMs may not expose the Metal GPU
        print(f"mlx-whisper couldn't run here: {type(exc).__name__}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    code = check_finbert()
    if "--mlx" in sys.argv:
        code = check_mlx() or code
    sys.exit(code)
