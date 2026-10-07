"""What every AI engine returns, and which engines exist.

Engines:
  local  - the free on-device machine-learning engine (newstrader/ml): FinBERT sentiment + a model trained on
           how stocks actually moved after past headlines. Default.
  claude - Anthropic's Claude (paid per call, needs an API key).

Both answer in the same JSON shape (see prompts.output_schema) and go through the same validator, signal
merging, risk checks and trade logic.
"""

from __future__ import annotations

from dataclasses import dataclass

ENGINES = {
    "local": "Local machine learning (free, runs on this computer)",
    "claude": "Claude (Anthropic API, paid per call)",
}


@dataclass
class AnalysisResult:
    ok: bool
    text: str = ""
    model: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    status: str = "ok"  # ok | refusal | error | truncated
    engine: str = "claude"
