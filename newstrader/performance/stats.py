"""Is the AI actually right? Win rate and average move by confidence level, source, and traded vs not."""

from __future__ import annotations

from collections import defaultdict

BUCKETS = [(0, 59, "under 60"), (60, 69, "60-69"), (70, 79, "70-79"), (80, 89, "80-89"), (90, 100, "90-100")]
HORIZONS = {"5m": "ret_5m", "1h": "ret_1h", "1d": "ret_1d"}


def bucket_of(conf: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= conf <= hi:
            return label
    return "under 60"


def _summarise(rets: list[float], confs: list[int]) -> dict:
    n = len(rets)
    wins = sum(1 for r in rets if r > 0)
    return {
        "count": n,
        "wins": wins,
        "win_rate": round(100 * wins / n, 1) if n else None,
        "avg_return": round(sum(rets) / n, 3) if n else None,
        "avg_confidence": round(sum(confs) / len(confs), 1) if confs else None,
    }


def compute_stats(rows: list[dict], horizon: str = "1h") -> dict:
    """rows: signals joined with their directional returns (ret_5m / ret_1h / ret_1d, already direction-adjusted)."""
    col = HORIZONS[horizon]
    usable = [r for r in rows if r.get(col) is not None and r.get("direction") in ("bullish", "bearish")]

    def group(key_fn) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in usable:
            groups[key_fn(r)].append(r)
        out = []
        for key, items in groups.items():
            out.append({"key": key, **_summarise([i[col] for i in items], [i["confidence"] for i in items])})
        return out

    by_bucket = {g["key"]: g for g in group(lambda r: bucket_of(int(r["confidence"])))}
    buckets = [by_bucket.get(label, {"key": label, **_summarise([], [])}) for _, _, label in BUCKETS]
    sources = sorted(group(lambda r: r.get("source_name") or "?"), key=lambda g: -g["count"])
    source_types = sorted(group(lambda r: r.get("source_type") or "?"), key=lambda g: -g["count"])
    traded = sorted(group(lambda r: "traded" if r.get("traded") else "not traded"), key=lambda g: g["key"])
    directions = sorted(group(lambda r: r["direction"]), key=lambda g: g["key"])
    return {
        "horizon": horizon,
        "overall": _summarise([r[col] for r in usable], [r["confidence"] for r in usable]),
        "by_confidence": buckets,
        "by_source": sources,
        "by_source_type": source_types,
        "by_traded": traded,
        "by_direction": directions,
        "pending": sum(1 for r in rows if r.get(col) is None),
    }
