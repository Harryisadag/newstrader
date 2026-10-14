"""Shared training: the label file format, and the checks every label file must pass.

People with a strong PC let Pro AI label old news ("Label old news", llm/teacher.py) and can share those labels
through the GitHub repo (folder shared/labels). A label file is gzip-compressed JSON Lines, one object per line and
nothing else:

    {"v":1,"id":<Alpaca news id>,"h":<first 16 hex of the headline's SHA-256>,"t":<ticker>,
     "d":"bullish|bearish|neutral","c":<0-100>,"e":<event kind or "">,"m":<Pro AI model key>,"day":"YYYY-MM-DD"}

No headline text, no keys, no trades and nothing about the person leave the PC: everyone downloads the same Benzinga
headlines from Alpaca themselves, and a label is matched to them by the news id (with the headline hash as a check).
The file name is newstrader-labels-<date>-<code>.jsonl.gz, where <code> is a random code made once on each PC (so the
app can count how many different PCs contributed; it isn't linked to anyone).

check_file() is used by the app (before a downloaded file is used) and by scripts/check_shared_labels.py, which GitHub
runs on every upload. Only the standard library is used here, so that check needs nothing installed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from ..llm.catalog import MODELS_BY_KEY
from .events import EVENT_KINDS

FORMAT_VERSION = 1
NAME_RE = re.compile(r"^newstrader-labels-(\d{4}-\d{2}-\d{2})-([0-9a-f]{8})\.jsonl\.gz$")
KEYS = frozenset({"v", "id", "h", "t", "d", "c", "e", "m", "day"})
DIRECTIONS = ("bullish", "bearish", "neutral")
TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
HASH_RE = re.compile(r"^[0-9a-f]{16}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EVENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{0,39}$")
MAX_BYTES = 20 * 1024 * 1024  # a file as uploaded (compressed)
MAX_UNPACKED = 120 * 1024 * 1024  # ...and unpacked (stops a "zip bomb")
MAX_ROWS = 300_000
MAX_LINE = 1000
MAX_ID = 10 ** 12
MAX_AGE_DAYS = 3 * 365 + 1  # labels for news up to 3 years old
MIX_FROM_ROWS = 500  # a file with more rows than this...
MIX_MAX_SHARE = 0.9  # ...may not be more than 90% one direction (all-bullish files are broken or made up)
EXPORT_MAX_SHARE = 0.85  # what "Share my labels" trims the most common direction to, to stay well inside that
OTHER_FILES = frozenset({"README.md"})  # the only other file allowed in shared/labels
MAX_ERRORS_SHOWN = 15


def headline_hash(headline: str) -> str:
    """First 16 hex characters of the SHA-256 of the headline (leading / trailing spaces removed)."""
    return hashlib.sha256((headline or "").strip().encode("utf-8")).hexdigest()[:16]


def blob_sha(data: bytes) -> str:
    """The fingerprint git (and GitHub's contents API) gives a file: SHA-1 of "blob <size>\\0" + the bytes."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def file_name(day: date, code: str) -> str:
    return f"newstrader-labels-{day.isoformat()}-{code}.jsonl.gz"


def utc_today() -> date:
    return datetime.now(UTC).date()


@dataclass
class CheckResult:
    """What check_file() found. `rows` holds the parsed rows when the file is fine."""

    name: str
    errors: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    n_rows: int = 0
    code: str = ""  # the contributor code in the file name
    mix: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.name}: OK - {self.n_rows:,} labels ({_mix_text(self.mix)})"
        shown = self.errors[:MAX_ERRORS_SHOWN]
        more = len(self.errors) - len(shown)
        return f"{self.name}: REJECTED - " + "; ".join(shown) + (f"; and {more} more" if more > 0 else "")


def _mix_text(mix: dict[str, int]) -> str:
    total = sum(mix.values()) or 1
    return ", ".join(f"{d} {100 * mix.get(d, 0) / total:.0f}%" for d in DIRECTIONS)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _parse_day(text) -> date | None:
    if not isinstance(text, str) or not DAY_RE.match(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def check_row(row, today: date, known_only: bool = True) -> str:
    """Why one parsed line is wrong ("" = it's fine). known_only=False accepts event kinds and model keys this
    version of NewsTrader doesn't know yet (they only need the right format) - the app uses that, so a file from a
    newer version isn't thrown away; GitHub's check knows the current lists and uses known_only=True."""
    if not isinstance(row, dict):
        return "not a JSON object"
    keys = set(row)
    if keys != KEYS:
        missing, extra = sorted(KEYS - keys), sorted(str(k)[:20] for k in keys - KEYS)
        return "; ".join(([f"missing {', '.join(missing)}"] if missing else [])
                         + ([f"unexpected {', '.join(extra)}"] if extra else []))
    if not _is_int(row["v"]) or row["v"] != FORMAT_VERSION:
        return f"v must be {FORMAT_VERSION}"
    if not _is_int(row["id"]) or not 0 < row["id"] < MAX_ID:
        return "id must be an Alpaca news id (a whole number)"
    if not isinstance(row["h"], str) or not HASH_RE.match(row["h"]):
        return "h must be 16 lower-case hex characters"
    if not isinstance(row["t"], str) or not TICKER_RE.match(row["t"]):
        return "t must be a ticker like AAPL or BRK.B"
    if row["d"] not in DIRECTIONS:
        return "d must be bullish, bearish or neutral"
    if not _is_int(row["c"]) or not 0 <= row["c"] <= 100:
        return "c must be a whole number from 0 to 100"
    e = row["e"]
    if not isinstance(e, str) or (e and not EVENT_RE.match(e)) or (e and known_only and e not in EVENT_KINDS):
        return f"e must be \"\" or a known event kind (got {str(e)[:40]!r})"
    m = row["m"]
    if not isinstance(m, str) or not MODEL_RE.match(m) or (known_only and m not in MODELS_BY_KEY):
        return f"m must be a Pro AI model key (got {str(m)[:40]!r})"
    day = _parse_day(row["day"])
    if day is None:
        return "day must be a date like 2026-01-31"
    if day > today + timedelta(days=1):
        return "day is in the future"
    if day < today - timedelta(days=MAX_AGE_DAYS):
        return "day is more than 3 years ago"
    return ""


def _unpack(data: bytes) -> bytes:
    """gunzip with a size limit."""
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        while True:
            chunk = gz.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            if out.tell() > MAX_UNPACKED:
                raise ValueError(f"unpacks to more than {MAX_UNPACKED // (1024 * 1024)} MB")
    return out.getvalue()


def check_file(name: str, data: bytes, today: date | None = None, known_only: bool = True) -> CheckResult:
    """Every check a label file must pass: its name, gzip, size, the number of rows, every row's keys, types and
    values, no (id, ticker) twice, and a believable mix of directions."""
    today = today or utc_today()
    res = CheckResult(name=name)
    m = NAME_RE.match(name)
    if not m:
        res.errors.append("the name must look like newstrader-labels-2026-01-31-1a2b3c4d.jsonl.gz")
        return res
    res.code = m.group(2)
    named = _parse_day(m.group(1))
    if named is None or named > today + timedelta(days=1):
        res.errors.append("the date in the name isn't a real date, or is in the future")
    if len(data) > MAX_BYTES:
        res.errors.append(f"the file is bigger than {MAX_BYTES // (1024 * 1024)} MB")
        return res
    try:
        text = _unpack(data).decode("utf-8")
    except (OSError, EOFError, zlib.error, ValueError, UnicodeDecodeError) as exc:
        res.errors.append(f"not a gzip-compressed UTF-8 text file ({type(exc).__name__}: {str(exc)[:80]})")
        return res
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the final newline
    if not lines:
        res.errors.append("the file has no labels")
        return res
    if len(lines) > MAX_ROWS:
        res.errors.append(f"more than {MAX_ROWS:,} labels in one file ({len(lines):,})")
        return res
    seen: set[tuple[int, str]] = set()
    rows = []
    for n, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE:
            res.errors.append(f"line {n}: longer than {MAX_LINE} characters")
            continue
        try:
            row = json.loads(line)
        except ValueError:
            res.errors.append(f"line {n}: not valid JSON")
            continue
        why = check_row(row, today, known_only)
        if why:
            res.errors.append(f"line {n}: {why}")
            continue
        key = (row["id"], row["t"])
        if key in seen:
            res.errors.append(f"line {n}: news {row['id']} / {row['t']} is in the file twice")
            continue
        seen.add(key)
        rows.append(row)
    res.n_rows = len(lines)
    res.mix = dict(Counter(r["d"] for r in rows))
    if len(rows) > MIX_FROM_ROWS:
        top, count = max(res.mix.items(), key=lambda kv: kv[1])
        if count > MIX_MAX_SHARE * len(rows):
            res.errors.append(f"{100 * count / len(rows):.0f}% of the labels are {top} (at most "
                              f"{MIX_MAX_SHARE:.0%} may be one direction) - this file looks broken")
    if res.ok:
        res.rows = rows
    return res


def check_folder(folder: Path, today: date | None = None) -> list[CheckResult]:
    """check_file() for every file under `folder` (sub-folders included). README.md at the top is the only other
    file allowed."""
    out = []
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file() or p.is_symlink()):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink():  # (never followed: it could point anywhere on the computer running the check)
            res = CheckResult(name=rel)
            res.errors.append("links aren't allowed - upload the label file itself")
            out.append(res)
            continue
        if rel in OTHER_FILES:
            continue
        if path.parent != Path(folder):
            res = CheckResult(name=rel)
            res.errors.append("label files go straight into shared/labels, not into a sub-folder")
            out.append(res)
            continue
        if path.stat().st_size > MAX_BYTES:
            res = CheckResult(name=rel)
            res.errors.append(f"the file is bigger than {MAX_BYTES // (1024 * 1024)} MB")
            out.append(res)
            continue
        out.append(check_file(path.name, path.read_bytes(), today))
    return out


# ------------------------------------------------------------------------------------------------ making a file
def to_row(label: dict) -> dict | None:
    """A pro_labels row -> a file row (None when it can't be shared, e.g. not an Alpaca news id)."""
    nid = str(label.get("news_id") or "")
    if not nid.isdigit() or not 0 < int(nid) < MAX_ID:
        return None
    row = {"v": FORMAT_VERSION, "id": int(nid), "h": label.get("headline_hash") or "", "t": label.get("ticker") or "",
           "d": label.get("direction") or "", "c": label.get("confidence"), "e": label.get("event") or "",
           "m": label.get("model") or "", "day": label.get("day") or ""}
    if row["e"] not in EVENT_KINDS:
        row["e"] = ""
    return row


def trim_mix(rows: list[dict], max_share: float = EXPORT_MAX_SHARE) -> tuple[list[dict], int]:
    """At most `max_share` of a bigger file may be one direction (usually "neutral": most news doesn't move a
    stock). Drops the extra rows of that direction - an even spread, picked by a hash of the news id - and returns
    (rows, how many were dropped)."""
    if len(rows) <= MIX_FROM_ROWS:
        return rows, 0
    mix = Counter(r["d"] for r in rows)
    top, count = mix.most_common(1)[0]
    others = len(rows) - count
    keep = max(int(max_share * others / (1 - max_share)), MIX_FROM_ROWS - others)  # (small files: no limit)
    if count <= keep:
        return rows, 0
    order = sorted((r for r in rows if r["d"] == top),
                   key=lambda r: hashlib.sha256(f"{r['id']}/{r['t']}".encode()).hexdigest())
    kept = {(r["id"], r["t"]) for r in order[:keep]}
    out = [r for r in rows if r["d"] != top or (r["id"], r["t"]) in kept]
    return out, len(rows) - len(out)


def dump(rows: list[dict]) -> bytes:
    """Rows -> gzip JSON Lines (same bytes for the same rows: no time stamp inside the gzip)."""
    text = "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=True) + "\n" for r in rows)
    return gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0)


def read_rows(path: Path) -> list[dict]:
    """The rows of a label file that already passed check_file() (cached by the app)."""
    text = _unpack(Path(path).read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]
