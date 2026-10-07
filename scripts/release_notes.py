"""Print the GitHub release notes for one version, taken from RELEASES.md (used by the release workflow).

    python scripts/release_notes.py 0.2.0  > notes.md

The notes are that version's section plus the "Which file do I need?" and "How to install" sections, so the
release page explains itself. Exits with an error if RELEASES.md has no entry for the version - every
release must be described in the catalogue first.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HEADING = re.compile(r"^## ", re.MULTILINE)


def sections(text: str) -> dict[str, str]:
    """{"## heading line": body} for every level-2 section."""
    out: dict[str, str] = {}
    parts = _HEADING.split(text)
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out[title.strip()] = body.strip()
    return out


def notes_for(version: str, text: str) -> str:
    version = version.lstrip("v")
    secs = sections(text)
    entry = next((body for title, body in secs.items()
                  if re.match(rf"v{re.escape(version)}(\s|$|—|-)", title)), None)
    if entry is None:
        raise SystemExit(f"RELEASES.md has no '## v{version}' section - add one before releasing.")
    which = secs.get("Which file do I need?", "").replace("<version>", version)
    install = secs.get("How to install a release", "")
    return "\n\n".join([
        entry,
        "## Which file do I need?\n\n" + which,
        "## How to install\n\n" + install,
        "Full catalogue of releases: [RELEASES.md](https://github.com/Harryisadag/newstrader/blob/"
        "claude/zen-gauss-5ktkek/RELEASES.md)",
    ]) + "\n"


def code_version() -> str:
    init = (ROOT / "newstrader" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else code_version()
    sys.stdout.write(notes_for(ver, (ROOT / "RELEASES.md").read_text(encoding="utf-8")))
