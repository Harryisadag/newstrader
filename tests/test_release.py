"""The release catalogue: every released version has notes, and the current version is described."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import release_notes  # noqa: E402

from newstrader import __version__  # noqa: E402


def test_current_version_has_release_notes():
    notes = release_notes.notes_for(__version__, (ROOT / "RELEASES.md").read_text(encoding="utf-8"))
    assert f"NewsTrader-{__version__}-windows-x64.zip" in notes
    assert f"NewsTrader-{__version__}-mac-apple-silicon.zip" in notes
    assert "How to install" in notes and "Open Anyway" in notes and "<version>" not in notes


def test_code_version_matches_package():
    assert release_notes.code_version() == __version__


def test_missing_version_is_refused():
    with pytest.raises(SystemExit):
        release_notes.notes_for("99.0.0", "## v1.0.0 — x\nnotes\n")


def test_version_headings_match_exactly():
    text = "## v0.2.10 — later\nten\n\n## v0.2.1 — earlier\none\n"
    assert release_notes.notes_for("0.2.1", text).startswith("one")
