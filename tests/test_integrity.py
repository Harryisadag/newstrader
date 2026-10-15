"""A downloaded app opened before "Extract All" finished gets a plain message, not a Python error."""

from __future__ import annotations

from zoneinfo import ZoneInfoNotFoundError

import pytest

from newstrader import __main__ as entry
from newstrader import integrity


def _app(tmp_path, files, listed=None):
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    if listed is not None:
        (tmp_path / "newstrader").mkdir(exist_ok=True)
        (tmp_path / integrity.MANIFEST).write_text("\n".join(listed) + "\n", encoding="utf-8")
    return tmp_path


def test_missing_files_listed(tmp_path):
    base = _app(tmp_path, ["a.dll", "tzdata/zoneinfo/America/New_York"],
                listed=["a.dll", "tzdata/zoneinfo/America/New_York", "tzdata/zoneinfo/Europe/London", "b.pyd"])
    assert integrity.missing_files(base, base / integrity.MANIFEST) == ["tzdata/zoneinfo/Europe/London", "b.pyd"]
    text = integrity.incomplete_message(ZoneInfoNotFoundError("x"), base)
    assert "2 of its files are missing" in text and "Extract All" in text


def test_complete_folder_is_not_blamed(tmp_path):
    base = _app(tmp_path, ["a.dll"], listed=["a.dll"])
    assert integrity.incomplete_message(RuntimeError("real bug"), base) is None
    assert integrity.incomplete_message(FileNotFoundError("x"), base) is None


def test_without_the_list_only_missing_file_errors_are_blamed(tmp_path):
    base = _app(tmp_path, ["a.dll"])
    assert integrity.incomplete_message(ZoneInfoNotFoundError("America/New_York"), base)
    assert integrity.incomplete_message(ModuleNotFoundError("x"), base)
    assert integrity.incomplete_message(ValueError("x"), base) is None


def test_source_runs_are_never_blamed(monkeypatch):
    monkeypatch.delattr("sys.frozen", raising=False)
    assert integrity.app_dir() is None
    assert integrity.incomplete_message(FileNotFoundError("x")) is None


def test_entry_point_shows_message_and_exits(tmp_path, monkeypatch):
    base = _app(tmp_path, [], listed=["tzdata/zoneinfo/America/New_York"])
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(base), raising=False)
    shown = []
    monkeypatch.setattr(integrity, "show_error", shown.append)

    def broken_main():
        raise ZoneInfoNotFoundError("No time zone found with key America/New_York")

    monkeypatch.setattr("newstrader.app.main", broken_main)
    assert entry.run() == 1
    assert shown and "incomplete" in shown[0]


def test_entry_point_reraises_real_bugs(tmp_path, monkeypatch):
    base = _app(tmp_path, ["a.dll"], listed=["a.dll"])
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(base), raising=False)

    def broken_main():
        raise RuntimeError("a real bug")

    monkeypatch.setattr("newstrader.app.main", broken_main)
    with pytest.raises(RuntimeError, match="a real bug"):
        entry.run()
