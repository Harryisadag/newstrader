"""Mac support: data folder, notifications, GPU status text, settings defaults and the speech engine choice."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from newstrader import paths
from newstrader.alerts.desktop import mac_notification_args
from newstrader.audio import transcriber as tr
from newstrader.config import TranscriptionSettings
from newstrader.system_monitor import describe_gpu


def test_mac_app_data_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("NEWSTRADER_DATA_DIR", raising=False)
    monkeypatch.delenv("NEWSTRADER_ENV_FILE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    d = paths.data_dir()
    assert d == tmp_path / "Library" / "Application Support" / "NewsTrader" and d.is_dir()
    assert paths.env_file() == d / ".env"  # never inside the .app bundle


def test_mac_notification_text_is_passed_as_arguments():
    nasty = 'Buy "NVDA" now" & do shell script "rm -rf ~" --help\nline2'
    args = mac_notification_args("-e title", nasty)
    assert args[0] == "/usr/bin/osascript"
    script = " ".join(args[1:7])
    assert "on run argv" in script and "do shell script" not in script  # text never becomes script source
    assert args[7] == "newstrader"  # fixed first argument, so user text can't look like an osascript option
    assert args[8] == "-e title" and "\n" not in args[9] and args[9].startswith('Buy "NVDA"')
    assert mac_notification_args("", "")[8:] == [" ", " "]


def test_mac_gpu_status_text():
    lvl, text = describe_gpu({"mac": {"chip": "Apple M3 Pro", "apple_silicon": True, "mlx": True}})
    assert lvl == "ok" and "Apple GPU" in text
    lvl, text = describe_gpu({"mac": {"chip": "Apple M1", "apple_silicon": True, "mlx": False}})
    assert lvl == "warn" and "run.command" in text
    lvl, text = describe_gpu({"mac": {"chip": "Intel(R) Core(TM) i7", "apple_silicon": False, "mlx": False}})
    assert lvl == "warn" and "CPU" in text


def test_mac_transcription_defaults(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    t = TranscriptionSettings()
    assert (t.device, t.model, t.compute_type) == ("auto", "large-v3-turbo", "int8")
    monkeypatch.setattr(sys, "platform", "win32")
    t = TranscriptionSettings()
    assert (t.device, t.model, t.compute_type) == ("cuda", "large-v3", "float16")
    assert TranscriptionSettings(device="mlx").device == "mlx"


# ---------------------------------------------------------------- speech engine selection
class FakeWhisper:
    def __init__(self, name, device, compute_type, download_root):
        self.name, self.device = name, device

    def transcribe(self, audio, **kw):
        return iter([SimpleNamespace(start=0.0, end=1.0, text="cpu words", no_speech_prob=0, avg_logprob=0,
                                     compression_ratio=1)]), None


class FakeMlx:
    instances: list = []

    def __init__(self, name):
        self.name = name
        self.calls = []
        FakeMlx.instances.append(self)

    def warm_up(self):
        pass

    def transcribe(self, audio, language, prompt, min_silence_ms):
        self.calls.append((language, prompt, min_silence_ms))
        return [SimpleNamespace(start=0.5, end=1.5, text=" apple gpu words ", no_speech_prob=0.1, avg_logprob=-0.2,
                                compression_ratio=1.1)]


def make_transcriber(settings):
    statuses = []
    t = tr.Transcriber(lambda: settings, lambda lvl, d: statuses.append((lvl, d)))
    return t, statuses


@pytest.fixture
def apple_silicon(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tr.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    FakeMlx.instances.clear()


def loud(n=16000):
    return (np.sin(np.linspace(0, 400, n)) * 0.3).astype(np.float32)


def test_apple_silicon_uses_the_apple_gpu(apple_silicon, monkeypatch):
    monkeypatch.setattr(tr, "MlxWhisper", FakeMlx)
    t, statuses = make_transcriber(TranscriptionSettings(device="auto", model="large-v3-turbo"))
    t._load()
    assert t.device == "mlx" and "Apple GPU" in t.model_desc and statuses[-1][0] == "ok"
    segs = t._transcribe(tr.Job("s", 1000.0, loud(), prompt="Fed"))
    assert [s.text for s in segs] == ["apple gpu words"] and segs[0].start == 1000.5
    assert FakeMlx.instances[0].calls == [("en", "Fed", 500)]  # no beam size passed to mlx-whisper


def test_apple_silicon_without_mlx_falls_back_to_cpu(apple_silicon, monkeypatch):
    def broken(_name):
        raise RuntimeError("the Apple-GPU speech engine (mlx-whisper) isn't installed")

    monkeypatch.setattr(tr, "MlxWhisper", broken)
    t, statuses = make_transcriber(TranscriptionSettings(device="auto", model="large-v3-turbo"))
    t._load()
    assert t.device == "cpu" and t.model.name == tr.CPU_FALLBACK_MODEL
    assert statuses[-1][0] == "warn" and "mlx-whisper" in statuses[-1][1]
    assert [s.text for s in t._transcribe(tr.Job("s", 0.0, loud()))] == ["cpu words"]


def test_intel_mac_goes_straight_to_cpu(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tr.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    t, statuses = make_transcriber(TranscriptionSettings(device="cuda", model="large-v3"))
    t._load()
    assert t.device == "cpu" and "Intel Mac" in statuses[-1][1]


def test_mlx_model_names():
    assert tr.MLX_REPOS["large-v3-turbo"] == "mlx-community/whisper-large-v3-turbo"
    for name in ("large-v3", "medium", "small", "base", "tiny"):
        assert tr.MLX_REPOS[name].startswith("mlx-community/")


def test_mac_gpu_status_rosetta_and_old_macos():
    lvl, text = describe_gpu({"mac": {"chip": "Apple M2", "apple_silicon": True, "rosetta": True, "mlx": False}})
    assert lvl == "warn" and "Rosetta" in text and ".venv" in text
    lvl, text = describe_gpu({"mac": {"chip": "Apple M1", "apple_silicon": True, "mlx": False, "macos": "13.6.1"}})
    assert lvl == "warn" and "macOS 14" in text


def test_run_command_scripts_are_valid_bash():
    import shutil
    import subprocess
    from pathlib import Path

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    root = Path(__file__).resolve().parent.parent
    for name in ("run.command", "update.command", "build_app.command"):
        script = root / name
        assert script.read_bytes().count(b"\r\n") == 0, f"{name} must use LF line endings"
        assert subprocess.run([bash, "-n", str(script)], capture_output=True).returncode == 0, name
