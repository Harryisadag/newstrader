"""Pro AI hardware detection: nvidia-smi parsing, the server build and the model picked for common machines."""

from __future__ import annotations

import pytest

from newstrader.llm import hardware
from newstrader.llm.catalog import MODELS, NOT_AUTO, SERVER_BUILDS
from newstrader.llm.hardware import HardwareInfo, memory_budget, parse_nvidia_smi, pick_build, pick_model

GIB = 1024**3


def card(name: str, total_mib: int, used_mib: int = 1100, driver: str = "581.57", cap: str = "8.9",
         os: str = "windows") -> HardwareInfo:
    return HardwareInfo(os, name, total_mib, used_mib, driver, cap)


def mac(ram_gb: int, apple: bool = True) -> HardwareInfo:
    return HardwareInfo("mac", "Apple M2" if apple else "Intel Core i9", apple_silicon=apple, ram_bytes=ram_gb * GIB)


RTX_5070_TI = card("NVIDIA GeForce RTX 5070 Ti", 16303, cap="12.0")

# (machine, TV transcription on, expected build, expected model key or None)
TABLE = [
    (RTX_5070_TI, True, "cuda13", "qwen3.5-9b"),
    (RTX_5070_TI, False, "cuda13", "qwen3.5-9b-q6"),
    (card("NVIDIA GeForce RTX 4060", 8188, 700, "566.36"), True, "cuda12", None),
    (card("NVIDIA GeForce RTX 4060", 8188, 700, "566.36"), False, "cuda12", "qwen3.5-4b"),
    (card("NVIDIA GeForce RTX 4090", 24564, 1300), True, "cuda13", "gemma-4-26b-a4b"),
    (card("NVIDIA GeForce RTX 4090", 24564, 1300), False, "cuda13", "gemma-4-26b-a4b"),
    (card("NVIDIA GeForce RTX 5090", 32607, 1300, cap="12.0"), True, "cuda13", "qwen3.6-35b-a3b"),
    (card("NVIDIA GeForce RTX 5090", 32607, 1300, cap="12.0"), False, "cuda13", "qwen3.6-35b-a3b"),
    (mac(16), True, "metal", "qwen3.5-9b"),
    (mac(16), False, "metal", "qwen3.5-9b-q6"),
    (mac(36), True, "metal", "gemma-4-26b-a4b"),
    (mac(64), True, "metal", "qwen3.6-35b-a3b"),
    (mac(32, apple=False), True, None, None),
    (HardwareInfo("windows", note="no NVIDIA card found"), True, "vulkan", None),
    (HardwareInfo("linux", note="no NVIDIA card found"), False, "linux-cpu", None),
    (card("NVIDIA GeForce GTX 1080", 8192, 500, "531.79", "6.1"), False, "vulkan", "qwen3.5-4b"),
]


@pytest.mark.parametrize(("hw", "tv_on", "build", "model"), TABLE)
def test_pick_table(hw, tv_on, build, model):
    key, why = pick_build(hw)
    assert key == build
    assert why
    choice, reason = pick_model(hw, tv_on)
    assert (choice.key if choice else None) == model, reason
    assert reason
    assert choice is None or choice.key not in NOT_AUTO


def test_rtx_5070_ti_with_tv_keeps_room_for_whisper():
    b = memory_budget(RTX_5070_TI, tv_on=True)
    assert b.kind == "gpu" and b.whisper_gb == 5.0
    assert b.reserve_mib == round((5.0 + 1.0) * 1024)
    assert any("TV transcription" in line for line in b.lines)
    # int8 isn't available on RTX 50-series, so transcription still needs the float16 room
    assert pick_model(RTX_5070_TI, True, "int8")[0].key == "qwen3.5-9b"
    # an older card really does save memory with int8
    older = card("NVIDIA GeForce RTX 4080", 16376)
    assert hardware.whisper_reserve_gb(older, True, "int8") == 3.3
    assert hardware.whisper_reserve_gb(older, True, "int8", "cpu") == 0.0


def test_whisper_already_on_the_card_isnt_counted_twice():
    loaded = card("NVIDIA GeForce RTX 5070 Ti", 16303, used_mib=1100 + 5120, cap="12.0")
    assert pick_model(loaded, True)[0].key == "qwen3.5-4b"  # without knowing, it looks like other apps' memory
    choice, _ = pick_model(loaded, True, whisper_loaded=True)
    assert choice.key == "qwen3.5-9b"
    b = memory_budget(loaded, True, whisper_loaded=True)
    assert b.reserve_mib == 1024  # it already holds its memory: llama-server only keeps the margin free
    assert pick_model(loaded, True, ours_mib=5120)[0].key == "qwen3.5-9b"


def test_too_small_says_why():
    choice, why = pick_model(card("NVIDIA GeForce RTX 3050", 6144, 600), True)
    assert choice is None
    assert "Turning off TV transcription" in why
    assert "GB" in why


def test_auto_never_picks_excluded_models():
    huge = card("NVIDIA RTX PRO 6000", 97887, 1000, cap="12.0")
    choice, _ = pick_model(huge, False)
    assert choice.key not in NOT_AUTO
    assert choice.vram_gb == max(m.vram_gb for m in MODELS if m.key not in NOT_AUTO)


@pytest.mark.parametrize(("driver", "build"), [("581.57", "cuda13"), ("580.88", "cuda13"), ("580.0", "cuda13"),
                                               ("576.02", "cuda12"), ("551.61", "cuda12"), ("551.6", "vulkan"),
                                               ("546.33", "vulkan"), ("", "vulkan"), ("garbage", "vulkan")])
def test_driver_versions(driver, build):
    assert pick_build(card("NVIDIA GeForce RTX 3080", 10240, driver=driver))[0] == build
    assert SERVER_BUILDS[build].os == "windows"


def test_old_driver_message_says_what_to_do():
    _, why = pick_build(card("NVIDIA GeForce GTX 1080", 8192, driver="531.79"))
    assert "Update the driver" in why and "551.61" in why


@pytest.mark.parametrize(("text", "expected"), [
    ("NVIDIA GeForce RTX 5070 Ti, 16303, 1234, 581.57, 12.0\n",
     {"gpu_name": "NVIDIA GeForce RTX 5070 Ti", "vram_total_mib": 16303.0, "vram_used_mib": 1234.0,
      "driver": "581.57", "compute_cap": "12.0"}),
    ("NVIDIA GeForce GTX 1080, 8192, 500, 472.12\n",
     {"gpu_name": "NVIDIA GeForce GTX 1080", "vram_total_mib": 8192.0, "vram_used_mib": 500.0, "driver": "472.12",
      "compute_cap": ""}),
    ("NVIDIA GeForce RTX 4090, 24564, 2000, 581.57, [N/A]\nNVIDIA GeForce RTX 3060, 12288, 0, 581.57, 8.6\n",
     {"gpu_name": "NVIDIA GeForce RTX 4090", "vram_total_mib": 24564.0, "vram_used_mib": 2000.0,
      "driver": "581.57", "compute_cap": ""}),
])
def test_parse_nvidia_smi(text, expected):
    assert parse_nvidia_smi(text) == expected


@pytest.mark.parametrize("text", ["", "\n", "No devices were found", "NVIDIA, [N/A], [N/A], 581.57"])
def test_parse_nvidia_smi_rejects_junk(text):
    assert parse_nvidia_smi(text) is None


def test_detect_windows_nvidia(monkeypatch):
    calls = []

    def run(args):
        calls.append(args)
        if any(a.startswith("--query-gpu") for a in args):
            return "NVIDIA GeForce RTX 5070 Ti, 16303, 2000, 581.57, 12.0\n"
        return "4242, 1500\n999, 300\n"

    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: "nvidia-smi")
    hw = hardware.detect(our_pids=[4242], run=run, system="Windows")
    assert hw.os == "windows" and hw.nvidia
    assert hw.gpu_name == "NVIDIA GeForce RTX 5070 Ti" and hw.driver_version == (581, 57)
    assert hw.ours_mib == 1500
    assert hw.as_dict()["vram_total_gb"] == 15.9


def test_detect_old_nvidia_smi_without_compute_cap(monkeypatch):
    def run(args):
        query = next(a for a in args if a.startswith("--query-gpu"))
        if "compute_cap" in query:
            raise RuntimeError('Field "compute_cap" is not a valid field to query.')
        return "NVIDIA GeForce GTX 1080, 8192, 400, 456.71\n"

    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: "nvidia-smi")
    hw = hardware.detect(run=run, system="Windows")
    assert hw.vram_total_mib == 8192 and hw.compute_cap == ""
    assert pick_build(hw)[0] == "vulkan"


def test_detect_without_nvidia(monkeypatch):
    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: None)
    hw = hardware.detect(run=lambda a: "", system="Linux")
    assert not hw.nvidia and hw.note
    assert pick_build(hw)[0] == "linux-cpu"
    assert pick_model(hw, False)[0] is None


def test_detect_nvidia_smi_failure_never_raises(monkeypatch):
    def run(args):
        raise RuntimeError("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver")

    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: "nvidia-smi")
    hw = hardware.detect(run=run, system="Windows")
    assert not hw.nvidia and "nvidia-smi" in hw.note


@pytest.mark.parametrize(("machine", "arm64", "apple"), [("arm64", "1", True), ("x86_64", "1", True),
                                                          ("x86_64", "0", False)])
def test_detect_mac(machine, arm64, apple):
    answers = {"hw.optional.arm64": arm64, "hw.memsize": str(36 * GIB), "machdep.cpu.brand_string": "Apple M3 Pro"}
    hw = hardware.detect(run=lambda a: answers[a[-1]] + "\n", system="Darwin", machine=machine)
    assert hw.os == "mac" and hw.apple_silicon is apple
    assert round(hw.ram_gb) == 36
    assert pick_build(hw)[0] == ("metal" if apple else None)
    if not apple:
        assert "Apple Silicon" in pick_model(hw, True)[1]
