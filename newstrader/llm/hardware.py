"""What graphics card / chip this computer has, which llama.cpp server build suits it, and which model fits in its
memory next to everything else (the desktop, other apps and TV transcription). Never raises."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .catalog import MODELS, NOT_AUTO, SERVER_BUILDS, ModelChoice

MIB = 1024 * 1024
GIB = 1024 * MIB

# the desktop, browser and the app's own window always hold at least this much graphics memory (more on Windows,
# whose desktop and browsers grow quickly; running short there makes the card spill into slow shared memory)
DESKTOP_MIN_GB = {"windows": 1.5}
DESKTOP_MIN_GB_DEFAULT = 0.8
MARGIN_GB = 0.5
FIT_MARGIN_GB = 1.0  # llama-server's own default margin when it fits a model on the card
# graphics memory TV transcription (Whisper) needs, by its compute type
WHISPER_RESERVE_GB = {"float16": 5.0, "float32": 5.0, "int8_float16": 3.3, "int8": 3.3}
WHISPER_MLX_GB = 1.6  # mlx-whisper on Apple Silicon
MAC_USABLE_SHARE = 0.70  # macOS lets the GPU use about this much of the memory
MAC_SYSTEM_GB = 2.0

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_SMI_FIELDS = "name,memory.total,memory.used,driver_version,compute_cap"
_WINDOWS_SMI = (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")


@dataclass
class HardwareInfo:
    os: str  # windows / mac / linux
    gpu_name: str = ""
    vram_total_mib: float = 0.0
    vram_used_mib: float = 0.0
    driver: str = ""
    compute_cap: str = ""
    apple_silicon: bool = False
    ram_bytes: int = 0
    ours_mib: float = 0.0  # graphics memory this app's own processes hold (when nvidia-smi can tell)
    note: str = ""  # e.g. "no NVIDIA card found"

    @property
    def nvidia(self) -> bool:
        return self.vram_total_mib > 0

    @property
    def intel_mac(self) -> bool:
        return self.os == "mac" and not self.apple_silicon

    @property
    def driver_version(self) -> tuple[int, ...] | None:
        """(major, minor) of the NVIDIA driver, e.g. (581, 57); None when unknown."""
        return version_tuple(self.driver)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / GIB

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(nvidia=self.nvidia, intel_mac=self.intel_mac, driver_version=self.driver_version,
                 ram_gb=round(self.ram_gb, 1), vram_total_gb=round(self.vram_total_mib / 1024, 1))
        return d


Runner = Callable[[list[str]], str]


def version_tuple(value: str | float) -> tuple[int, ...] | None:
    """A driver version as (major, minor): 581.57 -> (581, 57), 580 -> (580, 0); None if it isn't one."""
    parts = str(value).strip().split(".")[:2]
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError:
        return None
    return nums + (0,) * (2 - len(nums))


def _run(args: list[str]) -> str:
    out = subprocess.run(args, capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW,
                         check=False)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or f"exit code {out.returncode}").strip()[:300])
    return out.stdout


def _os_name(system: str) -> str:
    return {"Windows": "windows", "Darwin": "mac"}.get(system, "linux")


def find_nvidia_smi() -> str | None:
    path = shutil.which("nvidia-smi")
    if path:
        return path
    if sys.platform == "win32":
        return next((p for p in _WINDOWS_SMI if Path(p).is_file()), None)
    return None


def _num(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return 0.0


def parse_nvidia_smi(stdout: str) -> dict | None:
    """First GPU's line of `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version[,compute_cap]`."""
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 4 or _num(parts[1]) <= 0:
        return None
    cap = parts[4] if len(parts) > 4 and parts[4] not in ("[N/A]", "N/A") else ""
    return {"gpu_name": parts[0], "vram_total_mib": _num(parts[1]), "vram_used_mib": _num(parts[2]),
            "driver": parts[3], "compute_cap": cap}


def _process_mib(smi: str, run: Runner, pids: set[int]) -> float:
    """Graphics memory used by the given processes (Windows usually can't say - then 0)."""
    try:
        out = run([smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    except Exception:
        return 0.0
    total = 0.0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in pids:
            total += _num(parts[1])
    return total


def _detect_nvidia(hw: HardwareInfo, run: Runner, our_pids: set[int]) -> None:
    smi = find_nvidia_smi()
    if not smi:
        hw.note = "no NVIDIA card found"
        return
    base = [smi, "--format=csv,noheader,nounits"]
    try:
        info = parse_nvidia_smi(run([base[0], f"--query-gpu={_SMI_FIELDS}", base[1]]))
    except Exception:
        info = None
    if info is None:  # drivers older than ~510 don't know compute_cap
        try:
            info = parse_nvidia_smi(run([base[0], f"--query-gpu={_SMI_FIELDS.rsplit(',', 1)[0]}", base[1]]))
        except Exception as exc:
            hw.note = f"no NVIDIA card found (nvidia-smi: {str(exc)[:120]})"
            return
    if info is None:
        hw.note = "no NVIDIA card found"
        return
    for k, v in info.items():
        setattr(hw, k, v)
    if our_pids:
        hw.ours_mib = _process_mib(smi, run, our_pids)


def _detect_mac(hw: HardwareInfo, run: Runner, machine: str) -> None:
    hw.apple_silicon = machine == "arm64"
    try:  # an Intel-only Python running under Rosetta still reports x86_64
        hw.apple_silicon = hw.apple_silicon or run(["/usr/sbin/sysctl", "-n", "hw.optional.arm64"]).strip() == "1"
    except Exception:
        pass
    try:
        hw.ram_bytes = int(run(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip())
    except Exception:
        hw.ram_bytes = 0
    try:
        hw.gpu_name = run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    except Exception:
        pass
    if not hw.apple_silicon:
        hw.note = "Intel Mac"


def detect(our_pids: Iterable[int] = (), run: Runner | None = None, system: str | None = None,
           machine: str | None = None) -> HardwareInfo:
    """Look at this computer. `our_pids`: this app's processes (their graphics memory isn't counted as "in use by
    other apps"). Never raises."""
    run = run or _run
    hw = HardwareInfo(os=_os_name(system or platform.system()))
    try:
        if hw.os == "mac":
            _detect_mac(hw, run, machine or platform.machine())
        else:
            _detect_nvidia(hw, run, set(our_pids))
    except Exception as exc:  # pragma: no cover - belt and braces
        hw.note = f"couldn't read the hardware ({type(exc).__name__})"
    return hw


# ------------------------------------------------------------------------------------------- server build
def pick_build(hw: HardwareInfo) -> tuple[str | None, str]:
    """(SERVER_BUILDS key, plain-English reason). The key is None when Pro AI can't run here."""
    if hw.os == "mac":
        if not hw.apple_silicon:
            return None, ("Pro AI needs a Mac with Apple Silicon (M1 or newer). This Mac has an Intel processor, "
                          "which is too slow for it.")
        return "metal", "Apple Silicon: runs on the Mac's built-in graphics."
    if hw.os == "linux":
        if hw.nvidia:
            return "linux-vulkan", f"{hw.gpu_name}: runs on the graphics card (Vulkan)."
        return "linux-cpu", "No graphics card found: runs on the processor (slow)."
    if not hw.nvidia:
        return "vulkan", "No NVIDIA card found: uses the Vulkan build, which works with most graphics cards."
    drv = hw.driver_version or (0, 0)
    if drv >= version_tuple(SERVER_BUILDS["cuda13"].min_driver):
        return "cuda13", f"{hw.gpu_name}, driver {hw.driver}: the fastest NVIDIA build (CUDA 13)."
    if drv >= version_tuple(SERVER_BUILDS["cuda12"].min_driver):
        return "cuda12", (f"{hw.gpu_name}, driver {hw.driver}: the NVIDIA build for older drivers (CUDA 12). "
                          f"Updating the driver to {SERVER_BUILDS['cuda13'].min_driver:.0f} or newer allows the "
                          "faster CUDA 13 build.")
    return "vulkan", (f"{hw.gpu_name}: the NVIDIA driver ({hw.driver or 'unknown version'}) is too old for the "
                      f"NVIDIA builds, so the slower Vulkan build is used. Update the driver from nvidia.com to "
                      f"{SERVER_BUILDS['cuda12'].min_driver} or newer for full speed.")


# ------------------------------------------------------------------------------------------- memory budget
@dataclass
class MemoryBudget:
    kind: str  # "gpu" (graphics card memory) / "unified" (Apple Silicon) / "unknown"
    total_gb: float = 0.0
    other_gb: float = 0.0  # desktop and other apps (or, on a Mac, macOS and apps)
    whisper_gb: float = 0.0  # kept free for TV transcription
    margin_gb: float = 0.0
    free_gb: float = 0.0  # what the model may use
    whisper_loaded: bool = False  # TV transcription already holds its memory (so it's in "used" already)
    lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reserve_mib"] = self.reserve_mib
        return d

    @property
    def reserve_mib(self) -> int:
        """What llama-server must leave free when it loads (--fit-target): room for TV transcription if it isn't
        loaded yet, plus a margin of at least 1 GB."""
        pending = 0.0 if self.whisper_loaded else self.whisper_gb
        return int(round((pending + max(self.margin_gb, FIT_MARGIN_GB)) * 1024))


def whisper_reserve_gb(hw: HardwareInfo, tv_on: bool, whisper_compute: str = "float16",
                       whisper_device: str = "cuda") -> float:
    """Graphics memory TV transcription takes (0 when it's off or runs on the processor)."""
    if not tv_on or whisper_device == "cpu":
        return 0.0
    if hw.os == "mac":
        return WHISPER_MLX_GB if hw.apple_silicon else 0.0
    if not hw.nvidia:  # transcription needs an NVIDIA card; without one it runs on the processor
        return 0.0
    major = version_tuple(hw.compute_cap or "0")
    if whisper_compute.startswith("int8") and major and major[0] >= 12:
        whisper_compute = "float16"  # RTX 50-series: int8 isn't available, transcription falls back to float16
    return WHISPER_RESERVE_GB.get(whisper_compute, WHISPER_RESERVE_GB["float16"])


def memory_budget(hw: HardwareInfo, tv_on: bool, whisper_compute: str = "float16", ours_mib: float | None = None,
                  whisper_loaded: bool = False, whisper_device: str = "cuda") -> MemoryBudget:
    """How much memory a model may use. `ours_mib`: graphics memory this app already holds (Whisper, a running
    Pro AI server) - it's counted in nvidia-smi's "used" but isn't the desktop's. `whisper_loaded`: TV
    transcription is already on the card; when nvidia-smi can't say how much it holds (usual on Windows), its
    reserve is assumed. `whisper_device` "cpu": transcription doesn't use the graphics card."""
    whisper = whisper_reserve_gb(hw, tv_on, whisper_compute, whisper_device)
    if hw.os == "mac" and hw.apple_silicon and hw.ram_bytes:
        total = hw.ram_gb
        usable = total * MAC_USABLE_SHARE
        free = round(max(0.0, usable - whisper - MAC_SYSTEM_GB), 1)
        b = MemoryBudget("unified", round(total, 1), round(total - usable + MAC_SYSTEM_GB, 1), whisper, 0.0, free)
        b.lines = [f"{total:.0f} GB memory shared with macOS",
                   f"{MAC_SYSTEM_GB + total - usable:.1f} GB kept for macOS and apps"]
    elif hw.nvidia:
        total = hw.vram_total_mib / 1024
        ours = hw.ours_mib if ours_mib is None else ours_mib
        if whisper_loaded and whisper:
            ours = max(ours, whisper * 1024)
        other = max((hw.vram_used_mib - ours) / 1024, DESKTOP_MIN_GB.get(hw.os, DESKTOP_MIN_GB_DEFAULT))
        free = round(max(0.0, total - other - whisper - MARGIN_GB), 1)
        b = MemoryBudget("gpu", round(total, 1), round(other, 1), whisper, MARGIN_GB, free)
        b.lines = [f"{total:.0f} GB graphics card", f"{other:.1f} GB in use by the desktop and other apps",
                   f"{MARGIN_GB} GB kept spare"]
    else:
        return MemoryBudget("unknown", whisper_gb=whisper,
                            lines=["Couldn't measure this computer's graphics memory"])
    b.whisper_loaded = whisper_loaded and b.kind == "gpu"  # (Metal only counts the server's own memory)
    if whisper:
        b.lines.insert(2, f"{whisper:g} GB kept for TV transcription")
    b.lines.append(f"{free:.1f} GB left for Pro AI")
    return b


def pick_model(hw: HardwareInfo, tv_on: bool, whisper_compute: str = "float16", ours_mib: float | None = None,
               whisper_loaded: bool = False, whisper_device: str = "cuda") -> tuple[ModelChoice | None, str]:
    """The biggest automatic model that fits, and why (in plain words)."""
    return _choose(memory_budget(hw, tv_on, whisper_compute, ours_mib, whisper_loaded, whisper_device), hw)


def _choose(b: MemoryBudget, hw: HardwareInfo) -> tuple[ModelChoice | None, str]:
    head = _headline(b)
    if b.kind == "unknown":
        if hw.intel_mac:
            return None, "Pro AI needs a Mac with Apple Silicon (M1 or newer)."
        return None, ("Couldn't measure this computer's graphics memory, so no model was picked automatically - "
                      "choose one yourself.")
    fits = [m for m in MODELS if m.key not in NOT_AUTO and m.vram_gb <= b.free_gb + 1e-9]
    if not fits:
        smallest = min((m for m in MODELS if m.key not in NOT_AUTO), key=lambda m: m.vram_gb)
        tip = " Turning off TV transcription frees memory." if b.whisper_gb else ""
        return None, (f"{head} -> only {b.free_gb:.1f} GB left, and the smallest model needs "
                      f"{smallest.vram_gb:g} GB.{tip}")
    best = max(fits, key=lambda m: m.vram_gb)
    return best, f"{head} -> {best.label} ({best.vram_gb:g} GB)"


def _headline(b: MemoryBudget) -> str:
    if b.kind == "unified":
        text = f"{b.total_gb:.0f} GB Mac"
    elif b.kind == "gpu":
        text = f"{b.total_gb:.0f} GB card, {b.other_gb:.1f} GB already in use"
    else:
        return "Unknown graphics memory"
    if b.whisper_gb:
        text += f", {b.whisper_gb:g} GB kept for TV transcription"
    return text
