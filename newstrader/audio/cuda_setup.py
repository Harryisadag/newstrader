"""Make the pip-installed NVIDIA libraries (cuBLAS etc.) findable on Windows.

`pip install nvidia-cublas-cu12` puts cublas64_12.dll in site-packages/nvidia/cublas/bin. Windows won't
look there by itself, so we add those folders to the DLL search path *before* CTranslate2 loads.
This is what lets you skip installing the CUDA Toolkit - you only need a recent NVIDIA driver.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_done = False
_added: list[str] = []


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        roots.append(base / "nvidia")
        roots.append(base)
    for entry in sys.path:
        p = Path(entry) / "nvidia"
        if p.is_dir():
            roots.append(p)
    return roots


def setup_cuda_dll_paths() -> list[str]:
    """Idempotent. Returns the folders that were added (empty on non-Windows)."""
    global _done
    if _done:
        return _added
    _done = True
    if sys.platform != "win32":
        return _added
    seen: set[str] = set()
    for root in _candidate_roots():
        if not root.is_dir():
            continue
        dirs = [root] if any(root.glob("cublas64_*.dll")) else []
        dirs += [d for d in root.glob("*/bin") if d.is_dir()]
        for d in dirs:
            key = str(d.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                os.add_dll_directory(str(d))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            _added.append(str(d))
    if _added:
        log.info("Added NVIDIA library folders to the DLL search path: %s", "; ".join(_added))
    return _added
