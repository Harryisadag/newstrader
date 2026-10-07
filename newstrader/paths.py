"""Where things live on disk.

Dev mode (run.bat / python -m newstrader):
    data      -> <repo>/data/
    .env      -> <repo>/.env
Built app (PyInstaller):
    data      -> Windows: %LOCALAPPDATA%\\NewsTrader\\
                 Mac:     ~/Library/Application Support/NewsTrader/
    .env      -> Windows: next to NewsTrader.exe if one exists there, otherwise in the data folder
                 Mac:     in the data folder (files can't be kept inside a signed .app)

Both can be overridden with the NEWSTRADER_DATA_DIR / NEWSTRADER_ENV_FILE environment variables
(the tests use this).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_NAME


def is_frozen() -> bool:
    """True when running from the PyInstaller-built .exe / .app."""
    return bool(getattr(sys, "frozen", False))


def is_mac() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def user_data_base() -> Path:
    """The per-user folder where installed apps keep their data on this system."""
    if is_windows():
        return Path(os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local"))
    if is_mac():
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))


def package_dir() -> Path:
    """Folder containing the newstrader package (read-only resources like web/ live here)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "newstrader"
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def web_dir() -> Path:
    return package_dir() / "web"


def data_dir() -> Path:
    override = os.environ.get("NEWSTRADER_DATA_DIR")
    if override:
        path = Path(override)
    elif is_frozen():
        path = user_data_base() / APP_NAME
    else:
        path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_file() -> Path:
    override = os.environ.get("NEWSTRADER_ENV_FILE")
    if override:
        return Path(override)
    if is_frozen():
        if is_windows():
            beside_exe = Path(sys.executable).parent / ".env"
            if beside_exe.exists():
                return beside_exe
        return data_dir() / ".env"
    return project_root() / ".env"


def config_file() -> Path:
    return data_dir() / "config.json"


def db_file() -> Path:
    return data_dir() / "newstrader.db"


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exports_dir() -> Path:
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Whisper and sentiment models are downloaded here (~3 GB for large-v3)."""
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
