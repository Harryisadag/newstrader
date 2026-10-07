# PyInstaller build recipe for NewsTrader.
#   Windows: build_exe.bat      -> dist\NewsTrader\NewsTrader.exe  (+ an _internal folder - keep them together)
#   Mac:     build_app.command  -> dist/NewsTrader.app
# -*- mode: python ; coding: utf-8 -*-

import glob
import importlib.util
import os
import platform
import re
import sys

MAC = sys.platform == "darwin"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', open("newstrader/__init__.py", encoding="utf-8").read()).group(1)

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
datas = [("newstrader/web", "newstrader/web")]
binaries = []
hiddenimports = []

# Packages that ship data files, DLLs or load modules dynamically.
for pkg in [
    "faster_whisper",   # Silero VAD model
    "ctranslate2",      # Whisper engine + its DLLs
    "onnxruntime",
    "av",
    "tokenizers",
    "yt_dlp",
    "yt_dlp_ejs",       # YouTube JavaScript helpers
    "deno",             # JavaScript runtime binary for yt-dlp
    "imageio_ffmpeg",   # bundled ffmpeg.exe
    "webview",          # pywebview + WebView2 loader DLLs
    "clr_loader",
    "pythonnet",
    "winotify",
    "tzdata",
    "alpaca",
    "anthropic",
    "feedparser",
    "sklearn",          # trains the local price model
    "huggingface_hub",  # downloads FinBERT / Whisper models
    "certifi",
    "mlx",              # Mac Apple-GPU speech-to-text (libmlx.dylib + mlx.metallib)
    "mlx_whisper",      # its mel filters + tokenizer files
    "tiktoken",
    "tiktoken_ext",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # package not installed on this machine
        print(f"[newstrader.spec] skipping {pkg}: {exc}")

hiddenimports += collect_submodules("uvicorn") + collect_submodules("newstrader")
hiddenimports += ["clr", "feedparser_sgmllib", "httpx2", "websockets", "websockets.legacy", "websockets.asyncio",
                  "tiktoken_ext.openai_public", "scipy.sparse", "sklearn.linear_model", "sklearn.metrics",
                  "sklearn.feature_extraction.text"]

# The deno executable is installed in the environment's Scripts folder, not inside the package.
try:
    import deno

    binaries.append((deno.find_deno_bin(), "deno_bin"))
except Exception as exc:
    print(f"[newstrader.spec] deno binary not bundled: {exc}")

# NVIDIA CUDA libraries installed by pip (cuBLAS etc.) -> _internal/nvidia/<lib>/bin
spec = importlib.util.find_spec("nvidia")
if spec and spec.submodule_search_locations:
    for root in spec.submodule_search_locations:
        for dll in glob.glob(os.path.join(root, "*", "bin", "*.dll")):
            lib = os.path.basename(os.path.dirname(os.path.dirname(dll)))
            binaries.append((dll, f"nvidia/{lib}/bin"))

a = Analysis(
    ["newstrader/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NewsTrader",
    icon="assets/icon.icns" if MAC else "assets/icon.ico",
    console=False,          # a normal windowed app (logs go to the Logs tab and the log files). On a Mac,
                            # console=True would hide the Dock icon.
    disable_windowed_traceback=False,
    upx=False,              # UPX-packed files trigger antivirus false alarms
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NewsTrader",
)

if MAC:
    app = BUNDLE(
        coll,
        name="NewsTrader.app",
        icon="assets/icon.icns",
        bundle_identifier="com.newstrader.app",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            # pip picked packages for the Mac doing the build (e.g. MLX ships separate macOS 14/15/26 builds), so
            # the oldest macOS the app supports is that Mac's major version. Release builds run on macOS 15.
            "LSMinimumSystemVersion": os.environ.get("NT_MACOS_MIN") or f"{(platform.mac_ver()[0] or '13').split('.')[0]}.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSRequiresAquaSystemAppearance": False,
            "LSApplicationCategoryType": "public.app-category.finance",
        },
    )
