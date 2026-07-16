# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the installed (frozen) WhisperFlow build.
# Build:  python -m PyInstaller installer/whisperflow.spec --noconfirm
# Output: dist/WhisperFlow/WhisperFlow.exe  (onedir — faster startup than
# onefile and DLL problems are debuggable by looking in the folder)
#
# The distributed build is cloud-only (Groq / Gemini / OpenAI / Deepgram /
# NVIDIA over plain HTTPS): local inference deps (faster_whisper,
# ctranslate2, CUDA) are excluded, keeping the installer ~29MB. Local
# (on-device) mode works only when running from source.

import os

REPO = os.path.dirname(SPECPATH)  # spec lives in installer/, repo is one up

datas = [(os.path.join(REPO, "assets", "app.ico"), "assets")]

hiddenimports = [
    "pystray._win32",
    "PIL._tkinter_finder",
    "win32timezone",
    # registry.py dispatches these via importlib.import_module() with a
    # dynamically-built module path (kind -> "whisperflow.stt.X_engine") so
    # PyInstaller's static import scanner can't discover them on its own —
    # without this, picking e.g. Groq/OpenAI in the frozen app crashes with
    # "ModuleNotFoundError: No module named 'whisperflow.stt...'" the first
    # time create_engine() is called for that provider.
    "whisperflow.stt.gemini_engine",
    "whisperflow.stt.openai_compatible_engine",
    "whisperflow.stt.deepgram_engine",
    "whisperflow.stt.nvidia_engine",
    "whisperflow.stt.faster_whisper_engine",
]

a = Analysis(
    [os.path.join(REPO, "app.py")],
    pathex=[REPO],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "pytest",
        # local-inference deps — never bundled in the distributed build
        "ctranslate2", "faster_whisper", "onnxruntime", "nvidia",
        "torch", "tokenizers",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperFlow",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed — no console flash; logs go to %LOCALAPPDATA%\WhisperFlow\logs
    icon=os.path.join(REPO, "assets", "app.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="WhisperFlow",
)
