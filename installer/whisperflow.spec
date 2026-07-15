# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the installed (frozen) WhisperFlow build.
# Build:  python -m PyInstaller installer/whisperflow.spec --noconfirm
# Output: dist/WhisperFlow/WhisperFlow.exe  (onedir — faster startup than
# onefile and DLL problems are debuggable by looking in the folder)

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

REPO = os.path.dirname(SPECPATH)  # spec lives in installer/, repo is one up

datas = [(os.path.join(REPO, "assets", "app.ico"), "assets")]
# faster-whisper bundles the silero VAD onnx model as package data
datas += collect_data_files("faster_whisper")

binaries = []
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")

# CUDA runtime for ctranslate2 GPU inference (cublas + cudnn). These come from
# the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip wheels in the build venv —
# without them the frozen app dies with "cublas64_12.dll is not found".
# On CPU-only machines the DLLs are simply never loaded.
# Skipped (verified unused by ctranslate2's whisper pipeline on 2026-07-15 —
# smoke test reaches encode/decode fine without them; saves ~865MB):
#   cudnn_engines_precompiled (graph-API fusion engines), cudnn_adv (RNN ops),
#   *.alt.dll (old-driver nvrtc variant), nvblas (BLAS drop-in shim)
_CUDA_SKIP = ("cudnn_engines_precompiled", "cudnn_adv", ".alt.", "nvblas")
try:
    import glob

    import nvidia

    for _base in nvidia.__path__:
        for _dll in glob.glob(os.path.join(_base, "*", "bin", "*.dll")):
            if not any(s in os.path.basename(_dll) for s in _CUDA_SKIP):
                binaries.append((_dll, "."))
except ImportError:
    print("WARNING: nvidia CUDA wheels not installed — frozen build will be CPU-only")

hiddenimports = [
    "pystray._win32",
    "PIL._tkinter_finder",
    "win32timezone",
]

a = Analysis(
    [os.path.join(REPO, "app.py")],
    pathex=[REPO],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest"],
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
