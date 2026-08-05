"""WhisperFlow — free, fully-local voice dictation for Windows 11.

Global hotkey -> mic -> faster-whisper (local GPU) -> text injected into the
focused app. Zero cloud, non-destructive optional cleanup, pluggable models.
"""

# The ONLY hand-edited version. The installer build reads it
# (scripts/build_installer.ps1 -> ISCC /DAppVersion), the release tag must be
# v<this>, and the auto-updater compares GitHub's latest tag against it.
__version__ = "1.0.4"
