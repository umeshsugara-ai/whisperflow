# whisperflow/ui/first_run.py
"""First-run speech-engine chooser — shown once, before the app's first
launch, whenever no config.toml exists yet (and the app isn't --headless).

Never writes config.toml silently: the user must either click "Use
recommended" or pick another provider from the list. Reuses the same
badge/row rendering as the Settings "Speech engine" section
(whisperflow.ui.engine_picker) so there's exactly one place that decides
what a provider row looks like.
"""

from __future__ import annotations

import os
import tkinter as tk
import webbrowser

from whisperflow.stt.providers import Provider, get

BG = "#161412"
CARD = "#1c1917"
FIELD = "#26221e"
FG = "#f2ede1"
FG_DIM = "#9a938a"
BTN = "#3a3733"
ACCENT_OK = "#5cb85c"

# Where to send a user who wants Local on a cloud-only install.
FULL_INSTALLER_URL = "https://github.com/umeshsugara-ai/whisperflow/releases/latest"


def provider_already_has_key(provider: Provider) -> bool:
    """True if the user already has this provider's API key in the
    environment (e.g. from a prior .env or a shell-level env var) — local
    never "has a key" since it doesn't need one."""
    if provider.kind == "local":
        return False
    return bool(os.environ.get(provider.api_key_env))


def show_first_run_chooser(root, specs, rec, path):
    """Blocking modal. Returns the Config the user confirmed — either the
    recommendation or a manual pick — already saved to `path`."""
    from whisperflow.config import set_env_var
    from whisperflow.stt.registry import local_inference_available
    from whisperflow.sysinfo import build_config_for_engine, build_recommended_config
    from whisperflow.ui.engine_picker import badge_line, build_rows

    local_ok = local_inference_available()

    win = tk.Toplevel(root)
    win.title("Welcome to WhisperFlow")
    win.geometry("560x520")
    win.configure(bg=BG)
    win.grab_set()  # modal — block interaction with anything else until resolved

    result: dict = {}

    tk.Label(
        win, text="Choose your speech engine", bg=BG, fg=FG, font=("Segoe UI", 14, "bold")
    ).pack(anchor="w", padx=18, pady=(16, 4))
    tk.Label(
        win, text="This decides where your dictation audio gets transcribed.",
        bg=BG, fg=FG_DIM, font=("Segoe UI", 9),
    ).pack(anchor="w", padx=18, pady=(0, 12))

    rec_provider = get(rec.engine)
    rec_frame = tk.Frame(win, bg=CARD, padx=14, pady=10)
    rec_frame.pack(fill="x", padx=18, pady=(0, 12))
    tk.Label(
        rec_frame, text=f"★ Recommended for your PC — {rec.reason}",
        bg=CARD, fg=FG, font=("Segoe UI", 9, "bold"), wraplength=480, justify="left",
    ).pack(anchor="w")
    tk.Label(
        rec_frame, text=badge_line(rec_provider), bg=CARD, fg=FG_DIM, font=("Segoe UI", 9),
    ).pack(anchor="w", pady=(2, 8))

    def _use_recommended() -> None:
        cfg = build_recommended_config(rec)
        cfg.path = path
        _finish(cfg)

    tk.Button(
        rec_frame, text=f"Use recommended ({rec_provider.display_name})", command=_use_recommended,
        bg=ACCENT_OK, fg="#0d1a0d", relief="flat", padx=10, pady=4, cursor="hand2",
    ).pack(anchor="w")

    tk.Label(win, text="Or pick another:", bg=BG, fg=FG_DIM, font=("Segoe UI", 9, "bold")).pack(
        anchor="w", padx=18, pady=(4, 4)
    )

    list_frame = tk.Frame(win, bg=BG)
    list_frame.pack(fill="both", expand=True, padx=18)

    def _pick(provider_id: str) -> None:
        provider = get(provider_id)
        if provider_already_has_key(provider) or provider.kind == "local":
            cfg = build_config_for_engine(provider_id, specs)
            cfg.path = path
            _finish(cfg)
            return
        _show_key_step(provider)

    for row in build_rows(recommended_id=rec.engine, local_available=local_ok):
        if row["id"] == rec.engine:
            continue  # already shown above as the recommended option
        r = tk.Frame(list_frame, bg=CARD, padx=10, pady=6)
        r.pack(fill="x", pady=(0, 6))
        tk.Label(r, text=row["display_name"], bg=CARD, fg=FG, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(r, text=row["badge"], bg=CARD, fg=FG_DIM, font=("Segoe UI", 8)).pack(anchor="w")
        if row["available"]:
            tk.Button(
                r, text="Choose", command=lambda pid=row["id"]: _pick(pid),
                bg=BTN, fg=FG, relief="flat", padx=8, cursor="hand2",
            ).pack(anchor="e")
        else:
            # Local on a cloud-only install: don't let the user pick a dead
            # end — explain and point them at the Full installer instead.
            tk.Label(
                r, text=row["unavailable_note"], bg=CARD, fg=ACCENT_OK,
                font=("Segoe UI", 8), wraplength=460, justify="left",
            ).pack(anchor="w", pady=(2, 0))
            tk.Button(
                r, text="Get the Full installer →",
                command=lambda: webbrowser.open(FULL_INSTALLER_URL),
                bg=BTN, fg=FG, relief="flat", padx=8, cursor="hand2",
            ).pack(anchor="e")

    def _show_key_step(provider: Provider) -> None:
        for child in list_frame.winfo_children():
            child.destroy()
        tk.Label(
            list_frame, text=f"Set up {provider.display_name}", bg=BG, fg=FG, font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(4, 6))
        tk.Button(
            list_frame, text="Get a free key →", command=lambda: webbrowser.open(provider.signup_url),
            bg=BTN, fg=FG, relief="flat", padx=8, cursor="hand2",
        ).pack(anchor="w")
        for i, step in enumerate(provider.setup_steps, start=1):
            tk.Label(
                list_frame, text=f"{i}. {step}", bg=BG, fg=FG_DIM, font=("Segoe UI", 9),
                wraplength=480, justify="left", anchor="w",
            ).pack(anchor="w", pady=(4 if i == 1 else 0, 0))
        key_row = tk.Frame(list_frame, bg=BG)
        key_row.pack(anchor="w", pady=(8, 0))
        key_var = tk.StringVar()
        tk.Entry(key_row, textvariable=key_var, show="•", width=36, bg=FIELD, fg=FG, insertbackground=FG).pack(
            side="left"
        )

        def _confirm_with_key() -> None:
            value = key_var.get().strip()
            if not value:
                return
            set_env_var(provider.api_key_env, value, path=path.parent / ".env")
            cfg = build_config_for_engine(provider.id, specs)
            cfg.path = path
            _finish(cfg)

        tk.Button(
            key_row, text="Save & continue", command=_confirm_with_key,
            bg=ACCENT_OK, fg="#0d1a0d", relief="flat", padx=8, cursor="hand2",
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            list_frame, text="I'll add a key later (use Local for now)",
            command=lambda: _finish_local(),
            bg=BG, fg=FG_DIM, relief="flat", cursor="hand2",
        ).pack(anchor="w", pady=(10, 0))

    def _finish_local() -> None:
        cfg = build_config_for_engine("local", specs)
        cfg.path = path
        _finish(cfg)

    def _give_up_local() -> None:
        # Escape hatch for a persistent save failure (e.g. a genuinely
        # unwritable config directory): bypass save_config entirely — it's
        # what's failing — and hand the caller an in-memory-only Config so
        # the app can still attempt to launch with sane defaults, even
        # though nothing was persisted to disk.
        from whisperflow.config import Config

        cfg = Config()
        cfg.model.engine = "local"
        cfg.path = path
        result["cfg"] = cfg
        win.grab_release()
        win.destroy()

    def _show_error(message: str) -> None:
        for child in list_frame.winfo_children():
            child.destroy()
        tk.Label(
            list_frame, text="Couldn't save your configuration", bg=BG, fg=FG,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(4, 6))
        tk.Label(
            list_frame, text=message, bg=BG, fg=FG_DIM, font=("Segoe UI", 9),
            wraplength=480, justify="left", anchor="w",
        ).pack(anchor="w")
        tk.Label(
            list_frame, text="Close this window to fall back to Local (on-device).",
            bg=BG, fg=FG_DIM, font=("Segoe UI", 9, "italic"), wraplength=480, justify="left",
        ).pack(anchor="w", pady=(8, 0))
        tk.Button(
            list_frame, text="Quit (use Local in-memory, nothing saved)", command=_give_up_local,
            bg=BTN, fg=FG, relief="flat", padx=8, pady=4, cursor="hand2",
        ).pack(anchor="w", pady=(10, 0))

    def _finish(cfg) -> None:
        from whisperflow.config import ConfigError, save_config

        try:
            save_config(cfg, path)
        except (ConfigError, OSError) as exc:
            # Never let this propagate out of a Tk button callback — that
            # unwinds silently (Tk swallows it) and leaves wait_window()
            # blocked forever with the window still up. Recover instead:
            # a keyless-cloud-provider validation failure (e.g. the
            # recommendation picked a provider whose key the user doesn't
            # actually have) falls through to the same key-entry step
            # already used for a manually-picked keyless provider. Anything
            # else (engine="local", or a real disk-I/O failure) shows an
            # error and leaves the window open — WM_DELETE_WINDOW already
            # falls back to local.
            if cfg.model.engine != "local":
                _show_key_step(get(cfg.model.engine))
            else:
                _show_error(str(exc))
            return
        result["cfg"] = cfg
        win.grab_release()
        win.destroy()

    # Closing the window (X button) must never leave the app unconfigured —
    # treat it exactly like "I'll add a key later": fall back to local.
    win.protocol("WM_DELETE_WINDOW", _finish_local)

    win.wait_window()
    return result["cfg"]
