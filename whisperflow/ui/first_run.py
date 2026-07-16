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
import threading
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


def provider_already_has_key(provider: Provider) -> bool:
    """True if the user already has this provider's API key in the
    environment (e.g. from a prior .env or a shell-level env var) — local
    never "has a key" since it doesn't need one."""
    if provider.kind == "local":
        return False
    return bool(os.environ.get(provider.api_key_env))


def fallback_engine(local_available: bool) -> str | None:
    """What to do when the user skips key entry or closes the chooser:
    fall back to "local" when this build can actually run it, or None on a
    cloud-only install — meaning save nothing and quit cleanly (the chooser
    reappears on next launch) instead of writing an engine="local" config
    that can only dead-end into the startup recovery loop."""
    return "local" if local_available else None


def show_first_run_chooser(root, specs, rec, path):
    """Blocking modal. Returns the Config the user confirmed — either the
    recommendation or a manual pick — already saved to `path`. Returns None
    when setup was deferred (cloud-only build, user closed without a key):
    nothing was saved and the caller should exit cleanly."""
    from whisperflow.config import set_env_var
    from whisperflow.stt.registry import local_inference_available
    from whisperflow.sysinfo import build_config_for_engine, build_recommended_config
    from whisperflow.ui.engine_picker import badge_line, build_rows

    local_ok = local_inference_available()

    win = tk.Toplevel(root)
    win.title("Welcome to WhisperFlow")
    win.geometry("560x560")
    win.minsize(480, 380)
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

    # The provider list (and the key-entry step that replaces it) can run
    # taller than the window — e.g. 4+ cloud providers each with a badge and
    # button. Wrap it in a scrollable canvas so nothing is ever cut off
    # un-reachably (the earlier fixed-height version hid Deepgram's row).
    canvas_area = tk.Frame(win, bg=BG)
    canvas_area.pack(fill="both", expand=True, padx=(18, 0), pady=(0, 12))
    list_canvas = tk.Canvas(canvas_area, bg=BG, highlightthickness=0)
    list_scrollbar = tk.Scrollbar(canvas_area, orient="vertical", command=list_canvas.yview)
    list_canvas.configure(yscrollcommand=list_scrollbar.set)
    list_canvas.pack(side="left", fill="both", expand=True)
    list_scrollbar.pack(side="right", fill="y", padx=(4, 18))

    list_frame = tk.Frame(list_canvas, bg=BG)
    list_frame_window = list_canvas.create_window((0, 0), window=list_frame, anchor="nw")
    list_frame.bind(
        "<Configure>", lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all"))
    )
    list_canvas.bind(
        "<Configure>", lambda e: list_canvas.itemconfig(list_frame_window, width=e.width)
    )

    def _on_mousewheel(event) -> None:
        list_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Scope the wheel binding to only-while-hovering-this-canvas — bind_all
    # would leak a global mousewheel handler into the rest of the app (main
    # window, Settings) that never gets cleaned up after this dialog closes.
    list_canvas.bind("<Enter>", lambda e: list_canvas.bind_all("<MouseWheel>", _on_mousewheel))
    list_canvas.bind("<Leave>", lambda e: list_canvas.unbind_all("<MouseWheel>"))

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
            # Local isn't available in this install — don't let the user
            # pick a dead end, just explain (no dangling call-to-action).
            tk.Label(
                r, text=row["unavailable_note"], bg=CARD, fg=FG_DIM,
                font=("Segoe UI", 8), wraplength=460, justify="left",
            ).pack(anchor="w", pady=(2, 0))

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
        key_status = tk.Label(
            list_frame, text="", bg=BG, fg=FG_DIM, font=("Segoe UI", 8),
            wraplength=480, justify="left",
        )

        def _confirm_with_key() -> None:
            value = key_var.get().strip()
            if not value:
                return
            # Validate with a real (0.3s-of-silence) request BEFORE saving —
            # a mistyped key must fail here, next to the field, not on the
            # user's first dictation.
            save_btn.config(state="disabled")
            key_status.config(text="Checking your key…", fg=FG_DIM)
            key_status.pack(anchor="w", pady=(4, 0))

            def worker() -> None:
                from whisperflow.stt.registry import verify_provider_key

                err = verify_provider_key(provider.id, value)

                def apply() -> None:
                    try:
                        if err is not None:
                            save_btn.config(state="normal")
                            key_status.config(text=f"✗ {err}", fg="#e5484d")
                            return
                        set_env_var(provider.api_key_env, value, path=path.parent / ".env")
                        cfg = build_config_for_engine(provider.id, specs)
                        cfg.path = path
                        _finish(cfg)
                    except tk.TclError:
                        pass  # window closed while the check ran

                win.after(0, apply)

            threading.Thread(target=worker, daemon=True).start()

        save_btn = tk.Button(
            key_row, text="Save & continue", command=_confirm_with_key,
            bg=ACCENT_OK, fg="#0d1a0d", relief="flat", padx=8, cursor="hand2",
        )
        save_btn.pack(side="left", padx=(6, 0))
        skip_label = (
            "I'll add a key later (use Local for now)"
            if local_ok
            else "I'll finish setup later (WhisperFlow will ask again next launch)"
        )
        tk.Button(
            list_frame, text=skip_label,
            command=lambda: _skip_setup(),
            bg=BG, fg=FG_DIM, relief="flat", cursor="hand2",
        ).pack(anchor="w", pady=(10, 0))

    def _finish_local() -> None:
        cfg = build_config_for_engine("local", specs)
        cfg.path = path
        _finish(cfg)

    def _finish_deferred() -> None:
        # Cloud-only build, no key yet: there is nothing runnable to save.
        # Save nothing — with no config.toml the chooser simply reappears on
        # the next launch — and let the caller quit cleanly.
        result["cfg"] = None
        win.grab_release()
        win.destroy()

    def _skip_setup() -> None:
        if fallback_engine(local_ok) == "local":
            _finish_local()
        else:
            _finish_deferred()

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

    # Closing the window (X button) is treated exactly like the skip button:
    # fall back to local when this build can run it, otherwise defer setup
    # (save nothing, caller quits cleanly, chooser reappears next launch).
    win.protocol("WM_DELETE_WINDOW", _skip_setup)

    win.wait_window()
    return result.get("cfg")
