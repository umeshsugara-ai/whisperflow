"""Main application window — the Wispr-style "product screen".

One Toplevel with a left sidebar (Home / History / Dictionary / Settings) so a
non-technical user never needs the command line, config.toml, or raw logs:

- Home:       lifetime stats (total words, avg WPM, day streak, dictations),
              plain-language status strip, recent dictations
- History:    searchable list with RAW/INJECTED detail (HistoryPane)
- Dictionary: vocabulary + replacement rules, saved to config.toml
- Settings:   hotkey, language, cleanup tier, overlay, autostart — saved to
              config.toml and live-applied where possible

Closing the window only hides it (the app lives in the tray); quit stays in
the tray menu. Must be created on the tkinter main thread — the tray marshals
via root.after (see app.py on_open_main).
"""

from __future__ import annotations

import os
import time
import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Callable

from PIL import ImageTk

from whisperflow import sysinfo
from whisperflow.config import (
    Config,
    ConfigError,
    DictionaryConfig,
    Replacement,
    data_dir,
    save_config,
    set_env_var,
)
from whisperflow.history import History, average_wpm, compute_streak
from whisperflow.hotkey import format_hotkey_label
from whisperflow.stt import providers as _stt_providers
from whisperflow.ui import icons
from whisperflow.ui.engine_picker import badge_line
from whisperflow.ui.history_view import HistoryPane

BG = "#161412"
CARD = "#1c1917"
FIELD = "#26221e"
FG = "#f2ede1"
FG_DIM = "#9a938a"
BTN = "#3a3733"
ACCENT_OK = "#5cb85c"
ACCENT_WARN = "#f5a623"
ACCENT_ERR = "#e5484d"

HOTKEY_CHOICES = ("ctrl+windows", "alt+windows", "windows+space")
LANGUAGE_CHOICES = (
    ("Auto-detect", ""),
    ("English", "en"),
    ("Hindi (Devanagari)", "hi"),
    ("Hinglish (Roman Hindi + English)", "hinglish"),
)
TIER_CHOICES = (
    ("Off (verbatim)", "off"),
    ("Rules (fillers + punctuation)", "rules"),
    ("LLM (Ollama, local)", "llm"),
    ("LLM (Gemini cloud — text only)", "gemini"),
)


def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.1f}K"
    return f"{n:,}"


def humanize_ts(ts: str, today: str) -> str:
    """'2026-07-03T14:32:10' -> 'Today 14:32' / '07-02 14:32'."""
    if len(ts) < 16:
        return ts
    day, clock = ts[:10], ts[11:16]
    return f"Today {clock}" if day == today else f"{day[5:]} {clock}"


# First-run "How to use" card (Home screen). Dismissal is remembered in a
# marker file so the card never comes back after "Got it".
GUIDE_DISMISSED_FILE = ".guide_dismissed"


def guide_dismissed() -> bool:
    return (data_dir() / GUIDE_DISMISSED_FILE).exists()


def dismiss_guide() -> None:
    try:
        (data_dir() / GUIDE_DISMISSED_FILE).write_text("1", encoding="utf-8")
    except OSError:
        pass  # worst case the card shows again next launch


def guide_lines(hotkey_label: str) -> list[tuple[str, str]]:
    """(gesture, what it does) rows for the how-to-use card."""
    return [
        (f"Hold {hotkey_label}", "speak while holding, release — your words are typed"),
        (f"Tap {hotkey_label}", "hands-free: tap to start, speak freely, tap again to finish"),
        ("Esc", "cancel a recording (nothing is typed)"),
    ]


def _button(parent, text, command, **kw) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command, bg=BTN, fg=FG, relief="flat",
        padx=10, cursor="hand2", activebackground=FIELD, activeforeground=FG, **kw
    )


class MainWindow:
    _open_instance: "MainWindow | None" = None

    @classmethod
    def open(
        cls,
        root: tk.Tk,
        cfg: Config,
        history: History,
        apply_config: Callable[[], None],
        warnings_source: Callable[[], list[str]] | None = None,
        tab: str = "home",
    ) -> "MainWindow":
        """Single instance: re-open deiconifies, refreshes, and switches tab."""
        inst = cls._open_instance
        if inst is not None:
            try:
                inst.win.deiconify()
                inst.win.lift()
                inst.show_page(tab)
                return inst
            except tk.TclError:
                cls._open_instance = None
        inst = cls(root, cfg, history, apply_config, warnings_source)
        cls._open_instance = inst
        inst.show_page(tab)
        return inst

    def __init__(self, root, cfg, history, apply_config, warnings_source=None) -> None:
        self.cfg = cfg
        self.win = tk.Toplevel(root)
        self.win.title("WhisperFlow")
        self.win.geometry("920x580")
        self.win.minsize(760, 480)
        self.win.configure(bg=BG)
        # closing hides — the app lives in the tray; quit is tray-only
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        self._icon_ref = ImageTk.PhotoImage(icons.state_icon("idle"))
        self.win.iconphoto(False, self._icon_ref)

        def persist() -> None:
            save_config(cfg)
            apply_config()

        sidebar = tk.Frame(self.win, bg=CARD, width=170)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(
            sidebar, text="WhisperFlow", bg=CARD, fg=FG, font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=16, pady=(18, 2))
        tk.Label(
            sidebar, text="local dictation", bg=CARD, fg=FG_DIM, font=("Segoe UI", 8)
        ).pack(anchor="w", padx=16, pady=(0, 14))

        content = tk.Frame(self.win, bg=BG)
        content.pack(side="left", fill="both", expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        self._pages: dict[str, tk.Frame] = {
            "home": HomePage(
                content, history, warnings_source,
                open_history=lambda: self.show_page("history"), cfg=cfg,
            ),
            "history": HistoryPane(content, history),
            "dictionary": DictionaryPage(content, cfg, persist),
            "settings": SettingsPage(content, cfg, persist),
        }
        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self._nav: dict[str, tk.Button] = {}
        for key, label in (
            ("home", "  Home"),
            ("history", "  History"),
            ("dictionary", "  Dictionary"),
            ("settings", "  Settings"),
        ):
            btn = tk.Button(
                sidebar, text=label, anchor="w", relief="flat", bg=CARD, fg=FG,
                font=("Segoe UI", 10), padx=16, pady=6, bd=0, cursor="hand2",
                activebackground=BTN, activeforeground=FG,
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill="x")
            self._nav[key] = btn

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            key = "home"
        for k, btn in self._nav.items():
            btn.configure(bg=BTN if k == key else CARD)
        page = self._pages[key]
        refresh = getattr(page, "refresh", None)
        if refresh:
            refresh()
        page.tkraise()


class HomePage(tk.Frame):
    def __init__(self, parent, history: History, warnings_source, open_history, cfg: Config | None = None) -> None:
        super().__init__(parent, bg=BG)
        self.history = history
        self.warnings_source = warnings_source or (lambda: [])

        tk.Label(self, text="Home", bg=BG, fg=FG, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=16, pady=(14, 2)
        )
        self._status = tk.Label(self, text="", bg=BG, fg=ACCENT_OK, font=("Segoe UI", 9), cursor="hand2")
        self._status.pack(anchor="w", padx=16, pady=(0, 8))
        self._status.bind("<Button-1>", lambda e: self._show_warnings())

        if cfg is not None and not guide_dismissed():
            self._build_guide_card(format_hotkey_label(cfg.hotkey.combo))

        cards = tk.Frame(self, bg=BG)
        cards.pack(fill="x", padx=16)
        self._cards: dict[str, tk.Label] = {}
        for i, (key, caption) in enumerate(
            (
                ("words", "Total words"),
                ("wpm", "Avg WPM"),
                ("streak", "Day streak"),
                ("dictations", "Dictations"),
            )
        ):
            card = tk.Frame(cards, bg=CARD, padx=18, pady=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 10))
            cards.columnconfigure(i, weight=1)
            value = tk.Label(card, text="—", bg=CARD, fg=FG, font=("Segoe UI", 20, "bold"))
            value.pack(anchor="w")
            tk.Label(card, text=caption, bg=CARD, fg=FG_DIM, font=("Segoe UI", 9)).pack(anchor="w")
            self._cards[key] = value

        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=16, pady=(16, 4))
        tk.Label(head, text="Recent", bg=BG, fg=FG_DIM, font=("Segoe UI", 10, "bold")).pack(side="left")
        link = tk.Label(head, text="View all →", bg=BG, fg=ACCENT_OK, font=("Segoe UI", 9), cursor="hand2")
        link.pack(side="right")
        link.bind("<Button-1>", lambda e: open_history())

        self._recent = tk.Frame(self, bg=BG)
        self._recent.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def refresh(self) -> None:
        stats = self.history.stats()
        today = time.strftime("%Y-%m-%d")
        self._cards["words"].config(text=format_count(stats["total_words"]))
        self._cards["wpm"].config(text=f"{average_wpm(stats):.0f}")
        self._cards["streak"].config(text=str(compute_streak(stats["days"], today)))
        self._cards["dictations"].config(text=format_count(stats["total_dictations"]))

        warnings = self.warnings_source()
        if warnings:
            self._status.config(
                text=f"⚠ {len(warnings)} warning{'s' if len(warnings) > 1 else ''} — click to view",
                fg=ACCENT_WARN,
            )
        else:
            self._status.config(text="All good ✓", fg=ACCENT_OK)

        for child in self._recent.winfo_children():
            child.destroy()
        entries = list(reversed(self.history.entries(limit=5)))
        if not entries:
            tk.Label(
                self._recent, text="No dictations yet — press your hotkey and speak.",
                bg=BG, fg=FG_DIM, font=("Segoe UI", 9),
            ).pack(anchor="w", pady=8)
            return
        for e in entries:
            row = tk.Frame(self._recent, bg=CARD, padx=10, pady=6)
            row.pack(fill="x", pady=(0, 6))
            tk.Label(
                row, text=humanize_ts(e.get("ts", ""), today), bg=CARD, fg=FG_DIM,
                font=("Segoe UI", 8), width=12, anchor="w",
            ).pack(side="left")
            preview = e.get("injected", "").replace("\n", " ")
            preview = preview[:90] + ("…" if len(preview) > 90 else "")
            tk.Label(
                row, text=preview, bg=CARD, fg=FG, font=("Segoe UI", 9), anchor="w"
            ).pack(side="left", fill="x", expand=True)
            _button(row, "Copy", lambda text=e.get("injected", ""): self._copy(text), pady=0).pack(side="right")

    def _build_guide_card(self, hotkey_label: str) -> None:
        card = tk.Frame(self, bg=CARD, padx=14, pady=10)
        card.pack(fill="x", padx=16, pady=(0, 10))
        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x")
        tk.Label(
            head, text="👋 How to dictate", bg=CARD, fg=FG, font=("Segoe UI", 10, "bold")
        ).pack(side="left")

        def _dismiss() -> None:
            dismiss_guide()
            card.destroy()

        _button(head, "Got it", _dismiss, pady=0).pack(side="right")
        for gesture, what in guide_lines(hotkey_label):
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=(4, 0))
            tk.Label(
                row, text=gesture, bg=FIELD, fg=FG, font=("Segoe UI", 9, "bold"), padx=6, pady=1
            ).pack(side="left")
            tk.Label(
                row, text="  " + what, bg=CARD, fg=FG_DIM, font=("Segoe UI", 9), anchor="w"
            ).pack(side="left", fill="x", expand=True)

    def _copy(self, text: str) -> None:
        from whisperflow.inject import clipboard

        clipboard.write_text(text)

    def _show_warnings(self) -> None:
        warnings = self.warnings_source()
        if not warnings:
            return
        win = tk.Toplevel(self)
        win.title("WhisperFlow — recent warnings")
        win.geometry("560x300")
        win.configure(bg=BG)
        box = tk.Text(win, bg=FIELD, fg=FG, wrap="word", font=("Segoe UI", 9), relief="flat")
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("1.0", "\n\n".join(warnings))
        box.configure(state="disabled")


class DictionaryPage(tk.Frame):
    def __init__(self, parent, cfg: Config, persist: Callable[[], None]) -> None:
        super().__init__(parent, bg=BG)
        self.cfg = cfg
        self.persist = persist

        tk.Label(self, text="Dictionary", bg=BG, fg=FG, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=16, pady=(14, 2)
        )
        tk.Label(
            self,
            text="Vocabulary teaches recognition your words; replacements fix persistent mis-hearings.",
            bg=BG, fg=FG_DIM, font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(0, 10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(1, weight=1)

        # --- vocabulary (left) ---
        tk.Label(body, text="Vocabulary", bg=BG, fg=FG_DIM, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        vocab_frame = tk.Frame(body, bg=BG)
        vocab_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        vocab_wrap = tk.Frame(vocab_frame, bg=BG)
        vocab_wrap.pack(fill="both", expand=True)
        self.vocab_list = tk.Listbox(
            vocab_wrap, bg=FIELD, fg=FG, relief="flat", font=("Segoe UI", 9),
            selectbackground=BTN, selectforeground=FG, highlightthickness=0,
        )
        vocab_sb = ttk.Scrollbar(
            vocab_wrap, orient="vertical", command=self.vocab_list.yview, style="WF.Vertical.TScrollbar"
        )
        self.vocab_list.configure(yscrollcommand=vocab_sb.set)
        self.vocab_list.pack(side="left", fill="both", expand=True)
        vocab_sb.pack(side="right", fill="y")
        vrow = tk.Frame(vocab_frame, bg=BG)
        vrow.pack(fill="x", pady=(6, 0))
        self.vocab_entry = tk.Entry(vrow, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", font=("Segoe UI", 9))
        self.vocab_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self.vocab_entry.bind("<Return>", lambda e: self._add_vocab())
        _button(vrow, "Add", self._add_vocab).pack(side="left", padx=(6, 0))
        _button(vrow, "Remove", self._remove_vocab).pack(side="left", padx=(6, 0))

        # --- replacements (right) ---
        tk.Label(body, text="Replacements", bg=BG, fg=FG_DIM, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w"
        )
        rep_frame = tk.Frame(body, bg=BG)
        rep_frame.grid(row=1, column=1, sticky="nsew")
        rep_wrap = tk.Frame(rep_frame, bg=BG)
        rep_wrap.pack(fill="both", expand=True)
        self.rep_tree = ttk.Treeview(
            rep_wrap, columns=("from", "to"), show="headings", style="WF.Treeview", height=8
        )
        self.rep_tree.heading("from", text="When I say")
        self.rep_tree.heading("to", text="Write instead")
        rep_sb = ttk.Scrollbar(
            rep_wrap, orient="vertical", command=self.rep_tree.yview, style="WF.Vertical.TScrollbar"
        )
        self.rep_tree.configure(yscrollcommand=rep_sb.set)
        self.rep_tree.pack(side="left", fill="both", expand=True)
        rep_sb.pack(side="right", fill="y")
        self.rep_tree.bind("<Double-1>", lambda e: self._load_selected_rep())
        rrow = tk.Frame(rep_frame, bg=BG)
        rrow.pack(fill="x", pady=(6, 0))
        self.rep_from = tk.Entry(rrow, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", font=("Segoe UI", 9))
        self.rep_from.pack(side="left", fill="x", expand=True, ipady=3)
        tk.Label(rrow, text="→", bg=BG, fg=FG_DIM).pack(side="left", padx=4)
        self.rep_to = tk.Entry(rrow, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", font=("Segoe UI", 9))
        self.rep_to.pack(side="left", fill="x", expand=True, ipady=3)
        _button(rrow, "Add", self._add_rep).pack(side="left", padx=(6, 0))
        _button(rrow, "Remove", self._remove_rep).pack(side="left", padx=(6, 0))

        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=16, pady=12)
        _button(foot, "Save", self._save).pack(side="left")
        self._status = tk.Label(foot, text="", bg=BG, fg=ACCENT_OK, font=("Segoe UI", 9))
        self._status.pack(side="left", padx=10)
        tk.Label(
            foot, text="New words improve recognition from your next dictation.",
            bg=BG, fg=FG_DIM, font=("Segoe UI", 8),
        ).pack(side="right")

    def refresh(self) -> None:
        self.vocab_list.delete(0, "end")
        for word in self.cfg.dictionary.vocabulary:
            self.vocab_list.insert("end", word)
        self.rep_tree.delete(*self.rep_tree.get_children())
        for r in self.cfg.dictionary.replacements:
            self.rep_tree.insert("", "end", values=(r.from_, r.to))

    def _add_vocab(self) -> None:
        word = self.vocab_entry.get().strip()
        if word and word not in self.vocab_list.get(0, "end"):
            self.vocab_list.insert("end", word)
        self.vocab_entry.delete(0, "end")

    def _remove_vocab(self) -> None:
        for i in reversed(self.vocab_list.curselection()):
            self.vocab_list.delete(i)

    def _add_rep(self) -> None:
        src, dst = self.rep_from.get().strip(), self.rep_to.get().strip()
        if not src or not dst:
            return
        for iid in self.rep_tree.get_children():  # editing an existing rule replaces it
            if self.rep_tree.item(iid, "values")[0] == src:
                self.rep_tree.delete(iid)
        self.rep_tree.insert("", "end", values=(src, dst))
        self.rep_from.delete(0, "end")
        self.rep_to.delete(0, "end")

    def _remove_rep(self) -> None:
        for iid in self.rep_tree.selection():
            self.rep_tree.delete(iid)

    def _load_selected_rep(self) -> None:
        sel = self.rep_tree.selection()
        if not sel:
            return
        src, dst = self.rep_tree.item(sel[0], "values")
        self.rep_from.delete(0, "end")
        self.rep_from.insert(0, src)
        self.rep_to.delete(0, "end")
        self.rep_to.insert(0, dst)

    def _save(self) -> None:
        self.cfg.dictionary = DictionaryConfig(
            vocabulary=list(self.vocab_list.get(0, "end")),
            replacements=[
                Replacement(from_=v[0], to=v[1])
                for iid in self.rep_tree.get_children()
                if (v := self.rep_tree.item(iid, "values"))
            ],
        )
        try:
            self.persist()
        except (ConfigError, OSError) as exc:
            self._status.config(text=str(exc), fg=ACCENT_ERR)
            return
        self._status.config(text="Saved ✓", fg=ACCENT_OK)
        self.after(1500, lambda: self._status.config(text=""))


class SettingsPage(tk.Frame):
    def __init__(self, parent, cfg: Config, persist: Callable[[], None]) -> None:
        super().__init__(parent, bg=BG)
        self.cfg = cfg
        self.persist = persist
        # values at app launch — changing these needs a restart to take effect
        self._launch_combo = cfg.hotkey.combo
        self._launch_language = cfg.model.language
        self._launch_engine = cfg.model.engine

        tk.Label(self, text="Settings", bg=BG, fg=FG, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=16, pady=(14, 10)
        )

        style = ttk.Style(self)
        style.configure("WF.TCombobox", fieldbackground=FIELD, background=BTN, foreground=FG)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="x", padx=16)
        body.columnconfigure(1, weight=1)

        def row(r: int, label: str, caption: str = "") -> tk.Frame:
            tk.Label(body, text=label, bg=BG, fg=FG, font=("Segoe UI", 10)).grid(
                row=r, column=0, sticky="nw", pady=(0, 2)
            )
            holder = tk.Frame(body, bg=BG)
            holder.grid(row=r, column=1, sticky="w", padx=(16, 0))
            if caption:
                tk.Label(body, text=caption, bg=BG, fg=FG_DIM, font=("Segoe UI", 8)).grid(
                    row=r + 1, column=1, sticky="w", padx=(16, 0), pady=(0, 10)
                )
            else:
                tk.Frame(body, bg=BG, height=10).grid(row=r + 1, column=0)
            return holder

        combos = list(HOTKEY_CHOICES)
        if cfg.hotkey.combo not in combos:
            combos.insert(0, cfg.hotkey.combo)
        self.hotkey_var = tk.StringVar()
        ttk.Combobox(
            row(0, "Hotkey", "Takes effect after restart. Avoid alt+space (Windows system menu)."),
            textvariable=self.hotkey_var, values=combos, state="readonly", width=24,
        ).pack(anchor="w")

        self.language_var = tk.StringVar()
        ttk.Combobox(
            row(2, "Language", "Takes effect after restart. Hinglish = Roman-script Hindi + English mix."),
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_CHOICES],
            state="readonly", width=32,
        ).pack(anchor="w")

        self.tier_var = tk.StringVar()
        tier_holder = row(4, "Cleanup", "Applies immediately. Raw transcript is always kept in History.")
        for label, value in TIER_CHOICES:
            tk.Radiobutton(
                tier_holder, text=label, variable=self.tier_var, value=value,
                bg=BG, fg=FG, selectcolor=FIELD, activebackground=BG, activeforeground=FG,
                font=("Segoe UI", 9), anchor="w",
            ).pack(anchor="w")

        overlay_holder = row(6, "Overlay pill", "Applies immediately.")
        self.overlay_var = tk.BooleanVar()
        self.hint_var = tk.BooleanVar()
        for text, var in (
            ("Always show the pill at the bottom of the screen", self.overlay_var),
            ("Show the hotkey hint at startup", self.hint_var),
        ):
            tk.Checkbutton(
                overlay_holder, text=text, variable=var, bg=BG, fg=FG, selectcolor=FIELD,
                activebackground=BG, activeforeground=FG, font=("Segoe UI", 9),
            ).pack(anchor="w")

        autostart_holder = row(8, "Startup", "Applies immediately (Windows login entry).")
        self.autostart_var = tk.BooleanVar()
        tk.Checkbutton(
            autostart_holder, text="Start WhisperFlow when I sign in to Windows",
            variable=self.autostart_var, command=self._toggle_autostart,
            bg=BG, fg=FG, selectcolor=FIELD, activebackground=BG, activeforeground=FG,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self._engine_recommended_id = None  # set lazily in refresh() via sysinfo.recommend()
        self._local_available = True  # set lazily in refresh() — False on a cloud-only build
        engine_holder = row(10, "Speech engine", "Changing engine takes effect after restart.")
        self.engine_var = tk.StringVar()
        self._engine_combo = ttk.Combobox(
            engine_holder, textvariable=self.engine_var,
            values=[p.id for p in _stt_providers.all_providers()],
            state="readonly", width=32, style="WF.TCombobox",
        )
        self._engine_combo.pack(anchor="w")
        self._engine_combo.bind("<<ComboboxSelected>>", lambda e: self._on_engine_picked())

        self._engine_badge = tk.Label(
            engine_holder, text="", bg=BG, fg=FG_DIM, font=("Segoe UI", 8), wraplength=520, justify="left"
        )
        self._engine_badge.pack(anchor="w", pady=(2, 0))

        self._engine_key_frame = tk.Frame(engine_holder, bg=BG)
        self._engine_key_frame.pack(anchor="w", fill="x", pady=(6, 0))

        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=16, pady=(14, 4))
        _button(foot, "Save", self._save).pack(side="left")
        self._status = tk.Label(foot, text="", bg=BG, fg=ACCENT_OK, font=("Segoe UI", 9))
        self._status.pack(side="left", padx=10)

        self._banner = tk.Label(
            self, text="", bg=BG, fg=ACCENT_WARN, font=("Segoe UI", 9), wraplength=640, justify="left"
        )
        self._banner.pack(anchor="w", padx=16)

        adv = tk.Frame(self, bg=BG)
        adv.pack(side="bottom", fill="x", padx=16, pady=12)
        self._model_line = tk.Label(adv, text="", bg=BG, fg=FG_DIM, font=("Segoe UI", 8))
        self._model_line.pack(side="left")
        _button(adv, "Open logs folder", self._open_logs).pack(side="right", padx=(6, 0))
        _button(adv, "Open config file", lambda: os.startfile(str(self.cfg.path))).pack(side="right")  # noqa: S606

    def refresh(self) -> None:
        self.hotkey_var.set(self.cfg.hotkey.combo)
        label = next((lb for lb, v in LANGUAGE_CHOICES if v == self.cfg.model.language), LANGUAGE_CHOICES[0][0])
        self.language_var.set(label)
        self.tier_var.set(self.cfg.cleanup.tier)
        self.overlay_var.set(self.cfg.overlay.always_visible)
        self.hint_var.set(self.cfg.overlay.show_hint)
        self.autostart_var.set(sysinfo.is_autostart_enabled())
        m = self.cfg.model
        model_name = m.cloud_model if m.engine != "local" else m.name
        self._model_line.config(
            text=f"Model: {model_name} · engine {m.engine}"
            + (f" on {m.device}" if m.engine == "local" else "")
            + "  —  advanced settings live in the config file"
        )
        self._update_banner()
        if self._engine_recommended_id is None:
            from whisperflow.stt import registry

            self._local_available = registry.local_inference_available()
            specs = sysinfo.probe()
            has_key = any(
                os.environ.get(p.api_key_env) for p in _stt_providers.cloud_providers() if p.api_key_env
            )
            self._engine_recommended_id = sysinfo.recommend(
                specs, has_api_key=has_key, local_available=self._local_available
            ).engine
        self.engine_var.set(self.cfg.model.engine)
        self._on_engine_picked()

    def _selected_language(self) -> str:
        label = self.language_var.get()
        return next((v for lb, v in LANGUAGE_CHOICES if lb == label), "")

    def _on_engine_picked(self) -> None:
        engine_id = self.engine_var.get()
        provider = _stt_providers.get(engine_id)
        star = "★ Recommended for your PC — " if engine_id == self._engine_recommended_id else ""
        self._engine_badge.config(text=f"{star}{badge_line(provider)}")
        self._render_key_entry(provider)

    def _render_key_entry(self, provider) -> None:
        for child in self._engine_key_frame.winfo_children():
            child.destroy()
        if provider.kind == "local":
            if not self._local_available:
                from whisperflow.ui.engine_picker import LOCAL_UNAVAILABLE_NOTE

                tk.Label(
                    self._engine_key_frame, text=LOCAL_UNAVAILABLE_NOTE, bg=BG, fg=ACCENT_WARN,
                    font=("Segoe UI", 9), wraplength=520, justify="left",
                ).pack(anchor="w")
            return
        already_set = bool(os.environ.get(provider.api_key_env))
        if already_set:
            tk.Label(
                self._engine_key_frame, text=f"✓ {provider.api_key_env} is set",
                bg=BG, fg=ACCENT_OK, font=("Segoe UI", 9),
            ).pack(anchor="w")
            return
        _button(
            self._engine_key_frame, "Get a free key →",
            lambda: webbrowser.open(provider.signup_url),
        ).pack(anchor="w")
        for i, step in enumerate(provider.setup_steps, start=1):
            tk.Label(
                self._engine_key_frame, text=f"{i}. {step}", bg=BG, fg=FG_DIM,
                font=("Segoe UI", 8), wraplength=520, justify="left", anchor="w",
            ).pack(anchor="w", pady=(4 if i == 1 else 0, 0))
        key_row = tk.Frame(self._engine_key_frame, bg=BG)
        key_row.pack(anchor="w", pady=(6, 0))
        key_var = tk.StringVar()
        entry = tk.Entry(key_row, textvariable=key_var, show="•", width=40, bg=FIELD, fg=FG, insertbackground=FG)
        entry.pack(side="left")

        def _save_key() -> None:
            value = key_var.get().strip()
            if not value:
                return
            set_env_var(provider.api_key_env, value, path=self.cfg.path.parent / ".env")
            self._render_key_entry(provider)  # re-render to show "✓ ... is set"

        _button(key_row, "Save key", _save_key).pack(side="left", padx=(6, 0))

    def _toggle_autostart(self) -> None:
        try:
            if self.autostart_var.get():
                sysinfo.enable_autostart()
            else:
                sysinfo.disable_autostart()
        except OSError as exc:
            self._status.config(text=f"Autostart failed: {exc}", fg=ACCENT_ERR)

    def _update_banner(self) -> None:
        pending = []
        if self.cfg.hotkey.combo != self._launch_combo:
            pending.append(f"hotkey ({format_hotkey_label(self.cfg.hotkey.combo)})")
        if self.cfg.model.language != self._launch_language:
            pending.append("language")
        if self.cfg.model.engine != self._launch_engine:
            pending.append("speech engine")
        if pending:
            self._banner.config(
                text="⚠ Restart WhisperFlow to apply the new "
                + " and ".join(pending)
                + " (tray → Quit, then reopen)."
            )
        else:
            self._banner.config(text="")

    def _save(self) -> None:
        old = (self.cfg.hotkey.combo, self.cfg.model.language, self.cfg.cleanup.tier,
               self.cfg.overlay.always_visible, self.cfg.overlay.show_hint, self.cfg.model.engine,
               self.cfg.model.cloud_model, self.cfg.model.api_key_env, self.cfg.model.name,
               self.cfg.model.device, self.cfg.model.compute_type)
        self.cfg.hotkey.combo = self.hotkey_var.get()
        self.cfg.model.language = self._selected_language()
        self.cfg.cleanup.tier = self.tier_var.get()
        self.cfg.overlay.always_visible = self.overlay_var.get()
        self.cfg.overlay.show_hint = self.hint_var.get()
        new_engine = self.engine_var.get()
        if new_engine == "local" and not self._local_available:
            # Cloud-only build — don't save an engine that can't run here.
            self._status.config(
                text="Local isn't available in this install — pick a free cloud engine instead.",
                fg=ACCENT_ERR,
            )
            return
        if new_engine != self.cfg.model.engine:
            # Engine actually changed: rebuild cloud_model/api_key_env (and
            # local name/device/compute_type) from the registry/hardware —
            # same as first_run.py's build_config_for_engine — so we never
            # save a stale cloud_model/api_key_env pointing at the previous
            # provider (e.g. engine="groq" with a leftover Gemini model id
            # and GEMINI_API_KEY, which silently posts the wrong model to
            # the wrong API and reads the wrong key).
            rebuilt = sysinfo.build_config_for_engine(new_engine, sysinfo.probe())
            self.cfg.model.engine = new_engine
            self.cfg.model.cloud_model = rebuilt.model.cloud_model
            self.cfg.model.api_key_env = rebuilt.model.api_key_env
            self.cfg.model.name = rebuilt.model.name
            self.cfg.model.device = rebuilt.model.device
            self.cfg.model.compute_type = rebuilt.model.compute_type
        try:
            self.persist()
        except (ConfigError, OSError) as exc:
            (self.cfg.hotkey.combo, self.cfg.model.language, self.cfg.cleanup.tier,
             self.cfg.overlay.always_visible, self.cfg.overlay.show_hint, self.cfg.model.engine,
             self.cfg.model.cloud_model, self.cfg.model.api_key_env, self.cfg.model.name,
             self.cfg.model.device, self.cfg.model.compute_type) = old
            self._status.config(text=str(exc), fg=ACCENT_ERR)
            return
        self._status.config(text="Saved ✓", fg=ACCENT_OK)
        self.after(1500, lambda: self._status.config(text=""))
        self._update_banner()

    def _open_logs(self) -> None:
        logs = self.cfg.path.parent / "logs"
        if logs.exists():
            os.startfile(str(logs))  # noqa: S606
