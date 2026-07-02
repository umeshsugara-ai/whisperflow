"""History viewer — a proper window over history.jsonl (Wispr-style).

Lists recent dictations newest-first; selecting a row shows the full RAW and
INJECTED texts side by side with one-click copy buttons, so a dictation that
landed in the wrong window (or nowhere) is never lost.

Opened from the tray; must be created on the tkinter main thread.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from whisperflow.history import History

BG = "#1c1917"
FG = "#f2ede1"
FG_DIM = "#9a938a"
ACCENT = "#5cb85c"


class HistoryViewer:
    _open_instance: "HistoryViewer | None" = None

    def __init__(self, root: tk.Tk, history: History) -> None:
        # single instance — re-open just refreshes and raises
        if HistoryViewer._open_instance is not None:
            try:
                HistoryViewer._open_instance.refresh()
                HistoryViewer._open_instance.win.deiconify()
                HistoryViewer._open_instance.win.lift()
                return
            except tk.TclError:
                pass  # prior window was destroyed

        self.history = history
        self.win = tk.Toplevel(root)
        self.win.title("WhisperFlow — History")
        self.win.geometry("640x420")
        self.win.configure(bg=BG)
        HistoryViewer._open_instance = self
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        style = ttk.Style(self.win)
        style.theme_use("clam")
        style.configure(
            "WF.Treeview", background=BG, foreground=FG, fieldbackground=BG, rowheight=24, borderwidth=0
        )
        style.map("WF.Treeview", background=[("selected", "#3a3733")])

        columns = ("time", "tier", "text")
        self.tree = ttk.Treeview(self.win, columns=columns, show="headings", style="WF.Treeview", height=8)
        self.tree.heading("time", text="Time")
        self.tree.heading("tier", text="Cleanup")
        self.tree.heading("text", text="Injected text")
        self.tree.column("time", width=130, stretch=False)
        self.tree.column("tier", width=70, stretch=False)
        self.tree.column("text", width=420)
        self.tree.pack(fill="both", expand=False, padx=8, pady=(8, 4))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # detail panes
        detail = tk.Frame(self.win, bg=BG)
        detail.pack(fill="both", expand=True, padx=8, pady=4)

        for col, (label, attr) in enumerate((("RAW (exactly what you said)", "raw_box"), ("INJECTED (after cleanup)", "inj_box"))):
            frame = tk.Frame(detail, bg=BG)
            frame.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else 0)
            detail.columnconfigure(col, weight=1)
            detail.rowconfigure(0, weight=1)
            tk.Label(frame, text=label, bg=BG, fg=FG_DIM, font=("Segoe UI", 8)).pack(anchor="w")
            box = tk.Text(frame, height=6, bg="#26221e", fg=FG, wrap="word", font=("Segoe UI", 9), relief="flat")
            box.pack(fill="both", expand=True)
            setattr(self, attr, box)

        # buttons
        btns = tk.Frame(self.win, bg=BG)
        btns.pack(fill="x", padx=8, pady=(2, 8))
        self._status = tk.Label(btns, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 8))
        self._status.pack(side="right")
        for text, cmd in (
            ("Copy RAW", lambda: self._copy("raw")),
            ("Copy injected", lambda: self._copy("injected")),
            ("Refresh", self.refresh),
        ):
            tk.Button(
                btns, text=text, command=cmd, bg="#3a3733", fg=FG, relief="flat", padx=10, cursor="hand2"
            ).pack(side="left", padx=(0, 6))

        self._entries: list[dict] = []
        self.refresh()

    def _close(self) -> None:
        HistoryViewer._open_instance = None
        self.win.destroy()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._entries = list(reversed(self.history.entries(limit=200)))  # newest first
        for i, e in enumerate(self._entries):
            preview = e.get("injected", "").replace("\n", " ")
            preview = preview[:70] + ("…" if len(preview) > 70 else "")
            self.tree.insert("", "end", iid=str(i), values=(e.get("ts", ""), e.get("tier", ""), preview))
        if self._entries:
            self.tree.selection_set("0")

    def _selected(self) -> dict | None:
        sel = self.tree.selection()
        return self._entries[int(sel[0])] if sel else None

    def _on_select(self, _event=None) -> None:
        e = self._selected()
        if not e:
            return
        for box, key in ((self.raw_box, "raw"), (self.inj_box, "injected")):
            box.delete("1.0", "end")
            box.insert("1.0", e.get(key, ""))

    def _copy(self, key: str) -> None:
        e = self._selected()
        if not e:
            return
        from whisperflow.inject import clipboard

        clipboard.write_text(e.get(key, ""))
        self._status.config(text=f"{key} copied ✓")
        self.win.after(1500, lambda: self._status.config(text=""))
