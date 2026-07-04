"""History pane — Wispr-style list over history.jsonl (embedded in MainWindow).

Lists recent dictations newest-first with live search; selecting a row shows
the full RAW and INJECTED texts side by side with one-click copy buttons, so a
dictation that landed in the wrong window (or nowhere) is never lost.

Must be created on the tkinter main thread.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from whisperflow.history import History

BG = "#161412"
FIELD = "#26221e"
FG = "#f2ede1"
FG_DIM = "#9a938a"
ACCENT = "#5cb85c"
BTN = "#3a3733"


def filter_entries(entries: list[dict], query: str) -> list[dict]:
    """Case-insensitive substring filter over raw + injected text."""
    q = query.strip().lower()
    if not q:
        return list(entries)
    return [
        e
        for e in entries
        if q in e.get("raw", "").lower() or q in e.get("injected", "").lower()
    ]


class HistoryPane(tk.Frame):
    def __init__(self, parent: tk.Misc, history: History) -> None:
        super().__init__(parent, bg=BG)
        self.history = history

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "WF.Treeview", background=BG, foreground=FG, fieldbackground=BG, rowheight=24, borderwidth=0
        )
        style.map("WF.Treeview", background=[("selected", BTN)])
        style.configure(
            "WF.Vertical.TScrollbar",
            background=BTN, troughcolor=BG, bordercolor=BG, arrowcolor=FG, relief="flat",
        )

        # search row
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(top, text="Search", bg=BG, fg=FG_DIM, font=("Segoe UI", 9)).pack(side="left")
        self._query = tk.StringVar()
        self._query.trace_add("write", lambda *_: self.refresh())
        tk.Entry(
            top, textvariable=self._query, bg=FIELD, fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=3)

        columns = ("time", "tier", "text")
        tree_wrap = tk.Frame(self, bg=BG)
        tree_wrap.pack(fill="both", expand=False, padx=8, pady=(4, 4))
        self.tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", style="WF.Treeview", height=8)
        self.tree.heading("time", text="Time")
        self.tree.heading("tier", text="Cleanup")
        self.tree.heading("text", text="Injected text")
        self.tree.column("time", width=130, stretch=False)
        self.tree.column("tier", width=70, stretch=False)
        self.tree.column("text", width=420)
        tree_sb = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=self.tree.yview, style="WF.Vertical.TScrollbar"
        )
        self.tree.configure(yscrollcommand=tree_sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # detail panes
        detail = tk.Frame(self, bg=BG)
        detail.pack(fill="both", expand=True, padx=8, pady=4)

        for col, (label, attr) in enumerate((("RAW (exactly what you said)", "raw_box"), ("INJECTED (after cleanup)", "inj_box"))):
            frame = tk.Frame(detail, bg=BG)
            frame.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else 0)
            detail.columnconfigure(col, weight=1)
            detail.rowconfigure(0, weight=1)
            tk.Label(frame, text=label, bg=BG, fg=FG_DIM, font=("Segoe UI", 8)).pack(anchor="w")
            box_wrap = tk.Frame(frame, bg=BG)
            box_wrap.pack(fill="both", expand=True)
            box = tk.Text(box_wrap, height=6, bg=FIELD, fg=FG, wrap="word", font=("Segoe UI", 9), relief="flat")
            box_sb = ttk.Scrollbar(
                box_wrap, orient="vertical", command=box.yview, style="WF.Vertical.TScrollbar"
            )
            box.configure(yscrollcommand=box_sb.set)
            box.pack(side="left", fill="both", expand=True)
            box_sb.pack(side="right", fill="y")
            setattr(self, attr, box)

        # buttons
        btns = tk.Frame(self, bg=BG)
        btns.pack(fill="x", padx=8, pady=(2, 8))
        self._status = tk.Label(btns, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 8))
        self._status.pack(side="right", padx=(0, 6))
        for text, cmd in (
            ("Copy RAW", lambda: self._copy("raw")),
            ("Copy injected", lambda: self._copy("injected")),
            ("Refresh", self.refresh),
        ):
            tk.Button(
                btns, text=text, command=cmd, bg=BTN, fg=FG, relief="flat", padx=10, cursor="hand2"
            ).pack(side="left", padx=(0, 6))
        tk.Button(
            btns, text="Clear history", command=self._clear, bg=BTN, fg="#e5484d",
            relief="flat", padx=10, cursor="hand2",
        ).pack(side="right", padx=(0, 6))

        self._entries: list[dict] = []
        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        newest_first = list(reversed(self.history.entries(limit=200)))
        self._entries = filter_entries(newest_first, self._query.get())
        for i, e in enumerate(self._entries):
            preview = e.get("injected", "").replace("\n", " ")
            preview = preview[:70] + ("…" if len(preview) > 70 else "")
            self.tree.insert("", "end", iid=str(i), values=(e.get("ts", ""), e.get("tier", ""), preview))
        if self._entries:
            self.tree.selection_set("0")
        else:
            for box in (self.raw_box, self.inj_box):
                box.delete("1.0", "end")

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
        self.after(1500, lambda: self._status.config(text=""))

    def _clear(self) -> None:
        if not messagebox.askyesno(
            "Clear history",
            "Delete all dictation entries?\n(Lifetime stats on the Home screen are kept.)",
            parent=self,
        ):
            return
        self.history.clear()
        self.refresh()
