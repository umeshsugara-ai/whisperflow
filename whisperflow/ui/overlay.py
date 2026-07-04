"""Floating status overlay — Wispr-style compact pill.

Recording:  [ ×  ▂▄▆█▆▄▂  ✓ ]  — X cancels, live waveform shows the mic is
hearing you, ✓ stops and transcribes. Draggable anywhere (position persists).

- WS_EX_NOACTIVATE (+ TOOLWINDOW): can NEVER steal keyboard focus — otherwise
  dictated text would land in the overlay instead of the target app. Mouse
  clicks/drag still work (NOACTIVATE blocks activation, not mouse input).
- Must be driven from the tkinter main thread; other threads use `.post()`.
"""

from __future__ import annotations

import ctypes
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from typing import Callable

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
SW_SHOWNOACTIVATE = 4

BG = "#161412"
FG = "#f2ede1"
FG_DIM = "#9a938a"
BTN_DARK = "#3a3733"
WAVE = "#f2ede1"
ACCENT_PROC = "#f5a623"
ACCENT_OK = "#5cb85c"
ACCENT_ERR = "#e5484d"
BORDER = "#ffffff"  # white outline so the pill stays visible on dark AND light backgrounds
BORDER_W = 2
TRANSPARENT = "#010203"  # unlikely-to-clash colorkey for rounded corners

POS_FILE = Path(__file__).resolve().parent.parent.parent / "overlay_pos.txt"

N_BARS = 14


class Overlay:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.win = tk.Toplevel(root)
        # withdraw IMMEDIATELY: a Toplevel maps (and Windows activates it) on
        # creation, which would grab foreground focus once at startup.
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", TRANSPARENT)
        self.win.configure(bg=TRANSPARENT)

        # full-size pill (hover / recording / processing states)
        self.width, self.height = 168, 40
        self.canvas = tk.Canvas(
            self.win, width=self.width, height=self.height, bg=TRANSPARENT, highlightthickness=0
        )
        self.canvas.pack()

        cy = self.height // 2
        self._pill = self._rounded_rect(
            2, 4, self.width - 2, self.height - 4, radius=17, fill=BG, outline=BORDER, width=BORDER_W
        )

        # cancel button (left)
        self._btn_x = self.canvas.create_oval(8, cy - 12, 32, cy + 12, fill=BTN_DARK, outline="")
        self._btn_x_label = self.canvas.create_text(
            20, cy, text="✕", fill=FG, font=("Segoe UI", 10, "bold")
        )
        # confirm button (right)
        self._btn_ok = self.canvas.create_oval(self.width - 32, cy - 12, self.width - 8, cy + 12, fill=FG, outline="")
        self._btn_ok_label = self.canvas.create_text(
            self.width - 20, cy, text="✓", fill=BG, font=("Segoe UI", 10, "bold")
        )

        # waveform bars (center)
        self._levels: deque[float] = deque([0.0] * N_BARS, maxlen=N_BARS)
        span_x0, span_x1 = 40, self.width - 40
        step = (span_x1 - span_x0) / N_BARS
        self._bars = [
            self.canvas.create_rectangle(
                span_x0 + i * step + 1, cy - 1, span_x0 + (i + 1) * step - 2, cy + 1, fill=WAVE, outline=""
            )
            for i in range(N_BARS)
        ]
        self._bar_geo = (span_x0, step, cy)

        # status text (used in processing/done states, hidden while recording)
        self._label = self.canvas.create_text(
            self.width // 2, cy, anchor="center", fill=FG, font=("Segoe UI", 9), text="", state="hidden"
        )

        # idle resting pill: a small compact pill CENTERED in the canvas — slim
        # in width and height — shown at rest. On hover the full-size _pill
        # takes over, so the resting look is a neat little pill while the hover
        # look keeps the original taller shape.
        rest_pw, rest_ph = 46, 14
        self._rest_pill = self._rounded_rect(
            self.width // 2 - rest_pw // 2, cy - rest_ph // 2,
            self.width // 2 + rest_pw // 2, cy + rest_ph // 2,
            radius=7, fill=BG, outline=BORDER, width=BORDER_W, state="hidden",
        )

        self._place_initial()
        self._apply_noactivate()
        self._hide_job: str | None = None
        self._pulse_job: str | None = None
        self._idle_job: str | None = None
        self._recording: bool = False

        # wired by app.py
        self.level_source: Callable[[], float] = lambda: 0.0
        self.on_cancel: Callable[[], None] = lambda: None
        self.on_confirm: Callable[[], None] = lambda: None
        self.on_start: Callable[[], None] = lambda: None
        self.on_open_main: Callable[[], None] = lambda: None  # right-click → app window
        self.persistent: bool = True  # when False, show_idle() hides instead
        self.hotkey_label: str = "Ctrl+Win"  # set by app.py from cfg.hotkey.combo
        self._hovering: bool = False
        self._at_rest: bool = False  # true only in the resting/idle look

        # click-vs-drag discrimination
        self._press: tuple[int, int] | None = None
        self._dragged = False
        self.canvas.bind("<Button-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Button-3>", lambda e: self.on_open_main())
        # hover-to-reveal the hotkey while resting (Wispr-style)
        self.canvas.bind("<Enter>", self._on_hover_enter)
        self.canvas.bind("<Leave>", self._on_hover_leave)
        self.canvas.config(cursor="hand2")  # clickable hand pointer on hover

    def _rounded_rect(self, x1, y1, x2, y2, radius, **kwargs) -> int:
        pts = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.canvas.create_polygon(pts, smooth=True, **kwargs)

    # ---- placement, drag, and button clicks ----

    def _place_initial(self) -> None:
        x = y = None
        try:
            x, y = (int(v) for v in POS_FILE.read_text().strip().split(","))
        except (FileNotFoundError, ValueError):
            pass
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        if x is None or not (0 <= x <= screen_w - 40 and 0 <= y <= screen_h - 40):
            x = (screen_w - self.width) // 2
            y = screen_h - self.height - 80
        self.win.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def _mouse_down(self, event) -> None:
        self._press = (event.x, event.y)
        self._dragged = False

    def _mouse_move(self, event) -> None:
        if self._press is None:
            return
        dx, dy = event.x - self._press[0], event.y - self._press[1]
        if abs(dx) + abs(dy) > 3:
            self._dragged = True
            self.win.geometry(f"+{self.win.winfo_x() + dx}+{self.win.winfo_y() + dy}")

    def _mouse_up(self, event) -> None:
        press, self._press = self._press, None
        if self._dragged:
            try:
                POS_FILE.write_text(f"{self.win.winfo_x()},{self.win.winfo_y()}")
            except OSError:
                pass
            return
        if press is None:
            return
        if not self._recording:
            # click on the resting pill starts dictation (mouse-first trigger)
            self.on_start()
            return
        # click (no drag): hit-test the two buttons
        cy = self.height // 2
        if 8 <= event.x <= 32 and cy - 12 <= event.y <= cy + 12:
            self.on_cancel()
        elif self.width - 32 <= event.x <= self.width - 8 and cy - 12 <= event.y <= cy + 12:
            self.on_confirm()

    # ---- win32 no-activate ----

    def _hwnd(self) -> int:
        self.win.update_idletasks()
        # GA_ROOT (2): the overlay's own top-level frame. GetParent would
        # return the OWNER (the withdrawn Tk root) for owned toplevels.
        GA_ROOT = 2
        return ctypes.windll.user32.GetAncestor(self.win.winfo_id(), GA_ROOT) or self.win.winfo_id()

    def _apply_noactivate(self) -> None:
        hwnd = self._hwnd()
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        )

    def _show(self) -> None:
        # SW_SHOWNOACTIVATE instead of tk's deiconify(): deiconify activates
        # the window, stealing focus from the injection target. Re-assert the
        # exstyle first — Tk can recreate the OS window across hide/show.
        self._apply_noactivate()
        ctypes.windll.user32.ShowWindow(self._hwnd(), SW_SHOWNOACTIVATE)

    # ---- state views (call from tk main thread; use .post() from others) ----

    def post(self, fn, *args) -> None:
        """Thread-safe: schedule a UI update on the tk main thread."""
        self.root.after(0, fn, *args)

    def _cancel_jobs(self) -> None:
        for attr in ("_hide_job", "_pulse_job", "_idle_job"):
            job = getattr(self, attr)
            if job:
                self.win.after_cancel(job)
                setattr(self, attr, None)

    def _set_recording_widgets(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        for item in (self._btn_x, self._btn_x_label, self._btn_ok, self._btn_ok_label, *self._bars):
            self.canvas.itemconfig(item, state=state)
        self.canvas.itemconfig(self._label, state="hidden" if visible else "normal")
        self.canvas.itemconfig(self._rest_pill, state="hidden")  # compact pill is idle-only
        self.canvas.itemconfig(self._pill, state="normal")  # full-width box for content states

    def _pulse(self) -> None:
        """Scroll the waveform with the live mic level."""
        level = 0.0
        try:
            level = min(1.0, self.level_source() * 20.0)  # faint mics still visible
        except Exception:
            pass
        self._levels.append(level)
        x0, step, cy = self._bar_geo
        for i, (bar, lv) in enumerate(zip(self._bars, self._levels)):
            h = 1 + lv * 11  # bar half-height 1..12px
            self.canvas.coords(bar, x0 + i * step + 1, cy - h, x0 + (i + 1) * step - 2, cy + h)
        self._pulse_job = self.win.after(70, self._pulse)

    def show_idle(self, hint: bool = False) -> None:
        """Persistent resting state — the pill stays visible so the user knows
        the app is alive. No auto-hide. `hint` briefly shows the hotkey tip,
        then settles into a compact resting pill that expands on hover. When not
        persistent (config always_visible=false), the pill hides at rest
        instead (legacy behavior)."""
        if not self.persistent:
            self.hide()
            return
        self._cancel_jobs()
        self._recording = False
        self._at_rest = True
        self._set_recording_widgets(False)  # hide buttons/bars, also hides _rest
        if hint:
            # startup hint: briefly show the full pill + hotkey, then collapse
            self.canvas.itemconfig(self._rest_pill, state="hidden")
            self.canvas.itemconfig(self._pill, state="normal")
            self.canvas.itemconfig(
                self._label, text=f"● {self.hotkey_label}", fill=FG_DIM, state="normal"
            )
            self._idle_job = self.win.after(4000, self._idle_minimal)
        else:
            self._render_rest()
        self._show()

    def _idle_minimal(self) -> None:
        """Fade the startup hint away, settling into the compact resting pill."""
        self._idle_job = None
        self._render_rest()

    def _render_rest(self) -> None:
        """The persistent resting look. At rest it's a small compact pill with a
        dim dot (like Wispr) — not a big empty bar. Hovering expands it to the
        full-width pill showing the hotkey (e.g. '● Ctrl+Win')."""
        if self._hovering:
            self.canvas.itemconfig(self._rest_pill, state="hidden")
            self.canvas.itemconfig(self._pill, state="normal")
            self.canvas.itemconfig(
                self._label, text=f"● {self.hotkey_label}", fill=FG, state="normal"
            )
        else:
            self.canvas.itemconfig(self._pill, state="hidden")  # hide the wide box
            self.canvas.itemconfig(self._rest_pill, state="normal")  # small pill only
            self.canvas.itemconfig(self._label, text="●", fill=FG_DIM, state="normal")

    def _on_hover_enter(self, _event=None) -> None:
        self._hovering = True
        if self._at_rest and self._idle_job is None:
            self._render_rest()

    def _on_hover_leave(self, _event=None) -> None:
        self._hovering = False
        if self._at_rest and self._idle_job is None:
            self._render_rest()

    def show_recording(self, device_name: str = "") -> None:
        self._cancel_jobs()
        self._recording = True
        self._at_rest = False
        self._levels.extend([0.0] * N_BARS)
        self._set_recording_widgets(True)
        self._show()
        self._pulse()

    def show_processing(self) -> None:
        self._cancel_jobs()
        self._recording = False
        self._at_rest = False
        self._set_recording_widgets(False)
        self.canvas.itemconfig(self._label, text="Transcribing…", fill=ACCENT_PROC)
        self._show()

    def flash_done(self, message: str = "Injected ✓") -> None:
        self._cancel_jobs()
        self._recording = False
        self._at_rest = False
        self._set_recording_widgets(False)
        self.canvas.itemconfig(self._label, text=message, fill=ACCENT_OK)
        self._show()
        self._hide_job = self.win.after(1200, self.show_idle)

    def flash_warn(self, message: str, duration_ms: int = 4000) -> None:
        """Amber attention flash — for outcomes that need the user to act
        (e.g. 'Copied — press Ctrl+V'). Holds longer than flash_done."""
        self._cancel_jobs()
        self._recording = False
        self._at_rest = False
        self._set_recording_widgets(False)
        self.canvas.itemconfig(self._label, text=message[:28], fill=ACCENT_PROC)
        self._show()
        self._hide_job = self.win.after(duration_ms, self.show_idle)

    def flash_error(self, message: str) -> None:
        self._cancel_jobs()
        self._recording = False
        self._at_rest = False
        self._set_recording_widgets(False)
        self.canvas.itemconfig(self._label, text=message[:28], fill=ACCENT_ERR)
        self._show()
        self._hide_job = self.win.after(2500, self.show_idle)

    def hide(self) -> None:
        self._cancel_jobs()
        self._recording = False
        self._at_rest = False
        self.win.withdraw()
