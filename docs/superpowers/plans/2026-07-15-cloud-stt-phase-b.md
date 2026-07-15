# Phase B: Cloud STT Onboarding + Decision-Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user actually pick a speech-to-text provider from the app — a first-run chooser (confirm-required, never silent) plus a Settings "Speech engine" section — reusing Phase A's provider registry and engines (already merged, 140 tests green).

**Architecture:** Two new pure-logic modules (`providers.choose()` for the 2-question helper, `sysinfo.build_recommended_config()`/`build_config_for_engine()` for config-building without I/O) plus a shared Tkinter view-model helper (`whisperflow/ui/engine_picker.py`) that both the Settings page and a new first-run chooser (`whisperflow/ui/first_run.py`) render from. `app.py`'s `main()` is restructured so the Tk root is created *before* `build_controller()` runs whenever a first-run chooser needs to show (non-headless, no existing config.toml) — `build_controller` needs a fully-decided config, so the chooser must resolve one first.

**Tech Stack:** stdlib `tkinter`/`ttk` (matches existing UI), `webbrowser.open()` for signup links (no new dependency), `os.environ` for key detection. No new pip packages.

## Global Constraints

- No engine is ever picked or saved silently — `config.toml` is written only after explicit user confirmation ("Use recommended" click or a manual pick), or in the pre-existing `--headless` first-run path (unattended by definition — unchanged from Phase A).
- Reuse Phase A's registry (`whisperflow/stt/providers.py`: `Provider`, `get`, `all_providers`, `cloud_providers`, `is_cloud`) and `sysinfo.recommend()` — do not duplicate the VRAM-tier ladder.
- API keys are never logged or printed — `set_env_var` and all UI code interpolate only the env var *name*, never the value, in any log/print/exception.
- Every provider row's badge text is built from registry fields only (`cost_note`, `quality_tier`, `speed_note`, `kind`) — no per-provider UI special-casing.
- `--headless` first run keeps today's unattended `bootstrap_config()` behavior exactly (no dialog can show without a display).
- Follow existing UI conventions: colors `BG/CARD/FIELD/FG/FG_DIM/BTN/ACCENT_OK/ACCENT_WARN/ACCENT_ERR` and the `_button()` helper from `whisperflow/ui/main_window.py:41-49,82-86`, font `"Segoe UI"`.
- Only pure logic (data transforms, no live `tk` widgets) goes in the automated pytest suite, matching the existing repo convention (`tests/test_ui_pure.py`, `tests/test_guide_card.py` test pure functions, not rendered widgets). Widget-rendering code is verified by a manual offscreen-Tk smoke script per UI task (same pattern used to verify the "How to dictate" card in an earlier phase), not by pytest.

---

### Task 1: `providers.choose()` — Layer 3 "Help me choose" pure logic

**Files:**
- Modify: `whisperflow/stt/providers.py` (add one function; no changes to existing content)
- Test: `tests/test_providers.py` (append)

**Interfaces:**
- Consumes: `whisperflow.sysinfo.SystemSpecs` (dataclass with `gpu_name: str|None, vram_mb: int, ram_gb: float, cpu_cores: int` — already exists, do not import it directly to avoid a circular import; accept it as a duck-typed object with a `.vram_mb` attribute, matching how `providers.py` currently has zero dependency on `sysinfo.py`).
- Produces: `choose(privacy_pref: str, budget_pref: str, specs) -> str` — returns a provider id (`"local"`, `"groq"`, or `"openai"`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers.py`:

```python
class _FakeSpecs:
    def __init__(self, vram_mb=0):
        self.vram_mb = vram_mb


def test_choose_private_always_returns_local_regardless_of_budget():
    assert providers.choose("private", "free", _FakeSpecs(vram_mb=8192)) == "local"
    assert providers.choose("private", "paid_ok", _FakeSpecs(vram_mb=0)) == "local"


def test_choose_cloud_free_returns_groq():
    assert providers.choose("cloud_ok", "free", _FakeSpecs()) == "groq"


def test_choose_cloud_paid_returns_openai():
    assert providers.choose("cloud_ok", "paid_ok", _FakeSpecs()) == "openai"


def test_choose_unknown_privacy_pref_raises():
    with pytest.raises(ValueError, match="privacy_pref"):
        providers.choose("maybe", "free", _FakeSpecs())


def test_choose_unknown_budget_pref_raises():
    with pytest.raises(ValueError, match="budget_pref"):
        providers.choose("cloud_ok", "maybe", _FakeSpecs())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_providers.py -v`
Expected: FAIL with `AttributeError: module 'whisperflow.stt.providers' has no attribute 'choose'`

- [ ] **Step 3: Implement `choose()`**

Append to `whisperflow/stt/providers.py` (after the existing `is_cloud` function, same file — do not create a new file):

```python
def choose(privacy_pref: str, budget_pref: str, specs) -> str:
    """Map the 2-question "Help me choose" answers to a provider id.

    privacy_pref: "private" (fully offline) | "cloud_ok" (cloud is fine)
    budget_pref:  "free" | "paid_ok"
    specs: anything with a `.vram_mb` attribute (sysinfo.SystemSpecs) — kept
    duck-typed so this module has zero dependency on sysinfo.py.

    Privacy always wins: "private" returns "local" regardless of budget
    (the local engine works with or without a GPU — just slower on CPU).
    """
    if privacy_pref not in ("private", "cloud_ok"):
        raise ValueError(f"privacy_pref must be 'private' or 'cloud_ok', got {privacy_pref!r}")
    if budget_pref not in ("free", "paid_ok"):
        raise ValueError(f"budget_pref must be 'free' or 'paid_ok', got {budget_pref!r}")
    if privacy_pref == "private":
        return "local"
    return "groq" if budget_pref == "free" else "openai"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_providers.py -v`
Expected: PASS (13 passed — 8 existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add whisperflow/stt/providers.py tests/test_providers.py
git commit -m "Add providers.choose() — 2-question decision-support mapping"
```

---

### Task 2: `config.set_env_var()` — write a provider's API key to `.env`

**Files:**
- Modify: `whisperflow/config.py` (add one function near `load_dotenv`, e.g. right after it)
- Test: `tests/test_config_write.py` (append)

**Interfaces:**
- Consumes: `whisperflow.config.data_dir()` (existing, `config.py:34-44`).
- Produces: `set_env_var(key: str, value: str, path: Path | None = None) -> None` — creates or updates `path` (default `data_dir() / ".env"`), replacing an existing `KEY=...` line in place (preserving every other line and its order/comments) or appending a new line if the key isn't present yet. Also sets `os.environ[key] = value` immediately so the running process picks it up without a restart (the caller — the Settings page or the first-run chooser — needs the key to resolve right away, e.g. so `resolve_api_key()` works before the user restarts).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_write.py`:

```python
def test_set_env_var_creates_file_when_missing(tmp_path, monkeypatch):
    from whisperflow.config import set_env_var

    monkeypatch.delenv("TEST_NEW_KEY", raising=False)
    env_path = tmp_path / ".env"
    set_env_var("TEST_NEW_KEY", "abc123", path=env_path)
    assert env_path.read_text(encoding="utf-8") == "TEST_NEW_KEY=abc123\n"
    assert os.environ["TEST_NEW_KEY"] == "abc123"


def test_set_env_var_updates_existing_key_in_place(tmp_path, monkeypatch):
    from whisperflow.config import set_env_var

    env_path = tmp_path / ".env"
    env_path.write_text("# a comment\nOTHER_KEY=keep-me\nTARGET_KEY=old-value\n", encoding="utf-8")
    set_env_var("TARGET_KEY", "new-value", path=env_path)
    text = env_path.read_text(encoding="utf-8")
    assert text == "# a comment\nOTHER_KEY=keep-me\nTARGET_KEY=new-value\n"
    assert os.environ["TARGET_KEY"] == "new-value"


def test_set_env_var_appends_when_file_has_other_keys(tmp_path, monkeypatch):
    from whisperflow.config import set_env_var

    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=1\n", encoding="utf-8")
    set_env_var("NEW_ONE", "hello", path=env_path)
    text = env_path.read_text(encoding="utf-8")
    assert text == "EXISTING=1\nNEW_ONE=hello\n"


def test_set_env_var_never_logs_the_value(tmp_path, caplog):
    import logging

    from whisperflow.config import set_env_var

    caplog.set_level(logging.DEBUG)
    env_path = tmp_path / ".env"
    set_env_var("SECRET_KEY", "super-secret-value-xyz", path=env_path)
    assert "super-secret-value-xyz" not in caplog.text
```

Add `import os` at the top of `tests/test_config_write.py` if it isn't already imported — check the file's existing imports first.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_config_write.py -k set_env_var -v`
Expected: FAIL with `ImportError: cannot import name 'set_env_var'`

- [ ] **Step 3: Implement `set_env_var()`**

In `whisperflow/config.py`, add this function directly after `load_dotenv` (which ends around line 184 — locate it by its closing `return count` line, not by line number, since earlier tasks in this codebase have shifted line numbers before):

```python
def set_env_var(key: str, value: str, path: Path | None = None) -> None:
    """Create or update a KEY=VALUE line in the .env file, preserving every
    other line. Also updates os.environ so the change is picked up
    immediately in the running process (no restart needed to resolve the
    key via ModelConfig.resolve_api_key()).

    Never logs `value` — callers (Settings/first-run UI) must not either.
    """
    env_path = Path(path) if path else data_dir() / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                new_lines.append(f"{key}={value}")
                found = True
                continue
        new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ[key] = value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_config_write.py -v`
Expected: PASS (all tests in the file, including the 4 new ones)

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass (140 prior + new ones from Tasks 1-2).

- [ ] **Step 6: Commit**

```bash
git add whisperflow/config.py tests/test_config_write.py
git commit -m "Add config.set_env_var() — write a provider API key to .env in place"
```

---

### Task 3: Pure config-builders — `sysinfo.build_recommended_config()` + `build_config_for_engine()`

**Files:**
- Modify: `whisperflow/sysinfo.py` (add two functions after `recommend()`, before `startup_check()`)
- Modify: `app.py` — `bootstrap_config()` (around line 353-379) refactored to call the new pure function instead of duplicating the dataclass-building logic
- Test: `tests/test_sysinfo.py` (append)

**Interfaces:**
- Consumes: `whisperflow.sysinfo.Recommendation` (existing dataclass: `engine, name, device, compute_type, reason, alternatives` — already defined in this file), `whisperflow.sysinfo.SystemSpecs`, `whisperflow.stt.providers.get` (Task-1-adjacent, already exists from Phase A).
- Produces: `build_recommended_config(rec: Recommendation) -> Config` (pure — no file I/O, `cfg.path` stays at its dataclass default) and `build_config_for_engine(engine_id: str, specs: SystemSpecs) -> Config` (pure — for a user's MANUAL pick in the chooser, not the auto-recommendation).

**Design note:** `build_config_for_engine("local", specs)` reuses `recommend(specs, has_api_key=False)` for hardware-tiered local sizing instead of duplicating the VRAM ladder — passing `has_api_key=False` guarantees `recommend()` returns `engine="local"` (the only non-local branch in the whole ladder requires `has_api_key=True`), so this is always correct, not a special case.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sysinfo.py`:

```python
def test_build_recommended_config_local():
    rec = Recommendation(
        engine="local", name="large-v3-turbo", device="cuda", compute_type="int8_float16",
        reason="test", alternatives=[],
    )
    cfg = sysinfo.build_recommended_config(rec)
    assert cfg.model.engine == "local"
    assert cfg.model.name == "large-v3-turbo"
    assert cfg.model.device == "cuda"
    assert cfg.model.compute_type == "int8_float16"


def test_build_recommended_config_cloud_sets_api_key_env_from_registry():
    rec = Recommendation(
        engine="groq", name="whisper-large-v3-turbo", device="cpu", compute_type="int8",
        reason="test", alternatives=[],
    )
    cfg = sysinfo.build_recommended_config(rec)
    assert cfg.model.engine == "groq"
    assert cfg.model.cloud_model == "whisper-large-v3-turbo"
    assert cfg.model.api_key_env == "GROQ_API_KEY"


def test_build_config_for_engine_local_reuses_recommend_ladder():
    cfg = sysinfo.build_config_for_engine("local", specs(vram_mb=8192, gpu="RTX 4060"))
    assert cfg.model.engine == "local"
    assert cfg.model.name == "large-v3-turbo"  # matches the big-GPU ladder branch
    assert cfg.model.device == "cuda"


def test_build_config_for_engine_cloud_uses_provider_default():
    cfg = sysinfo.build_config_for_engine("openai", specs(vram_mb=0, ram_gb=4, cores=2))
    assert cfg.model.engine == "openai"
    assert cfg.model.cloud_model == "gpt-4o-transcribe"
    assert cfg.model.api_key_env == "OPENAI_API_KEY"
```

(This file already has a `specs(...)` helper at the top — reuse it, don't redefine it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_sysinfo.py -k "build_recommended_config or build_config_for_engine" -v`
Expected: FAIL with `AttributeError: module 'whisperflow.sysinfo' has no attribute 'build_recommended_config'`

- [ ] **Step 3: Implement both functions**

In `whisperflow/sysinfo.py`, add directly after the `recommend()` function (locate it by its closing lines — the final `return Recommendation(engine="local", name="small", device="cpu", ...)` fallback block — and insert before `def startup_check(...)`):

```python
def build_recommended_config(rec: Recommendation):
    """Pure: build a Config from a Recommendation, no file I/O. Used by both
    the unattended `--headless` first-run path (app.py bootstrap_config) and
    the interactive first-run chooser's "Use recommended" button."""
    from whisperflow.config import Config
    from whisperflow.stt import providers

    cfg = Config()
    cfg.model.engine = rec.engine
    if rec.engine == "local":
        cfg.model.name = rec.name
        cfg.model.device = rec.device
        cfg.model.compute_type = rec.compute_type
    else:
        cfg.model.cloud_model = rec.name
        cfg.model.api_key_env = providers.get(rec.engine).api_key_env
    return cfg


def build_config_for_engine(engine_id: str, specs: SystemSpecs):
    """Pure: build a Config for a user's MANUAL provider pick (first-run
    chooser or Settings), as opposed to the system's auto-recommendation.

    For "local" this reuses recommend()'s hardware-tiered sizing rather than
    duplicating the VRAM ladder: recommend(specs, has_api_key=False) always
    returns engine="local" (the only non-local branch requires
    has_api_key=True), so its name/device/compute_type are exactly the
    right local sizing for this machine regardless of why the caller wants
    local.
    """
    if engine_id == "local":
        return build_recommended_config(recommend(specs, has_api_key=False))
    from whisperflow.config import Config
    from whisperflow.stt import providers

    provider = providers.get(engine_id)
    cfg = Config()
    cfg.model.engine = engine_id
    cfg.model.cloud_model = provider.default_model
    cfg.model.api_key_env = provider.api_key_env
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_sysinfo.py -v`
Expected: PASS (all tests in the file, including the 4 new ones)

- [ ] **Step 5: Refactor `bootstrap_config()` in `app.py` to reuse `build_recommended_config()`**

Read the current `bootstrap_config` function in `app.py` first (search for `def bootstrap_config` — it's around line 353, but confirm by content, not line number, since this file has changed across prior tasks). Replace its body so it delegates the dataclass-building to the new pure function instead of duplicating it:

```python
def bootstrap_config(path: Path):
    """First run with no config.toml (installed build, or --headless):
    probe the hardware, generate a config from the recommendation, and save
    it — used only when there's no interactive first-run chooser (headless
    mode) or as the chooser's own "Use recommended" action."""
    from whisperflow import sysinfo
    from whisperflow.config import save_config

    specs = sysinfo.probe()
    rec = sysinfo.recommend(specs, has_api_key=_any_cloud_api_key_available())
    cfg = sysinfo.build_recommended_config(rec)
    cfg.path = path
    save_config(cfg, path)
    log.info(
        "first run — generated %s for %s (%s)",
        path.name,
        specs.gpu_name or f"CPU ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM)",
        rec.reason,
    )
    return cfg
```

This preserves the exact same behavior and log line as before — it's a pure refactor (the dataclass-building moved to `sysinfo.build_recommended_config`, everything else is identical).

- [ ] **Step 6: Run the existing bootstrap tests to confirm no regression**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_bootstrap_config.py -v`
Expected: PASS (these tests were added in Phase A and must still pass unchanged — they test `bootstrap_config`'s observable behavior, which hasn't changed).

- [ ] **Step 7: Run the full suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 8: Commit**

```bash
git add whisperflow/sysinfo.py app.py tests/test_sysinfo.py
git commit -m "Add sysinfo.build_recommended_config()/build_config_for_engine(); bootstrap_config reuses it"
```

---

### Task 4: `whisperflow/ui/engine_picker.py` — shared pure view-model for the provider list

**Files:**
- Create: `whisperflow/ui/engine_picker.py`
- Test: `tests/test_engine_picker.py`

**Interfaces:**
- Consumes: `whisperflow.stt.providers.Provider`, `all_providers()` (Phase A, existing).
- Produces: `badge_line(provider: Provider) -> str` and `build_rows(recommended_id: str | None = None) -> list[dict]` (each row dict: `{"id": str, "display_name": str, "badge": str, "is_recommended": bool}`, in `providers.all_providers()` registry order). This is the ONLY module both `SettingsPage` (Task 5) and the first-run chooser (Task 6) import for "what to show" — no duplicated row-building logic in either.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_picker.py
# -*- coding: utf-8 -*-
"""Pure view-model helpers for the provider picker (Settings + first-run chooser)."""

from whisperflow.stt import providers
from whisperflow.ui.engine_picker import badge_line, build_rows


def test_badge_line_for_cloud_provider_mentions_cost_and_quality():
    groq = providers.get("groq")
    line = badge_line(groq)
    assert "☁ Cloud" in line
    assert groq.cost_note in line
    assert "Better" in line  # groq.quality_tier capitalized
    assert groq.speed_note in line


def test_badge_line_for_local_provider_says_offline():
    local = providers.get("local")
    line = badge_line(local)
    assert "🔒 Offline" in line
    assert local.cost_note in line


def test_build_rows_covers_every_registered_provider_in_order():
    rows = build_rows()
    ids = [r["id"] for r in rows]
    assert ids == [p.id for p in providers.all_providers()]
    for row in rows:
        assert set(row.keys()) == {"id", "display_name", "badge", "is_recommended"}
        assert row["is_recommended"] is False


def test_build_rows_marks_the_recommended_provider():
    rows = build_rows(recommended_id="groq")
    flagged = [r for r in rows if r["is_recommended"]]
    assert len(flagged) == 1
    assert flagged[0]["id"] == "groq"


def test_build_rows_no_recommendation_flags_nothing():
    rows = build_rows(recommended_id=None)
    assert all(not r["is_recommended"] for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_engine_picker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.ui.engine_picker'`

- [ ] **Step 3: Implement**

```python
# whisperflow/ui/engine_picker.py
"""Pure view-model helpers for the speech-engine picker — shared by the
Settings "Speech engine" section and the first-run chooser so there's one
place that decides what a provider row says, not two.

No tkinter imports here on purpose: this module is plain data transforms
over whisperflow.stt.providers, unit-testable without a display.
"""

from __future__ import annotations

from whisperflow.stt.providers import Provider, all_providers


def badge_line(provider: Provider) -> str:
    """One-line summary: privacy · cost · quality · speed."""
    privacy = "🔒 Offline" if provider.kind == "local" else "☁ Cloud"
    cost_icon = "💚" if provider.cost_tier == "free" else "💛"
    quality = provider.quality_tier.capitalize()
    return f"{privacy} · {cost_icon} {provider.cost_note} · {quality} · {provider.speed_note}"


def build_rows(recommended_id: str | None = None) -> list[dict]:
    """One row per registered provider, in registry order."""
    return [
        {
            "id": p.id,
            "display_name": p.display_name,
            "badge": badge_line(p),
            "is_recommended": p.id == recommended_id,
        }
        for p in all_providers()
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_engine_picker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add whisperflow/ui/engine_picker.py tests/test_engine_picker.py
git commit -m "Add engine_picker — shared pure view-model for provider list rows"
```

---

### Task 5: Settings "Speech engine" section

**Files:**
- Modify: `whisperflow/ui/main_window.py` — `SettingsPage.__init__` (add a new section) and `SettingsPage._save`/`refresh`/`_update_banner` (wire the new fields)
- Manual verification: offscreen-Tk smoke script (not part of pytest — matches the repo's established pattern for verifying rendered widgets)

**Interfaces:**
- Consumes: `whisperflow.ui.engine_picker.build_rows` (Task 4), `whisperflow.config.set_env_var` (Task 2), `whisperflow.stt.providers.get/cloud_providers/is_cloud` (Phase A), `sysinfo.recommend`/`sysinfo.probe` (existing).
- Produces: nothing new consumed by later tasks — this is a leaf UI feature. (Task 6's first-run chooser reuses `engine_picker` directly, not anything from this task.)

**Design (read before implementing):** Read the CURRENT `SettingsPage` class in `whisperflow/ui/main_window.py` in full first (search `class SettingsPage`) — this task inserts a new section using the same `row(r, label, caption)` grid helper already defined in `SettingsPage.__init__` (rows 0, 2, 4, 6, 8 are taken; use row 10 for this new section, with row 11 reserved for its caption per the existing `row()` convention of `r+1`). Do not change the existing hotkey/language/cleanup/overlay/autostart sections.

- [ ] **Step 1: Add the Speech engine section to `SettingsPage.__init__`**

Insert this block in `whisperflow/ui/main_window.py`'s `SettingsPage.__init__`, immediately after the existing `autostart_holder` block (which ends with the `.pack(anchor="w")` call for the autostart `Checkbutton`) and before the `foot = tk.Frame(...)` block that adds the Save button:

```python
        from whisperflow.stt import providers as _stt_providers
        from whisperflow.ui.engine_picker import build_rows

        self._engine_recommended_id = None  # set lazily in refresh() via sysinfo.recommend()
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
```

- [ ] **Step 2: Add the engine-related methods to `SettingsPage`**

Insert these methods into the `SettingsPage` class, right after the existing `_selected_language` method:

```python
    def _on_engine_picked(self) -> None:
        from whisperflow.stt import providers as _stt_providers

        engine_id = self.engine_var.get()
        provider = _stt_providers.get(engine_id)
        star = "★ Recommended for your PC — " if engine_id == self._engine_recommended_id else ""
        from whisperflow.ui.engine_picker import badge_line

        self._engine_badge.config(text=f"{star}{badge_line(provider)}")
        self._render_key_entry(provider)

    def _render_key_entry(self, provider) -> None:
        for child in self._engine_key_frame.winfo_children():
            child.destroy()
        if provider.kind == "local":
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
            lambda: __import__("webbrowser").open(provider.signup_url),
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
            from whisperflow.config import set_env_var

            value = key_var.get().strip()
            if not value:
                return
            set_env_var(provider.api_key_env, value, path=self.cfg.path.parent / ".env")
            self._render_key_entry(provider)  # re-render to show "✓ ... is set"

        _button(key_row, "Save key", _save_key).pack(side="left", padx=(6, 0))
```

- [ ] **Step 3: Wire `refresh()` to compute the recommendation and pre-select the current engine**

In `SettingsPage.refresh()`, add this at the end of the method (after the existing `self._update_banner()` line):

```python
        if self._engine_recommended_id is None:
            from whisperflow import sysinfo

            specs = sysinfo.probe()
            has_key = any(
                os.environ.get(p.api_key_env) for p in __import__(
                    "whisperflow.stt.providers", fromlist=["cloud_providers"]
                ).cloud_providers() if p.api_key_env
            )
            self._engine_recommended_id = sysinfo.recommend(specs, has_api_key=has_key).engine
        self.engine_var.set(self.cfg.model.engine)
        self._on_engine_picked()
```

(The `__import__` calls above match this codebase's existing lazy-import style used elsewhere in `SettingsPage`/`HomePage` — e.g. `_toggle_autostart` imports `sysinfo` at module scope already, but `providers` is imported lazily throughout this file per Phase A's pattern. If a cleaner import is available — i.e. `providers` is already imported at the top of the method — use `from whisperflow.stt import providers` instead of `__import__`; do not leave the `__import__` form if a normal import reads better, this is written defensively because the exact top-of-method import state depends on Task 5's own insertion point.)

- [ ] **Step 4: Wire `_save()` to persist the engine change**

In `SettingsPage._save()`, add `self.cfg.model.engine = self.engine_var.get()` to the list of fields being set (alongside the existing `self.cfg.hotkey.combo = ...` etc. assignments), and add `"engine"` to the `old` tuple being captured/restored on failure — read the current `_save()` method in full first and extend its existing tuple-based save/rollback pattern consistently (add the field to both the `old = (...)` line and the rollback assignment, in the same position).

- [ ] **Step 5: Wire `_update_banner()` to flag an engine change as restart-required**

In `SettingsPage.__init__`, add `self._launch_engine = cfg.model.engine` alongside the existing `self._launch_combo`/`self._launch_language` lines. In `_update_banner()`, add a check: `if self.cfg.model.engine != self._launch_engine: pending.append("speech engine")` alongside the existing hotkey/language checks.

- [ ] **Step 6: Manual smoke verification (offscreen Tk, not part of pytest)**

Write a throwaway script (do not commit it — this matches how the "How to dictate" card was verified in an earlier phase) that imports `SettingsPage`, constructs it with a real `Config()` and a fake `History`, and asserts the engine combobox contains all 5 provider ids and that picking `"groq"` populates `self._engine_badge['text']` with the Groq badge text and shows a "Get a free key" button when `GROQ_API_KEY` isn't set in the environment. Run it with:
`C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe <script path>`
Expected output: no exceptions, printed confirmation lines for each assertion.

- [ ] **Step 7: Run the full pytest suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass (this task adds no new pytest files, so the count should match Task 4's final count exactly — confirms nothing broke).

- [ ] **Step 8: Commit**

```bash
git add whisperflow/ui/main_window.py
git commit -m "Add Speech engine section to Settings — provider picker, badges, key entry"
```

---

### Task 6: First-run chooser + `app.py` restructure

**Files:**
- Create: `whisperflow/ui/first_run.py`
- Modify: `app.py` — `main()` (restructure the config-bootstrap section) and `run_with_ui()` (accept an optional pre-existing `root`)
- Test: `tests/test_first_run.py` (pure-logic parts only, per this plan's Global Constraints)
- Manual verification: offscreen-Tk smoke script for the dialog itself

**Interfaces:**
- Consumes: `whisperflow.ui.engine_picker.build_rows` (Task 4), `whisperflow.sysinfo.build_recommended_config`/`build_config_for_engine` (Task 3), `whisperflow.config.set_env_var`/`save_config` (Task 2, existing), `whisperflow.stt.providers.get`.
- Produces: `whisperflow.ui.first_run.show_first_run_chooser(root, specs, rec, path) -> Config` — shown by `app.py` before `build_controller()` runs, whenever `config.toml` doesn't exist AND the app isn't `--headless`.

**Design (read before implementing):** Read the CURRENT `app.py` `main()` function in full — specifically the block that currently reads:
```python
    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        cfg = bootstrap_config(cfg_path)
    else:
        cfg = load_config(cfg_path)
```
and the final two lines of `main()`:
```python
    if args.headless:
        return run_headless(cfg, ctl, listener)
    return run_with_ui(cfg, ctl, listener, history, autostarted=args.autostart)
```
and the start of `run_with_ui`:
```python
def run_with_ui(cfg, ctl, listener, history, autostarted: bool = False) -> int:
    import threading
    import tkinter as tk
    ...
    root = tk.Tk()
    root.withdraw()  # the root stays hidden — MainWindow/overlay are Toplevels
```
Locate these by their content (they may have shifted a few lines from earlier tasks in this same file), not by a hardcoded line number.

- [ ] **Step 1: Write the failing test for the pure logic in this task**

`show_first_run_chooser` itself is a blocking Tkinter dialog and isn't unit-tested directly (per this plan's Global Constraints — UI rendering gets a manual smoke check). But the "does a cloud pick with an existing key skip straight to saving" decision and "does a keyless cloud pick fall back to local when deferred" decision are pure enough to extract and test. Create:

```python
# tests/test_first_run.py
# -*- coding: utf-8 -*-
"""Pure decision logic backing the first-run chooser dialog."""

from whisperflow.ui.first_run import provider_already_has_key


def test_provider_already_has_key_true_when_env_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "some-key")
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("groq")) is True


def test_provider_already_has_key_false_when_unset(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("groq")) is False


def test_provider_already_has_key_false_for_local():
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("local")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_first_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.ui.first_run'`

- [ ] **Step 3: Implement `whisperflow/ui/first_run.py`**

```python
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
from tkinter import ttk

from whisperflow.stt.providers import Provider, cloud_providers, get

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


def show_first_run_chooser(root, specs, rec, path):
    """Blocking modal. Returns the Config the user confirmed — either the
    recommendation or a manual pick — already saved to `path`."""
    from whisperflow.config import set_env_var
    from whisperflow.sysinfo import build_config_for_engine, build_recommended_config
    from whisperflow.ui.engine_picker import badge_line, build_rows

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

    for row in build_rows(recommended_id=rec.engine):
        if row["id"] == rec.engine:
            continue  # already shown above as the recommended option
        r = tk.Frame(list_frame, bg=CARD, padx=10, pady=6)
        r.pack(fill="x", pady=(0, 6))
        tk.Label(r, text=row["display_name"], bg=CARD, fg=FG, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(r, text=row["badge"], bg=CARD, fg=FG_DIM, font=("Segoe UI", 8)).pack(anchor="w")
        tk.Button(
            r, text="Choose", command=lambda pid=row["id"]: _pick(pid),
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
            set_env_var(provider.api_key_env, value)
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

    def _finish(cfg) -> None:
        from whisperflow.config import save_config

        save_config(cfg, path)
        result["cfg"] = cfg
        win.grab_release()
        win.destroy()

    # Closing the window (X button) must never leave the app unconfigured —
    # treat it exactly like "I'll add a key later": fall back to local.
    win.protocol("WM_DELETE_WINDOW", _finish_local)

    win.wait_window()
    return result["cfg"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_first_run.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Restructure `app.py`'s `main()` to show the chooser before `build_controller()`**

Replace the config-bootstrap block in `main()`:

```python
    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        cfg = bootstrap_config(cfg_path)
    else:
        cfg = load_config(cfg_path)
```

with:

```python
    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    first_run_root = None
    if not cfg_path.exists():
        if args.headless:
            cfg = bootstrap_config(cfg_path)  # unattended — no display to show a chooser on
        else:
            import tkinter as tk

            from whisperflow import sysinfo
            from whisperflow.ui.first_run import show_first_run_chooser

            first_run_root = tk.Tk()
            first_run_root.withdraw()
            specs = sysinfo.probe()
            rec = sysinfo.recommend(specs, has_api_key=_any_cloud_api_key_available())
            cfg = show_first_run_chooser(first_run_root, specs, rec, cfg_path)
            log.info("first run — user chose %s via the chooser dialog", cfg.model.engine)
    else:
        cfg = load_config(cfg_path)
```

Then find the final two lines of `main()`:

```python
    if args.headless:
        return run_headless(cfg, ctl, listener)
    return run_with_ui(cfg, ctl, listener, history, autostarted=args.autostart)
```

and change the second line to pass the chooser's root through (so `run_with_ui` doesn't create a second, conflicting `tk.Tk()` instance):

```python
    if args.headless:
        return run_headless(cfg, ctl, listener)
    return run_with_ui(cfg, ctl, listener, history, autostarted=args.autostart, root=first_run_root)
```

- [ ] **Step 6: Make `run_with_ui()` accept an optional pre-existing root**

Change the function signature:

```python
def run_with_ui(cfg, ctl, listener, history, autostarted: bool = False) -> int:
```

to:

```python
def run_with_ui(cfg, ctl, listener, history, autostarted: bool = False, root=None) -> int:
```

And change its body — find:

```python
    import threading
    import tkinter as tk

    from whisperflow import sysinfo
    from whisperflow.processing import build_processor
    from whisperflow.ui.overlay import Overlay
    from whisperflow.ui.tray import Tray

    root = tk.Tk()
    root.withdraw()  # the root stays hidden — MainWindow/overlay are Toplevels
```

to:

```python
    import threading
    import tkinter as tk

    from whisperflow import sysinfo
    from whisperflow.processing import build_processor
    from whisperflow.ui.overlay import Overlay
    from whisperflow.ui.tray import Tray

    if root is None:
        root = tk.Tk()
        root.withdraw()  # the root stays hidden — MainWindow/overlay are Toplevels
```

(When the first-run chooser already created and withdrew a root, it's reused as-is here — Tkinter does not support more than one `Tk()` instance per process.)

- [ ] **Step 7: Manual smoke verification (offscreen Tk, not part of pytest)**

Since this dialog blocks on `win.wait_window()`, a smoke test needs to programmatically click a button rather than a human waiting. Write a throwaway script (do not commit) that:
1. Builds a fake `Recommendation(engine="groq", name="whisper-large-v3-turbo", device="cpu", compute_type="int8", reason="test", alternatives=[])` and a fake `SystemSpecs(gpu_name=None, vram_mb=0, ram_gb=4, cpu_cores=2)`.
2. Creates a `tk.Tk()` root, calls `show_first_run_chooser(root, specs, rec, tmp_path / "config.toml")` — but BEFORE calling it, schedule `root.after(200, lambda: <find and invoke the "Use recommended" button's command programmatically>)` so the dialog auto-confirms instead of hanging forever waiting for a real click. (Tkinter widgets support `.invoke()` on buttons — locate the recommended-button by walking `rec_frame.winfo_children()` for a `tk.Button` widget, or simpler: capture a reference to the button before returning from `show_first_run_chooser` by having the smoke script monkeypatch `tkinter.Toplevel.wait_window` to auto-invoke instead of actually waiting — pick whichever approach is simpler to write correctly, the goal is just confirming the dialog constructs without exceptions and produces a Config when confirmed).
3. Assert the returned `Config.model.engine == "groq"` and that `(tmp_path / "config.toml").exists()`.

Run it with: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe <script path>`
Expected: no exceptions, prints confirmation.

- [ ] **Step 8: Run the full pytest suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass, including `tests/test_bootstrap_config.py` (unaffected — `bootstrap_config` itself didn't change again in this task) and `tests/test_first_run.py` (Step 4). No regressions in the full count.

- [ ] **Step 9: Commit**

```bash
git add whisperflow/ui/first_run.py tests/test_first_run.py app.py
git commit -m "Add first-run chooser dialog; app.py shows it before build_controller (non-headless first run)"
```

---

### Task 7: README docs — "Which speech engine should I pick?"

**Files:**
- Modify: `README.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add the section**

Add a new `## Which speech engine should I pick?` section to `README.md`, placed after the existing `## Which model should I use?` section (search for that heading to find the insertion point — do not guess the line number, this file has been edited across every phase of this project). Content:

```markdown
## Which speech engine should I pick?

WhisperFlow never picks silently — on first launch you'll see a chooser with a
recommendation and the full list, and you confirm before anything is saved.
Change it anytime in **Settings → Speech engine**.

| Engine | Privacy | Cost | Quality | Speed | Needs |
|---|---|---|---|---|---|
| **Local** | 🔒 Fully offline | Free | Best | Depends on your GPU | A decent NVIDIA GPU for good speed |
| **Groq** | ☁ Cloud | Free — 2,000/day | Better | Instant | A free account (30 seconds to sign up) |
| **Gemini** | ☁ Cloud | Free tier | Better | Fast | A free Google account |
| **OpenAI** | ☁ Cloud | Paid (~$0.006/min) | Best | Fast | Billing set up on your OpenAI account |
| **Deepgram** | ☁ Cloud | $200 free credit, then paid | Best | Fast | A free account to start |

**Quick picks:**
- Good NVIDIA GPU and want everything to stay on your machine → **Local**.
- No GPU, or your GPU is weak/shared with other apps → **Groq** (free, same
  Whisper model as Local, but instant — this is what the app recommends for you
  automatically if it doesn't find a good GPU).
- Want the best possible accuracy and don't mind paying a little → **OpenAI**
  or **Deepgram**.

Each cloud option needs its own free (or paid) API key — the in-app chooser and
Settings screen walk you through getting one with a direct sign-up link, or you
can add it by hand to a `.env` file next to your data folder:
```
GROQ_API_KEY=paste-your-key-here
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "README: document the speech-engine picker and provider comparison table"
```

---

## Self-Review Notes (completed during plan authoring)

- **Spec coverage:** Layer 1 (recommendation + reason, confirm-required) ✓ Task 6, Layer 2 (badges, Settings section, key entry, `.env` write) ✓ Tasks 2/4/5, Layer 3 ("Help me choose" pure mapping) ✓ Task 1 — note: the spec's 2-question UI widget itself (asking the two questions interactively) is deliberately deferred; `providers.choose()` is built and tested, but wiring a 2-question dialog to call it is small enough to fold into a follow-up if desired, since the plan already delivers full provider choice via the main chooser/Settings list without it. First-run chooser replacing silent `bootstrap_config` write ✓ Task 6. README docs ✓ Task 7. `set_env_var` ✓ Task 2.
- **Placeholder scan:** none found — every step has literal code, exact file paths, exact commands. The two "manual verification" steps (Task 5 Step 6, Task 6 Step 7) intentionally describe a throwaway/uncommitted script rather than a pytest file, consistent with this plan's Global Constraints on what belongs in the automated suite — this mirrors how the pre-existing "How to dictate" card was verified in an earlier phase of this same project.
- **Type consistency:** `Config`/`ModelConfig` fields (`engine`, `name`, `device`, `compute_type`, `cloud_model`, `api_key_env`) used identically across Tasks 3, 5, 6, matching Phase A's already-shipped `whisperflow/config.py` dataclasses. `Provider` fields (`kind`, `cost_note`, `quality_tier`, `speed_note`, `signup_url`, `setup_steps`, `api_key_env`, `display_name`) used identically across Tasks 4, 5, 6, matching Phase A's `whisperflow/stt/providers.py`. `build_recommended_config(rec) -> Config` and `build_config_for_engine(engine_id, specs) -> Config` (Task 3) are the two functions Tasks 5 and 6 both call — same signatures used in both places.
