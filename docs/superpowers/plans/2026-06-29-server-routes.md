# Server Routes — Implementation Plan (Plan 4 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the resolver to the frontend over HTTP: `GET /anim_coord/characters`, `/options`, `/resolve`, and a path-traversal-hardened `/frame`. All JSON-payload logic is pure and unit-tested; the aiohttp routes are thin wrappers registered on `PromptServer`.

**Architecture:** `andypack/api.py` (pure stdlib) builds every JSON payload from the manifest + rendered tree and is fully unit-tested. `andypack/server.py` registers `PromptServer.instance.routes`, parses query params, and delegates to `api.*`; `/frame` confines paths with `io.safe_path` (Plan 2) and streams with `web.FileResponse`.

**Tech Stack:** Python ≥3.10, stdlib (`api.py`); `aiohttp` + ComfyUI `server.PromptServer` (`server.py`); `pytest`/`ruff`/`mypy`.

**Prerequisites:** Plans 1–2 complete (`resolve.py`, `manifest.py`, `io.safe_path`). Plan 3 is not required but typically lands first.

**Source of truth:** `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md` §7 (server routes).

## Global Constraints

- `andypack/api.py` is stdlib-only and unit-tested. `andypack/server.py` imports `from server import PromptServer` (ComfyUI-only) and is integration-verified.
- `/frame` MUST reject `..`, absolute paths, and symlink escapes via `io.safe_path`, and 404 anything outside `{root}` or non-existent. It serves untrusted input — treat every request as hostile.
- `status` strings are exactly `ready | generated | blocked | stale` (from `resolve.status`).
- `blocked_by` is rendered as `"<ref>@<dir>"` strings for the UI.
- Preview URLs carry `v=<prompt_hash|"">` as a cache-buster so a re-rendered dependency invalidates the browser thumbnail.
- Every task ends green: `pytest -q`; `ruff check . && mypy andypack`.

---

## File Structure

- `andypack/api.py` — `list_characters`, `list_options`, `resolve_payload`, `frame_path`, `format_blocked`. Pure.
- `andypack/server.py` — registers the four routes on `PromptServer`. Thin.
- `tests/test_api.py` — unit tests for `api.py`.

---

## Task 1: `api.py` — characters + options payloads

**Files:**
- Create: `andypack/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `andypack.resolve` (`resolve_pose`, `resolve_animation`, `status`), `andypack.manifest.node_kind`.
- Produces:
  - `format_blocked(blocked_by: list) -> list[str]`
  - `list_characters(root: str) -> list[dict]` → `[{"name": str}, …]`
  - `list_options(manifest, root, character) -> list[dict]` → `[{"kind","id","direction","status","blocked_by"}, …]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_api.py`:

```python
import os

from andypack import api


def test_list_characters_finds_dirs_with_concept(tree):
    tree.concept()  # creates <root>/Cortex/_concept.png
    os.makedirs(os.path.join(tree.root, "NotAChar"), exist_ok=True)  # empty -> excluded
    names = [c["name"] for c in api.list_characters(tree.root)]
    assert names == ["Cortex"]


def test_format_blocked_renders_ref_at_dir():
    blocked = [{"start_from": {"ref": "fighting_stance_idle"}, "dir": "E"},
               {"end_at": {"ref": "fighting_stance_idle"}, "dir": "E"}]
    assert api.format_blocked(blocked) == ["fighting_stance_idle@E", "fighting_stance_idle@E"]


def test_list_options_reports_status_and_blocked(manifest, tree):
    tree.concept()  # only concept present
    opts = {(o["kind"], o["id"], o["direction"]): o for o in api.list_options(manifest, tree.root, tree.char)}

    assert opts[("pose", "base", "E")]["status"] == "ready"
    assert opts[("pose", "fighting_stance", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["blocked_by"] == [
        "fighting_stance_idle@E", "fighting_stance_idle@E"
    ]
    # base covers three directions
    assert {k for k in opts if k[0] == "pose" and k[1] == "base"} == {
        ("pose", "base", "E"), ("pose", "base", "SE"), ("pose", "base", "S")
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'andypack.api'`

- [ ] **Step 3: Implement the characters/options half of `andypack/api.py`**

```python
"""Pure JSON-payload builders for the anim_coord HTTP routes (stdlib only)."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

from andypack import io
from andypack.manifest import node_kind
from andypack.resolve import (
    read_rendered_hash,
    resolve_animation,
    resolve_pose,
    resolved_dir,
    status,
)

Manifest = dict[str, Any]


def format_blocked(blocked_by: list) -> list[str]:
    """Render resolve blocked_by entries as '<ref>@<dir>' strings."""
    out: list[str] = []
    for entry in blocked_by:
        ddir = entry["dir"]
        for key, dep in entry.items():
            if key == "dir":
                continue
            out.append(f"{dep['ref']}@{ddir}")
    return out


def _is_character(root: str, name: str) -> bool:
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        return False
    if os.path.exists(os.path.join(d, "_concept.png")):
        return True
    try:
        return any(os.path.isdir(os.path.join(d, c)) for c in os.listdir(d))
    except OSError:
        return False


def list_characters(root: str) -> list[dict]:
    """One-level scan of `root` for character directories."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [{"name": n} for n in names if _is_character(root, n)]


def list_options(manifest: Manifest, root: str, character: str) -> list[dict]:
    """Every (pose|animation, direction) with its UI status and blocked_by."""
    out: list[dict] = []
    for pid, pose in manifest.get("poses", {}).items():
        for direction in pose.get("directions", {}):
            r = resolve_pose(manifest, root, character, pid, direction)
            out.append({
                "kind": "pose", "id": pid, "direction": direction,
                "status": status(manifest, root, character, pid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    for aid, anim in manifest.get("animations", {}).items():
        for direction in anim.get("directions", {}):
            r = resolve_animation(manifest, root, character, aid, direction)
            out.append({
                "kind": "animation", "id": aid, "direction": direction,
                "status": status(manifest, root, character, aid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/api.py tests/test_api.py
git commit -m "feat: api payload builders for characters + options"
```

---

## Task 2: `api.py` — resolve payload (with previews) + frame path

**Files:**
- Modify: `andypack/api.py` (append)
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Produces:
  - `resolve_payload(manifest, root, character, ref, direction) -> dict`
  - `frame_path(root: str, rel: str) -> Optional[str]` → confined absolute path or `None`.

- [ ] **Step 1: Write failing tests (append to `tests/test_api.py`)**

```python
def test_resolve_payload_pose_has_source_preview(manifest, tree):
    tree.concept()
    p = api.resolve_payload(manifest, tree.root, tree.char, "base", "E")
    assert p["selectable"] is True
    assert p["source_preview"]["ref"] == "concept"
    assert "/anim_coord/frame?" in p["source_preview"]["url"]


def test_resolve_payload_animation_has_dual_previews(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    p = api.resolve_payload(manifest, tree.root, tree.char, "punch", "E")
    assert p["selectable"] is True
    assert p["start_preview"]["ref"] == "fighting_stance_idle"
    assert p["end_preview"]["ref"] == "fighting_stance_idle"
    assert p["start_preview"]["url"].count("path=") == 1


def test_frame_path_confines_to_root(tree):
    tree.concept()
    ok = api.frame_path(tree.root, os.path.join("Cortex", "_concept.png"))
    assert ok is not None and ok.endswith("_concept.png")
    assert api.frame_path(tree.root, "../escape.png") is None
    assert api.frame_path(tree.root, "Cortex/missing.png") is None  # 404: doesn't exist
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `pytest tests/test_api.py -q`
Expected: FAIL — `AttributeError: module 'andypack.api' has no attribute 'resolve_payload'`

- [ ] **Step 3: Append resolve/frame helpers to `andypack/api.py`**

```python
def _preview(
    manifest: Manifest, root: str, character: str,
    dep_ref: str, dep_dir: str, image_path: Optional[str], stale: bool,
) -> Optional[dict]:
    if not image_path:
        return None
    rel = os.path.relpath(image_path, root)
    version = read_rendered_hash(manifest, root, character, dep_ref, dep_dir) or ""
    url = (
        "/anim_coord/frame?"
        f"root={quote(root, safe='')}&path={quote(rel, safe='')}&v={quote(version, safe='')}"
    )
    return {"ref": dep_ref, "direction": dep_dir, "url": url, "stale": stale}


def resolve_payload(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> dict:
    """Full resolve trimmed to UI fields, with source/dual anchor previews."""
    kind = node_kind(manifest, ref)
    if kind == "pose":
        r = resolve_pose(manifest, root, character, ref, direction)
        frm = manifest["poses"][ref]["from"]
        sdir = resolved_dir(frm, direction)
        return {
            "selectable": r["selectable"],
            "blocked_by": format_blocked(r["blocked_by"]),
            "source_preview": _preview(
                manifest, root, character, frm["ref"], sdir, r["source_image"], bool(r["stale"])
            ),
        }
    r = resolve_animation(manifest, root, character, ref, direction)
    anim = manifest["animations"][ref]
    previews: dict[str, Any] = {"start_preview": None, "end_preview": None}
    for slot, key in (("start_from", "start_preview"), ("end_at", "end_preview")):
        dep = anim.get(slot)
        if not dep:
            continue
        ddir = resolved_dir(dep, direction)
        image = r["start_image"] if slot == "start_from" else r["end_image"]
        previews[key] = _preview(
            manifest, root, character, dep["ref"], ddir, image, slot in r["stale"]
        )
    return {
        "selectable": r["selectable"],
        "blocked_by": format_blocked(r["blocked_by"]),
        **previews,
    }


def frame_path(root: str, rel: str) -> Optional[str]:
    """Confine `rel` under `root` and require it to exist; else None (=> 404)."""
    target = io.safe_path(root, rel)
    if target is None or not os.path.isfile(target):
        return None
    return target
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/api.py tests/test_api.py
git commit -m "feat: api resolve payload with previews + confined frame path"
```

---

## Task 3: `server.py` — register routes on `PromptServer`

ComfyUI-only wiring; verified inside ComfyUI. Keep it a thin translation of query params → `api.*` → `web` responses.

**Files:**
- Create: `andypack/server.py`
- Modify: `andypack/__init__.py` (import `server` so routes register on load)

**Interfaces:**
- Consumes: `andypack.api`, `andypack.manifest.load_manifest`.

- [ ] **Step 1: Implement `andypack/server.py`**

```python
"""ComfyUI HTTP routes for the Animation Coordinator. Registered on import."""

from __future__ import annotations

from aiohttp import web

from andypack import api
from andypack.manifest import load_manifest

try:
    from server import PromptServer  # provided by ComfyUI
    _routes = PromptServer.instance.routes
except Exception:  # pragma: no cover - import-time guard outside ComfyUI
    _routes = None


def _manifest_from_request(request):
    path = request.query.get("manifest", "")
    return load_manifest(path)


if _routes is not None:

    @_routes.get("/anim_coord/characters")
    async def _characters(request):
        root = request.query.get("root", "")
        return web.json_response(api.list_characters(root))

    @_routes.get("/anim_coord/options")
    async def _options(request):
        root = request.query.get("root", "")
        character = request.query.get("character", "")
        manifest = _manifest_from_request(request)
        return web.json_response(api.list_options(manifest, root, character))

    @_routes.get("/anim_coord/resolve")
    async def _resolve(request):
        root = request.query.get("root", "")
        character = request.query.get("character", "")
        ref = request.query.get("id", "")
        direction = request.query.get("direction", "")
        manifest = _manifest_from_request(request)
        return web.json_response(
            api.resolve_payload(manifest, root, character, ref, direction)
        )

    @_routes.get("/anim_coord/frame")
    async def _frame(request):
        root = request.query.get("root", "")
        rel = request.query.get("path", "")
        target = api.frame_path(root, rel)
        if target is None:
            return web.Response(status=404)
        return web.FileResponse(target)
```

Note: `/options` and `/resolve` take a `manifest=<path>` query param so the route can reload the validated manifest server-side; the frontend passes the same path the loader node uses.

- [ ] **Step 2: Register routes by importing `server` in `andypack/__init__.py`**

Update `andypack/__init__.py` to import the server module for its side effect (route registration):

```python
"""comfyui-andypack — Animation Coordinator (dependency-aware FFLF resolver)."""

from andypack import server  # noqa: F401  (registers HTTP routes on import)
from andypack.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
```

- [ ] **Step 3: Verify pure modules still import without aiohttp/ComfyUI**

`server.py` imports `aiohttp`; if absent in the dev env, the `try/except` only guards `server.PromptServer`, not `aiohttp`. Confirm the pure suite is unaffected:
```bash
python3 -c "import andypack.api, andypack.io, andypack.resolve, andypack.manifest; print('pure ok')"
pytest -q
```
Expected: `pure ok`; all tests pass. (`import andypack.server` and full `import andypack` are exercised inside ComfyUI, where `aiohttp` + `server` exist.)

- [ ] **Step 4: Lint + commit**

```bash
ruff check . && mypy andypack
git add andypack/server.py andypack/__init__.py
git commit -m "feat: register anim_coord HTTP routes on PromptServer"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §7 `/characters` → Task 1; `/options` (status + blocked_by) → Task 1; `/resolve` with `start/end/source_preview` + cache-buster `v` → Task 2; `/frame` traversal hardening (`..`/absolute/symlink/404) → Task 2 (`frame_path` over `io.safe_path`) + Task 3 route.
- **Placeholder scan:** none; complete code per step.
- **Type consistency:** `format_blocked` consumes both pose (`{"from":…,"dir":…}`) and animation (`{"start_from"/"end_at":…,"dir":…}`) entry shapes; `resolve_payload` uses `resolve.resolved_dir` and `read_rendered_hash` with the same signatures defined in Plan 1; preview `url` shape matches the `/frame` route's `root`/`path`/`v` params.

## Notes for the implementer

- The `try/except` around `from server import PromptServer` lets `andypack.api` be imported and tested anywhere; the routes only register inside ComfyUI. If you prefer, move the import guard to fail loudly in production — but keep `api.py` import-clean.
- `/frame` returns raw 404 on any path that fails `io.safe_path` or doesn't exist — never echo the attempted path back in the body.
- The browser cache-buster `v` is the dependency's `prompt_hash`; when a dep is re-rendered with a new prompt, its hash changes and the thumbnail URL changes, forcing a refetch.
