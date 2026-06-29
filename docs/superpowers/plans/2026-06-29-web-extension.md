# Web Extension — Implementation Plan (Plan 5 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ComfyUI frontend extension (`web/anim_coord.js`) that turns the Selector nodes into dependency-aware progress boards: character-scoped dynamic combos, status glyphs, source/dual anchor thumbnails, amber stale tinting, and auto-refresh after a writer runs.

**Architecture:** A single `app.registerExtension` module. On the `CharacterPoseSelector` and `CharacterAnimationSelector` nodes it converts the `character`/`pose|animation`/`direction` widgets into combos populated from `/anim_coord/characters` and `/anim_coord/options`, renders status glyphs into the option labels, draws previews fetched from `/anim_coord/resolve` into a DOM widget, and re-fetches options on the execution-finished event.

**Tech Stack:** Browser JS (ES modules) using ComfyUI's `scripts/app.js` + `scripts/api.js`. No build step. Not unit-testable — verified manually in a running ComfyUI.

**Prerequisites:** Plans 1–4 complete (nodes registered, routes live, `WEB_DIRECTORY = "./web"` set in Plan 2).

**Source of truth:** `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md` §7 (frontend extension).

## Global Constraints

- File lives at `web/anim_coord.js` (matches `WEB_DIRECTORY = "./web"`).
- Status glyphs: `✓ generated`, `○ ready`, `⨯ blocked`, `▲ stale`.
- Stale previews tint amber; blocked options stay visible (greyed) so the node doubles as a progress board.
- All server calls go through `api.fetchApi(...)` so they inherit ComfyUI's base path/auth.
- The frontend passes the same `root`, `manifest` path, and `character` the user configured on the node, as query params.
- **Verification is manual in ComfyUI** — there is no pytest gate. Each task lists concrete in-app checks. Confirm the exact widget/event API names against the installed ComfyUI frontend version before relying on them (they evolve); adjust if the version differs.

---

## File Structure

- `web/anim_coord.js` — the entire extension (one module, built up across the three tasks below).

---

## Task 1: Extension skeleton + dynamic combos + status glyphs

**Files:**
- Create: `web/anim_coord.js`

- [ ] **Step 1: Implement the skeleton with combo population**

Create `web/anim_coord.js`:

```javascript
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SELECTOR_NODES = {
  CharacterPoseSelector: { idWidget: "pose", kind: "pose" },
  CharacterAnimationSelector: { idWidget: "animation", kind: "animation" },
};

const GLYPH = { generated: "✓", ready: "○", blocked: "⨯", stale: "▲" };

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function widgetValue(node, name, fallback = "") {
  const w = widget(node, name);
  return w ? w.value : fallback;
}

// Turn a STRING widget into a combo with the given values, preserving selection.
function asCombo(w, values) {
  if (!w) return;
  w.type = "combo";
  w.options = w.options || {};
  w.options.values = values;
  if (!values.includes(w.value)) w.value = values[0] ?? "";
}

function queryBase(node) {
  return {
    root: widgetValue(node, "root_dir"),
    manifest: widgetValue(node, "manifest_path", "animations.json"),
    character: widgetValue(node, "character"),
  };
}

async function fetchCharacters(node) {
  const { root } = queryBase(node);
  if (!root) return [];
  const res = await api.fetchApi(`/anim_coord/characters?root=${encodeURIComponent(root)}`);
  return res.ok ? await res.json() : [];
}

async function fetchOptions(node) {
  const { root, manifest, character } = queryBase(node);
  if (!root || !character) return [];
  const url =
    `/anim_coord/options?root=${encodeURIComponent(root)}` +
    `&manifest=${encodeURIComponent(manifest)}` +
    `&character=${encodeURIComponent(character)}`;
  const res = await api.fetchApi(url);
  return res.ok ? await res.json() : [];
}

// Build "id  ✓" labels for the id combo and a list of valid directions for the selected id.
async function refreshOptions(node, cfg) {
  const options = await fetchOptions(node);
  const kindOptions = options.filter((o) => o.kind === cfg.kind);

  const ids = [...new Set(kindOptions.map((o) => o.id))];
  const labelFor = (id) => {
    const statuses = kindOptions.filter((o) => o.id === id).map((o) => o.status);
    const worst = statuses.includes("blocked") ? "blocked"
      : statuses.includes("stale") ? "stale"
      : statuses.includes("ready") ? "ready" : "generated";
    return `${GLYPH[worst]} ${id}`;
  };

  const idW = widget(node, cfg.idWidget);
  // store raw id <-> label maps so we can translate on submit
  node.__anim_idMap = Object.fromEntries(ids.map((id) => [labelFor(id), id]));
  asCombo(idW, ids.map(labelFor));

  refreshDirections(node, cfg, kindOptions);
  node.setDirtyCanvas(true, true);
}

function selectedId(node, cfg) {
  const raw = widgetValue(node, cfg.idWidget);
  return (node.__anim_idMap && node.__anim_idMap[raw]) || raw;
}

function refreshDirections(node, cfg, kindOptions) {
  const id = selectedId(node, cfg);
  const dirs = kindOptions.filter((o) => o.id === id);
  const dirW = widget(node, "direction");
  node.__anim_dirStatus = Object.fromEntries(dirs.map((o) => [o.direction, o.status]));
  asCombo(dirW, dirs.map((o) => `${GLYPH[o.status]} ${o.direction}`));
  node.__anim_dirMap = Object.fromEntries(dirs.map((o) => [`${GLYPH[o.status]} ${o.direction}`, o.direction]));
}

function selectedDirection(node) {
  const raw = widgetValue(node, "direction");
  return (node.__anim_dirMap && node.__anim_dirMap[raw]) || raw;
}

app.registerExtension({
  name: "andypack.animCoord",
  async nodeCreated(node) {
    const cfg = SELECTOR_NODES[node.comfyClass];
    if (!cfg) return;

    const charW = widget(node, "character");
    fetchCharacters(node).then((chars) => asCombo(charW, chars.map((c) => c.name)));

    if (charW) {
      const prev = charW.callback;
      charW.callback = (...a) => { prev?.(...a); refreshOptions(node, cfg); };
    }
    const idW = widget(node, cfg.idWidget);
    if (idW) {
      const prev = idW.callback;
      idW.callback = (...a) => { prev?.(...a); refreshOptions(node, cfg); };
    }

    node.__anim_cfg = cfg;
    refreshOptions(node, cfg);
  },
});
```

- [ ] **Step 2: Verify in ComfyUI (manual)**

1. Restart ComfyUI; add a **Character Pose Selector** node.
2. Set `root_dir` to a folder containing `Cortex/_concept.png` and `manifest_path` to your `animations.json`.
3. The `character` widget should become a combo listing `Cortex`; selecting it repopulates `pose` with `⨯/○/✓`-prefixed entries.
4. The `pose` combo shows `base` as `○`/`✓` and `fighting_stance` as `⨯` when only the concept (or only base) exists.

Expected: combos populate; glyphs reflect status. If widget mutation doesn't render, confirm the frontend's combo widget shape (`w.type`/`w.options.values`) for your version.

- [ ] **Step 3: Commit**

```bash
git add web/anim_coord.js
git commit -m "feat(web): dynamic character/id/direction combos with status glyphs"
```

---

## Task 2: Anchor previews + amber stale tint

**Files:**
- Modify: `web/anim_coord.js`

- [ ] **Step 1: Add a preview DOM widget and resolve-driven rendering**

Append to `web/anim_coord.js` (above `app.registerExtension`, add the helpers; then wire them in `nodeCreated`):

```javascript
async function fetchResolve(node, cfg) {
  const { root, manifest, character } = queryBase(node);
  const id = selectedId(node, cfg);
  const direction = selectedDirection(node);
  if (!root || !character || !id || !direction) return null;
  const url =
    `/anim_coord/resolve?root=${encodeURIComponent(root)}` +
    `&manifest=${encodeURIComponent(manifest)}` +
    `&character=${encodeURIComponent(character)}` +
    `&id=${encodeURIComponent(id)}` +
    `&direction=${encodeURIComponent(direction)}`;
  const res = await api.fetchApi(url);
  return res.ok ? await res.json() : null;
}

function previewCard(label, preview) {
  const card = document.createElement("div");
  card.style.cssText = "display:flex;flex-direction:column;align-items:center;font-size:10px;gap:2px;flex:1";
  const cap = document.createElement("div");
  cap.textContent = preview ? `${label} ${preview.ref}.${preview.direction}` : `${label} none`;
  card.appendChild(cap);
  if (preview && preview.url) {
    const img = document.createElement("img");
    img.src = api.apiURL(preview.url);
    img.style.cssText = "width:64px;height:64px;object-fit:contain;border:1px solid #444";
    if (preview.stale) img.style.outline = "2px solid #d9a521"; // amber
    card.appendChild(img);
  }
  return card;
}

async function renderPreviews(node, cfg) {
  const host = node.__anim_previewHost;
  if (!host) return;
  host.innerHTML = "";
  const data = await fetchResolve(node, cfg);
  if (!data) return;
  if (cfg.kind === "pose") {
    host.appendChild(previewCard("from →", data.source_preview));
  } else {
    host.appendChild(previewCard("starts →", data.start_preview));
    host.appendChild(previewCard("ends →", data.end_preview));
  }
}
```

Then, inside `nodeCreated` (after `node.__anim_cfg = cfg;`), add the DOM widget and hook direction/id changes:

```javascript
    const host = document.createElement("div");
    host.style.cssText = "display:flex;gap:6px;padding:4px;width:100%";
    node.__anim_previewHost = host;
    node.addDOMWidget("anim_preview", "preview", host, {});

    const dirW = widget(node, "direction");
    if (dirW) {
      const prev = dirW.callback;
      dirW.callback = (...a) => { prev?.(...a); renderPreviews(node, cfg); };
    }
    const idW2 = widget(node, cfg.idWidget);
    if (idW2) {
      const prev = idW2.callback;
      idW2.callback = (...a) => { prev?.(...a); refreshOptions(node, cfg).then(() => renderPreviews(node, cfg)); };
    }
```

- [ ] **Step 2: Verify in ComfyUI (manual)**

1. With `base@E` generated and `fighting_stance@E` generated, select an **Animation Selector**, choose `punch`, direction `E`.
2. Two thumbnails appear: "starts → fighting_stance_idle.E" and "ends → fighting_stance_idle.E" once idle is generated.
3. Edit the `base` pose prompt in the manifest and reload it → the stance previews gain an amber outline (stale) but stay selectable.

Expected: previews render; amber outline appears on stale. If `addDOMWidget` differs in your version, substitute the equivalent DOM-widget API.

- [ ] **Step 3: Commit**

```bash
git add web/anim_coord.js
git commit -m "feat(web): source/dual anchor previews with amber stale tint"
```

---

## Task 3: Auto-refresh after a writer runs

**Files:**
- Modify: `web/anim_coord.js`

- [ ] **Step 1: Refresh options on execution finished**

Inside `app.registerExtension({...})`, add a `setup` hook that listens for the execution-finished event and refreshes every selector node on the graph:

```javascript
  async setup() {
    const refreshAll = () => {
      for (const node of app.graph?._nodes || []) {
        const cfg = SELECTOR_NODES[node.comfyClass];
        if (cfg) refreshOptions(node, cfg).then(() => renderPreviews(node, cfg));
      }
    };
    api.addEventListener("execution_success", refreshAll);
    api.addEventListener("executed", refreshAll);
  },
```

(Place `setup` as a sibling of `name` and `nodeCreated` in the `registerExtension` object.)

- [ ] **Step 2: Verify in ComfyUI (manual)**

1. With only the concept present, queue a graph that generates `base@E` via `CharacterPoseSelector` → FLUX → `PoseFrameWriter`.
2. When it finishes, the other selector nodes' combos update **without a manual reselect** — `fighting_stance` flips from `⨯` to `○`.
3. Generate up through `fighting_stance_idle`; `punch`/`entry`/`exit` flip to `○` automatically.

Expected: combos and previews refresh on execution finish. If neither event name fires in your version, check the frontend's event list (`api.addEventListener` names) and use the execution-completed event it exposes.

- [ ] **Step 3: Commit**

```bash
git add web/anim_coord.js
git commit -m "feat(web): auto-refresh selectors on execution finished"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §7 dynamic combos repopulated on character change → Task 1; status glyphs (`✓ ○ ⨯ ▲`) → Task 1; dual/source previews drawn in the node body → Task 2; amber stale tint → Task 2; auto-refresh on execution-finished → Task 3.
- **Placeholder scan:** none; full JS in each step.
- **Type consistency:** query params (`root`, `manifest`, `character`, `id`, `direction`) match Plan 4's route parsing; preview object keys (`ref`, `direction`, `url`, `stale`) match `api._preview`; option keys (`kind`, `id`, `direction`, `status`, `blocked_by`) match `api.list_options`.

## Notes for the implementer

- ComfyUI frontend internals (widget mutation, `addDOMWidget`, event names) shift between versions. This module uses widely-supported shapes, but **verify against the installed frontend** and adjust the three touch-points if needed: (1) combo widget = `{type:"combo", options:{values:[…]}}`, (2) `node.addDOMWidget(name, type, element, opts)`, (3) execution-finished event name on `api`.
- The id/direction combos carry glyph-prefixed labels for display; the `__anim_idMap`/`__anim_dirMap` translate the label back to the raw id/direction. Ensure the **raw** value is what the node submits — if the frontend serializes `w.value` (the label) into the prompt, strip the glyph in a `beforeQueued`/serialize hook so the Python node receives the bare id/direction.
- Blocked options remain in the combo (greyed via glyph) by design — the node doubles as a progress board. If you'd rather hide them, filter `kindOptions` by `status !== "blocked"` behind a toggle widget.
```
