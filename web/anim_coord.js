import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAG = "[andypack]";
console.debug(`${TAG} anim_coord.js loaded`);

// The two selector nodes whose pose/animation + direction widgets become combos
// driven by the connected manifest.
const SELECTOR_NODES = {
  CharacterPoseSelector: { idWidget: "pose", kind: "poses" },
  CharacterAnimationSelector: { idWidget: "animation", kind: "animations" },
};

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

// Replace the named widget with a REAL combo widget (or refresh an existing
// one's values). The Python input stays STRING, so the server accepts whatever
// the combo submits; we only change the client-side rendering. Just flipping
// widget.type = "combo" does NOT render as a dropdown in @comfyorg/litegraph —
// the widget has to be constructed as a combo via node.addWidget.
function applyCombo(node, name, values, onChange) {
  if (!values || values.length === 0) return null;
  const widgets = node.widgets || (node.widgets = []);
  const idx = widgets.findIndex((w) => w.name === name);
  const existing = idx >= 0 ? widgets[idx] : null;

  // Already our combo -> just update its option list, keep a valid selection.
  if (existing && existing.__anim_combo) {
    existing.options = existing.options || {};
    existing.options.values = values;
    if (!values.includes(existing.value)) existing.value = values[0];
    return existing;
  }

  let value = existing ? existing.value : values[0];
  if (!values.includes(value)) value = values[0];

  if (typeof node.addWidget !== "function") {
    // Fallback for older litegraph: best-effort in-place mutation.
    if (existing) {
      existing.type = "combo";
      existing.options = { ...(existing.options || {}), values };
      existing.value = value;
    }
    return existing;
  }

  if (idx >= 0) widgets.splice(idx, 1); // drop the STRING text widget
  const w = node.addWidget("combo", name, value, (v) => onChange?.(v), { values });
  w.__anim_combo = true;
  // addWidget appends — move it back to the original slot to preserve layout.
  const at = widgets.indexOf(w);
  if (at >= 0 && idx >= 0 && at !== idx) {
    widgets.splice(at, 1);
    widgets.splice(idx, 0, w);
  }
  return w;
}

// litegraph's graph.links is a plain object in older versions and a Map in newer
// ones — read it tolerantly.
function graphLink(linkId) {
  const links = app.graph && app.graph.links;
  if (!links) return null;
  return typeof links.get === "function" ? links.get(linkId) : links[linkId];
}

function getNode(id) {
  const g = app.graph;
  return (g && g.getNodeById && g.getNodeById(id)) || null;
}

// Trace the selector's `manifest` input link back to the AnimationManifestLoader
// and read its chosen manifest filename. Returns null if not connected yet.
function manifestNameFor(node) {
  const input = (node.inputs || []).find((i) => i.name === "manifest");
  if (!input || input.link == null) return null;
  const link = graphLink(input.link);
  if (!link) return null;
  const loader = getNode(link.origin_id);
  if (!loader) return null;
  const w = widget(loader, "manifest");
  return w && w.value ? w.value : null;
}

async function fetchManifestOptions(manifestName) {
  const url = `/anim_coord/manifest_options?manifest=${encodeURIComponent(manifestName)}`;
  let res;
  try {
    res = await api.fetchApi(url);
  } catch (e) {
    console.warn(`${TAG} manifest_options fetch failed`, e);
    return null;
  }
  if (!res.ok) {
    console.warn(`${TAG} manifest_options ${res.status} for`, manifestName);
    return null;
  }
  return res.json();
}

// Narrow the direction combo to the directions valid for the selected id.
function refreshDirections(node, cfg) {
  const idW = widget(node, cfg.idWidget);
  const map = node.__anim_dirMap || {};
  const dirs = (idW && map[idW.value]) || [];
  applyCombo(node, "direction", dirs);
  node.setDirtyCanvas(true, true);
}

// Pull the manifest's selectable structure and build the id + direction combos.
// No-op (leaves the plain text widgets) until the manifest is connected.
async function refreshCombos(node, cfg) {
  const manifestName = manifestNameFor(node);
  if (!manifestName) {
    console.debug(`${TAG} ${node.comfyClass}: no manifest connected yet`);
    return;
  }
  const data = await fetchManifestOptions(manifestName);
  if (!data) return;
  const map = data[cfg.kind] || {};
  const ids = Object.keys(map);
  if (ids.length === 0) return;
  node.__anim_dirMap = map;
  applyCombo(node, cfg.idWidget, ids, () => refreshDirections(node, cfg));
  refreshDirections(node, cfg);
  node.setDirtyCanvas(true, true);
  console.debug(
    `${TAG} ${node.comfyClass}: ${cfg.idWidget} combo set (${ids.length})`
  );
}

function wire(node) {
  const cfg = SELECTOR_NODES[node.comfyClass];
  if (!cfg) return;
  if (!node.__anim_wired) {
    node.__anim_wired = true;
    const prevOCC = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      prevOCC?.apply(this, args);
      refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
    };
  }
  refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
}

app.registerExtension({
  name: "andypack.animCoord",
  async setup() {
    console.debug(`${TAG} extension registered (setup)`);
  },
  async nodeCreated(node) {
    wire(node);
  },
  // Fires for nodes restored from a saved graph (connections already present).
  async loadedGraphNode(node) {
    wire(node);
  },
});
