import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAG = "[andypack]";
console.info(`${TAG} anim_coord.js loaded`);

// The two selector nodes whose pose/animation + direction widgets become combos
// driven by the connected manifest.
const SELECTOR_NODES = {
  CharacterPoseSelector: { idWidget: "pose", kind: "poses" },
  CharacterAnimationSelector: { idWidget: "animation", kind: "animations" },
};

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

// Turn a STRING widget into a combo over `values` (raw ids/directions — no glyph
// prefixes, so the value the node receives is exactly the id/direction). Never
// clobber to "": if there are no values, leave the widget as-is.
function asCombo(w, values) {
  if (!w || !values || values.length === 0) return;
  w.type = "combo";
  w.options = w.options || {};
  w.options.values = values;
  if (!values.includes(w.value)) w.value = values[0];
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
  if (!g) return null;
  return (g.getNodeById ? g.getNodeById(id) : null) || null;
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

function refreshDirections(node, cfg) {
  const idW = widget(node, cfg.idWidget);
  const map = node.__anim_dirMap || {};
  const dirs = (idW && map[idW.value]) || [];
  asCombo(widget(node, "direction"), dirs);
}

// Pull the manifest's selectable structure and turn the id + direction widgets
// into combos. No-op (leaves plain text widgets) until the manifest is connected.
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
  console.info(
    `${TAG} ${node.comfyClass}: manifest='${manifestName}' -> ${ids.length} ${cfg.kind}`
  );
  if (ids.length === 0) return;
  node.__anim_dirMap = map;
  asCombo(widget(node, cfg.idWidget), ids);
  refreshDirections(node, cfg);
  node.setDirtyCanvas(true, true);
}

function wire(node) {
  const cfg = SELECTOR_NODES[node.comfyClass];
  if (!cfg) return;
  if (node.__anim_wired) {
    refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
    return;
  }
  node.__anim_wired = true;

  const idW = widget(node, cfg.idWidget);
  if (idW) {
    const prev = idW.callback;
    idW.callback = (...a) => {
      prev?.(...a);
      refreshDirections(node, cfg);
      node.setDirtyCanvas(true, true);
    };
  }

  const prevOCC = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    prevOCC?.apply(this, args);
    refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
  };

  refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
}

app.registerExtension({
  name: "andypack.animCoord",
  async setup() {
    console.info(`${TAG} extension registered (setup)`);
  },
  async nodeCreated(node) {
    wire(node);
  },
  // Fires for nodes restored from a saved graph (connections already present).
  async loadedGraphNode(node) {
    wire(node);
  },
});
