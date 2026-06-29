import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
// clobber to "": if there are no values, leave the widget as-is so the user's
// typed value still submits.
function asCombo(w, values) {
  if (!w || !values || values.length === 0) return;
  w.type = "combo";
  w.options = w.options || {};
  w.options.values = values;
  if (!values.includes(w.value)) w.value = values[0];
}

// Trace the selector's `manifest` input link back to the AnimationManifestLoader
// and read its chosen manifest filename. Returns null if not connected yet.
function manifestNameFor(node) {
  const input = (node.inputs || []).find((i) => i.name === "manifest");
  if (!input || input.link == null) return null;
  const link = app.graph.links[input.link];
  if (!link) return null;
  const loader = app.graph.getNodeById(link.origin_id);
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
    return null;
  }
  return res.ok ? await res.json() : null;
}

// Repopulate the direction combo for the currently-selected id.
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
  if (!manifestName) return;
  const data = await fetchManifestOptions(manifestName);
  if (!data) return;
  const map = data[cfg.kind] || {};
  const ids = Object.keys(map);
  if (ids.length === 0) return;
  node.__anim_dirMap = map;
  asCombo(widget(node, cfg.idWidget), ids);
  refreshDirections(node, cfg);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "andypack.animCoord",
  async nodeCreated(node) {
    const cfg = SELECTOR_NODES[node.comfyClass];
    if (!cfg) return;

    // When the pose/animation changes, the valid directions change with it.
    const idW = widget(node, cfg.idWidget);
    if (idW) {
      const prev = idW.callback;
      idW.callback = (...a) => {
        prev?.(...a);
        refreshDirections(node, cfg);
        node.setDirtyCanvas(true, true);
      };
    }

    // Repopulate whenever the manifest link is (dis)connected.
    const prevOCC = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      prevOCC?.apply(this, args);
      refreshCombos(node, cfg).catch(() => {});
    };

    // Initial attempt (the manifest may not be wired yet — onConnectionsChange
    // will catch it when it is).
    refreshCombos(node, cfg).catch(() => {});
  },
});
