import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAG = "[andypack]";
console.debug(`${TAG} anim_coord.js loaded`);

// Status glyphs shown as a prefix on each combo option.
const GLYPH = { generated: "✓", ready: "○", blocked: "⨯", stale: "▲" };

// Selector nodes whose id (pose/animation) + direction widgets become combos
// driven by the connected manifest + character. `kind` matches /options.
const SELECTOR_NODES = {
  CharacterPoseSelector: { idWidget: "pose", kind: "pose" },
  CharacterAnimationSelector: { idWidget: "animation", kind: "animation" },
};

const enc = encodeURIComponent;

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

// The raw (un-glyphed) value currently selected on a combo we manage.
function selectedRaw(node, name) {
  const w = widget(node, name);
  if (!w) return null;
  return w.__anim_raw ?? w.__anim_labelToRaw?.[w.value] ?? w.value;
}

// Build (or refresh) a real combo widget whose options are glyph-prefixed labels
// but whose SERIALIZED value is the raw id/direction. The Python input stays
// STRING, so the server accepts the raw value; serializeValue strips the glyph.
// entries: [{ raw, label }].
function applyCombo(node, name, entries, onChange) {
  if (!entries || entries.length === 0) return null;
  const labels = entries.map((e) => e.label);
  const labelToRaw = Object.fromEntries(entries.map((e) => [e.label, e.raw]));
  const rawToLabel = Object.fromEntries(entries.map((e) => [e.raw, e.label]));
  const widgets = node.widgets || (node.widgets = []);
  const idx = widgets.findIndex((w) => w.name === name);
  const existing = idx >= 0 ? widgets[idx] : null;

  // Preserve the current selection by its RAW value across refreshes/reloads.
  const prevRaw = existing
    ? existing.__anim_raw ?? labelToRaw[existing.value] ?? existing.value
    : null;
  const selLabel = (prevRaw != null && rawToLabel[prevRaw]) || labels[0];

  if (existing && existing.__anim_combo) {
    existing.options = existing.options || {};
    existing.options.values = labels;
    existing.value = selLabel;
    existing.__anim_labelToRaw = labelToRaw;
    existing.__anim_raw = labelToRaw[selLabel];
    return existing;
  }

  if (typeof node.addWidget !== "function") return existing;
  if (idx >= 0) widgets.splice(idx, 1);
  const w = node.addWidget(
    "combo",
    name,
    selLabel,
    (v) => {
      w.__anim_raw = w.__anim_labelToRaw?.[v] ?? v;
      onChange?.(w.__anim_raw);
    },
    { values: labels }
  );
  w.__anim_combo = true;
  w.__anim_labelToRaw = labelToRaw;
  w.__anim_raw = labelToRaw[selLabel];
  w.serializeValue = () => w.__anim_labelToRaw?.[w.value] ?? w.value;
  const at = widgets.indexOf(w);
  if (at >= 0 && idx >= 0 && at !== idx) {
    widgets.splice(at, 1);
    widgets.splice(idx, 0, w);
  }
  return w;
}

// litegraph's graph.links is a plain object (older) or a Map (newer).
function graphLink(linkId) {
  const links = app.graph && app.graph.links;
  if (!links) return null;
  return typeof links.get === "function" ? links.get(linkId) : links[linkId];
}

function getNode(id) {
  const g = app.graph;
  return (g && g.getNodeById && g.getNodeById(id)) || null;
}

// Find the node feeding `inputName`, if connected.
function sourceNode(node, inputName) {
  const input = (node.inputs || []).find((i) => i.name === inputName);
  if (!input || input.link == null) return null;
  const link = graphLink(input.link);
  return link ? getNode(link.origin_id) : null;
}

function manifestNameFor(node) {
  const loader = sourceNode(node, "manifest");
  const w = loader && widget(loader, "manifest");
  return w && w.value ? w.value : null;
}

// Character context: a connected CharacterSelector's chosen folder name, else
// the node's own character_dir widget text.
function characterQuery(node) {
  const src = sourceNode(node, "character_dir");
  const nameW = src && widget(src, "character");
  if (nameW && nameW.value) return `&character=${enc(nameW.value)}`;
  const dirW = widget(node, "character_dir");
  if (dirW && dirW.value) return `&character_dir=${enc(dirW.value)}`;
  return "";
}

async function fetchJSON(url) {
  let res;
  try {
    res = await api.fetchApi(url);
  } catch (e) {
    console.warn(`${TAG} fetch failed`, url, e);
    return null;
  }
  if (!res.ok) {
    console.warn(`${TAG} ${res.status} for`, url);
    return null;
  }
  return res.json();
}

// Worst/most-actionable status for an id across its directions: blocked only
// when EVERY direction is blocked; otherwise prefer ready > stale > generated.
function idStatus(opts) {
  const st = opts.map((o) => o.status);
  if (st.every((s) => s === "blocked")) return "blocked";
  if (st.includes("ready")) return "ready";
  if (st.includes("stale")) return "stale";
  return "generated";
}

function refreshDirections(node, cfg) {
  const id = selectedRaw(node, cfg.idWidget);
  const opts = (node.__anim_byId && node.__anim_byId[id]) || [];
  const entries = opts.map((o) => ({ raw: o.direction, label: `${GLYPH[o.status]} ${o.direction}` }));
  applyCombo(node, "direction", entries);
  node.setDirtyCanvas(true, true);
}

async function refreshCombos(node, cfg) {
  const manifestName = manifestNameFor(node);
  if (!manifestName) {
    console.debug(`${TAG} ${node.comfyClass}: no manifest connected yet`);
    return;
  }
  const url = `/anim_coord/options?manifest=${enc(manifestName)}${characterQuery(node)}`;
  const opts = await fetchJSON(url);
  if (!opts) return;
  const mine = opts.filter((o) => o.kind === cfg.kind);
  if (mine.length === 0) return;

  const byId = {};
  for (const o of mine) (byId[o.id] ||= []).push(o);
  node.__anim_byId = byId;

  const entries = Object.keys(byId).map((id) => ({
    raw: id,
    label: `${GLYPH[idStatus(byId[id])]} ${id}`,
  }));
  applyCombo(node, cfg.idWidget, entries, () => refreshDirections(node, cfg));
  refreshDirections(node, cfg);
  node.setDirtyCanvas(true, true);
  console.debug(`${TAG} ${node.comfyClass}: ${entries.length} ${cfg.kind}(s)`);
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

function refreshAll() {
  for (const node of app.graph?._nodes || []) {
    const cfg = SELECTOR_NODES[node.comfyClass];
    if (cfg) refreshCombos(node, cfg).catch((e) => console.warn(`${TAG} refresh`, e));
  }
}

app.registerExtension({
  name: "andypack.animCoord",
  async setup() {
    console.debug(`${TAG} extension registered (setup)`);
    // Re-evaluate status (glyphs) after a writer runs.
    api.addEventListener("execution_success", refreshAll);
    api.addEventListener("executed", refreshAll);
  },
  async nodeCreated(node) {
    wire(node);
  },
  async loadedGraphNode(node) {
    wire(node);
  },
});
