import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAG = "[andypack]";
console.debug(`${TAG} anim_coord.js loaded`);

const NO_CHARACTER = "(select character)"; // must match nodes._NO_CHARACTER
// Status indicators prefixed onto combo options.
//   ✅ generated · 🟢 ready · 🟠 stale · 🔴 blocked
const GLYPH = { generated: "✅", ready: "🟢", blocked: "🔴", stale: "🟠" };

// Inputs stay disabled until the extension is live AND the pack's routes answer.
let READY = false;

// Selector nodes and their leaf widget. The cascade is:
//   character (python combo) -> category -> pose|animation -> direction
const SELECTOR_NODES = {
  CharacterPoseSelector: { idWidget: "pose", kind: "pose" },
  CharacterAnimationSelector: { idWidget: "animation", kind: "animation" },
};

const enc = encodeURIComponent;

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

// The raw (un-glyphed) value selected on a combo we manage.
function selectedRaw(node, name) {
  const w = widget(node, name);
  if (!w) return null;
  return w.__anim_raw ?? w.__anim_labelToRaw?.[w.value] ?? w.value;
}

// Build (or refresh) a real combo widget whose options are display labels but
// whose SERIALIZED value is the raw id/direction/category. Python inputs stay
// STRING, so the server accepts the raw value; serializeValue strips the label.
// entries: [{ raw, label }].
function applyCombo(node, name, entries, onChange) {
  if (!entries || entries.length === 0) return null;
  const labels = entries.map((e) => e.label);
  const labelToRaw = Object.fromEntries(entries.map((e) => [e.label, e.raw]));
  const rawToLabel = Object.fromEntries(entries.map((e) => [e.raw, e.label]));
  const widgets = node.widgets || (node.widgets = []);
  const idx = widgets.findIndex((w) => w.name === name);
  const existing = idx >= 0 ? widgets[idx] : null;

  const prevRaw = existing
    ? existing.__anim_raw ?? labelToRaw[existing.value] ?? existing.value
    : null;
  const selLabel = (prevRaw != null && rawToLabel[prevRaw]) || labels[0];

  if (existing && existing.__anim_combo) {
    existing.options = existing.options || {};
    existing.options.values = labels;
    existing.value = selLabel;
    existing.disabled = false;
    existing.__anim_labelToRaw = labelToRaw;
    existing.__anim_raw = labelToRaw[selLabel];
    if (onChange) existing.__anim_onChange = onChange;
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
      w.__anim_onChange?.(w.__anim_raw);
    },
    { values: labels }
  );
  w.__anim_combo = true;
  w.disabled = false;
  w.__anim_labelToRaw = labelToRaw;
  w.__anim_raw = labelToRaw[selLabel];
  w.__anim_onChange = onChange;
  w.serializeValue = () => w.__anim_labelToRaw?.[w.value] ?? w.value;
  const at = widgets.indexOf(w);
  if (at >= 0 && idx >= 0 && at !== idx) {
    widgets.splice(at, 1);
    widgets.splice(idx, 0, w);
  }
  return w;
}

// A disabled, single-entry placeholder combo (no real selection yet).
function setPlaceholder(node, name, text) {
  const w = applyCombo(node, name, [{ raw: "", label: text }]);
  if (w) w.disabled = true;
}

function setCharacterEnabled(node, enabled) {
  const w = widget(node, "character");
  if (w) w.disabled = !enabled;
}

// Everything off, with a single message — used until the extension+API are ready.
function lockAll(node, cfg, text) {
  setCharacterEnabled(node, false);
  setPlaceholder(node, "category", text);
  setPlaceholder(node, cfg.idWidget, text);
  setPlaceholder(node, "direction", text);
  if (node.__anim_previewHost) node.__anim_previewHost.innerHTML = "";
  node.setDirtyCanvas(true, true);
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

function manifestNameFor(node) {
  const input = (node.inputs || []).find((i) => i.name === "manifest");
  if (!input || input.link == null) return null;
  const link = graphLink(input.link);
  const loader = link ? getNode(link.origin_id) : null;
  const w = loader && widget(loader, "manifest");
  return w && w.value ? w.value : null;
}

function characterValue(node) {
  const w = widget(node, "character");
  return w ? w.value : null;
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

const categoryOf = (o) => o.category || "(all)";

// Most-actionable status for a group: blocked only when EVERY option is blocked.
function groupStatus(opts) {
  const st = opts.map((o) => o.status);
  if (st.every((s) => s === "blocked")) return "blocked";
  if (st.includes("ready")) return "ready";
  if (st.includes("stale")) return "stale";
  return "generated";
}

// --- the cascade ------------------------------------------------------------ //

async function refreshCascade(node, cfg) {
  setCharacterEnabled(node, true); // READY is guaranteed by the caller (wire)
  const character = characterValue(node);
  const manifestName = manifestNameFor(node);
  if (!character || character === NO_CHARACTER || !manifestName) {
    node.__anim_opts = null;
    setPlaceholder(node, "category", character && character !== NO_CHARACTER
      ? "(connect manifest)" : "(select character)");
    setPlaceholder(node, cfg.idWidget, "(select category)");
    setPlaceholder(node, "direction", `(select ${cfg.kind})`);
    if (node.__anim_previewHost) node.__anim_previewHost.innerHTML = "";
    node.setDirtyCanvas(true, true);
    return;
  }
  // Show a loading state on the downstream widgets while /options is in flight.
  setPlaceholder(node, "category", "(loading…)");
  setPlaceholder(node, cfg.idWidget, "(loading…)");
  setPlaceholder(node, "direction", "(loading…)");
  node.setDirtyCanvas(true, true);

  const url = `/anim_coord/options?manifest=${enc(manifestName)}&character=${enc(character)}`;
  const opts = await fetchJSON(url);
  if (!opts) {
    setPlaceholder(node, "category", "(failed to load)");
    node.setDirtyCanvas(true, true);
    return;
  }
  node.__anim_opts = opts.filter((o) => o.kind === cfg.kind);
  buildCategoryCombo(node, cfg);
}

function buildCategoryCombo(node, cfg) {
  const opts = node.__anim_opts || [];
  const cats = [...new Set(opts.map(categoryOf))];
  const entries = cats.map((c) => ({ raw: c, label: c }));
  applyCombo(node, "category", entries, () => buildIdCombo(node, cfg));
  buildIdCombo(node, cfg);
}

function buildIdCombo(node, cfg) {
  const opts = node.__anim_opts || [];
  const cat = selectedRaw(node, "category");
  const inCat = opts.filter((o) => categoryOf(o) === cat);
  const byId = {};
  for (const o of inCat) (byId[o.id] ||= []).push(o);
  node.__anim_byId = byId;
  const entries = Object.keys(byId).map((id) => ({
    raw: id,
    label: `${GLYPH[groupStatus(byId[id])]} ${id}`,
  }));
  applyCombo(node, cfg.idWidget, entries, () => buildDirectionCombo(node, cfg));
  buildDirectionCombo(node, cfg);
}

function buildDirectionCombo(node, cfg) {
  const id = selectedRaw(node, cfg.idWidget);
  const opts = (node.__anim_byId && node.__anim_byId[id]) || [];
  const entries = opts.map((o) => ({ raw: o.direction, label: `${GLYPH[o.status]} ${o.direction}` }));
  applyCombo(node, "direction", entries, () => renderPreviews(node, cfg));
  renderPreviews(node, cfg).catch(() => {});
  node.setDirtyCanvas(true, true);
}

// --- anchor thumbnails (guarded DOM widget) --------------------------------- //

function previewCard(label, preview) {
  const card = document.createElement("div");
  card.style.cssText =
    "display:flex;flex-direction:column;align-items:center;font-size:10px;gap:2px;flex:1;color:#bbb";
  const cap = document.createElement("div");
  cap.textContent = preview ? `${label}: ${preview.ref}.${preview.direction}` : `${label}: —`;
  card.appendChild(cap);
  if (preview && preview.url) {
    const img = document.createElement("img");
    img.src = api.apiURL ? api.apiURL(preview.url) : preview.url;
    img.style.cssText =
      "width:72px;height:72px;object-fit:contain;border:1px solid #444;background:#222";
    if (preview.stale) img.style.outline = "2px solid #d9a521";
    card.appendChild(img);
  }
  return card;
}

async function renderPreviews(node, cfg) {
  const host = node.__anim_previewHost;
  if (!host) return;
  const character = characterValue(node);
  const id = selectedRaw(node, cfg.idWidget);
  const direction = selectedRaw(node, "direction");
  const manifestName = manifestNameFor(node);
  if (!character || character === NO_CHARACTER || !id || !direction || !manifestName) {
    host.innerHTML = "";
    return;
  }
  const url =
    `/anim_coord/resolve?manifest=${enc(manifestName)}&character=${enc(character)}` +
    `&id=${enc(id)}&direction=${enc(direction)}`;
  const data = await fetchJSON(url);
  host.innerHTML = "";
  if (!data) return;
  if (cfg.kind === "pose") {
    host.appendChild(previewCard("from", data.source_preview));
  } else {
    host.appendChild(previewCard("start", data.start_preview));
    host.appendChild(previewCard("end", data.end_preview));
  }
}

function ensurePreviewHost(node) {
  if (node.__anim_previewHost || typeof node.addDOMWidget !== "function") return;
  const host = document.createElement("div");
  host.style.cssText = "display:flex;gap:6px;padding:4px;width:100%";
  try {
    node.addDOMWidget("anim_preview", "preview", host, {});
    node.__anim_previewHost = host;
  } catch (e) {
    console.warn(`${TAG} addDOMWidget unavailable`, e);
  }
}

function wire(node) {
  const cfg = SELECTOR_NODES[node.comfyClass];
  if (!cfg) return;
  ensurePreviewHost(node);
  if (!node.__anim_wired) {
    node.__anim_wired = true;
    const prevOCC = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      prevOCC?.apply(this, args);
      if (READY) refreshCascade(node, cfg).catch((e) => console.warn(`${TAG} cascade`, e));
    };
    const charW = widget(node, "character");
    if (charW) {
      const prev = charW.callback;
      charW.callback = (...a) => {
        prev?.(...a);
        if (READY) refreshCascade(node, cfg).catch((e) => console.warn(`${TAG} cascade`, e));
      };
    }
  }
  if (!READY) {
    lockAll(node, cfg, "(loading…)");
    return;
  }
  refreshCascade(node, cfg).catch((e) => console.warn(`${TAG} cascade`, e));
}

function refreshAll() {
  for (const node of app.graph?._nodes || []) {
    if (SELECTOR_NODES[node.comfyClass]) wire(node);
  }
}

app.registerExtension({
  name: "andypack.animCoord",
  async setup() {
    console.debug(`${TAG} extension registered (setup)`);
    api.addEventListener("execution_success", refreshAll);
    api.addEventListener("executed", refreshAll);
    // Only enable the selector inputs once the pack's routes answer.
    const ok = await fetchJSON("/anim_coord/ping");
    READY = !!(ok && ok.ok);
    console.debug(`${TAG} api ready: ${READY}`);
    refreshAll();
  },
  async nodeCreated(node) {
    wire(node);
  },
  async loadedGraphNode(node) {
    wire(node);
  },
});
