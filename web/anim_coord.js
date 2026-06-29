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

    refreshOptions(node, cfg);
  },
});
