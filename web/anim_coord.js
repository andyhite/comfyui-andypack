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
