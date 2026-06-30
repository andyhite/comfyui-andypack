// andypack sidebar panel — manage the manifest + character files from a GUI.
//
// Registers a custom sidebar tab (app.extensionManager.registerSidebarTab) with
// three sections backed by the pack's /anim_coord/* routes:
//   • Manifest  — load / edit / validate-and-save a manifest JSON
//   • Characters — create a character, edit its character.json prompt layer
//   • Coverage  — a live status grid over every (entity, direction) for a character
//
// All writes go through JSON-only routes that take no client filesystem path, so
// the panel sends data, never paths. Vanilla DOM, no build step.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAG = "[andypack-panel]";
const NO_CHARACTER = "(select character)";
const GLYPH = { generated: "✅", ready: "🔵", stale: "🟠", blocked: "🔴" };
const STATUS_ORDER = ["blocked", "stale", "ready", "generated"];

// --- tiny DOM helper -------------------------------------------------------- //
function h(tag, props = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") Object.assign(el.style, v);
    else if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el[k.toLowerCase()] = v;
    else if (v != null) el.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return el;
}

const enc = encodeURIComponent;

async function apiGet(url) {
  try {
    const res = await api.fetchApi(url);
    if (!res.ok) return { __error: `${res.status}` };
    return await res.json();
  } catch (e) {
    console.warn(TAG, "GET failed", url, e);
    return { __error: String(e) };
  }
}

async function apiPost(url, body) {
  try {
    const res = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, ...data };
  } catch (e) {
    console.warn(TAG, "POST failed", url, e);
    return { ok: false, error: String(e) };
  }
}

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 4000 });
  } catch {
    console.log(TAG, severity, summary, detail || "");
  }
}

// --- shared styling --------------------------------------------------------- //
const PAD = "8px";
const btnStyle = {
  cursor: "pointer", padding: "5px 10px", border: "1px solid var(--border-color, #444)",
  borderRadius: "4px", background: "var(--comfy-input-bg, #222)",
  color: "var(--input-text, #ddd)", fontSize: "12px",
};
const fieldStyle = {
  width: "100%", boxSizing: "border-box", background: "var(--comfy-input-bg, #1a1a1a)",
  color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #444)",
  borderRadius: "4px", padding: "5px", fontSize: "12px", fontFamily: "monospace",
};
const labelStyle = { fontSize: "11px", opacity: "0.8", margin: "6px 0 2px" };

function button(label, onclick, extra = {}) {
  return h("button", { style: { ...btnStyle, ...extra }, onClick: onclick }, label);
}

// --- Manifest section ------------------------------------------------------- //
function manifestSection() {
  const root = h("div", { style: { padding: PAD } });
  const picker = h("select", { style: fieldStyle });
  const editor = h("textarea", {
    style: { ...fieldStyle, height: "320px", whiteSpace: "pre", resize: "vertical" },
    spellcheck: "false",
  });
  const statusLine = h("div", { style: { fontSize: "11px", margin: "6px 0", minHeight: "16px" } });

  async function loadList(selected) {
    const data = await apiGet("/anim_coord/manifests");
    const names = (data && data.manifests) || [];
    picker.innerHTML = "";
    for (const n of names) picker.appendChild(h("option", { value: n }, n));
    if (selected && names.includes(selected)) picker.value = selected;
    if (names.length) await loadOne();
    else statusLine.textContent = "no manifests found";
  }

  async function loadOne() {
    const name = picker.value;
    if (!name) return;
    const data = await apiGet(`/anim_coord/manifest?name=${enc(name)}`);
    if (data && data.text != null) {
      editor.value = data.text;
      statusLine.textContent = `loaded ${name}`;
      statusLine.style.color = "var(--fg-color, #ccc)";
    } else {
      statusLine.textContent = `failed to load ${name}`;
      statusLine.style.color = "#e66";
    }
  }

  async function save() {
    const name = picker.value;
    if (!name) return;
    const res = await apiPost("/anim_coord/manifest/save", { name, content: editor.value });
    if (res.ok) {
      const warns = res.warnings || [];
      statusLine.style.color = warns.length ? "#e9a" : "#7c7";
      statusLine.textContent = warns.length
        ? `saved with ${warns.length} lint warning(s)` : "saved ✓";
      toast("success", `Saved ${name}`, warns.length ? warns.join("\n") : "Valid manifest");
    } else {
      statusLine.style.color = "#e66";
      statusLine.textContent = res.error || "save failed";
      toast("error", "Save failed", res.error || "");
    }
  }

  picker.onchange = loadOne;
  root.append(
    h("div", { style: labelStyle }, "Manifest"),
    picker,
    h("div", { style: { display: "flex", gap: "6px", margin: "6px 0" } }, [
      button("Reload", () => loadList(picker.value)),
      button("Validate + Save", save, { marginLeft: "auto", fontWeight: "600" }),
    ]),
    editor,
    statusLine,
    h("div", { style: { fontSize: "10px", opacity: "0.6", marginTop: "4px" } },
      "Edits are validated before they touch disk — a broken edit is rejected, not saved."),
  );
  loadList();
  return root;
}

// --- Characters section ----------------------------------------------------- //
function charactersSection() {
  const root = h("div", { style: { padding: PAD } });
  const list = h("div", { style: { margin: "4px 0" } });
  const newName = h("input", { style: fieldStyle, placeholder: "new character name" });
  const editorWrap = h("div", { style: { marginTop: "10px", display: "none" } });
  const editTitle = h("div", { style: { fontWeight: "600", fontSize: "12px", margin: "4px 0" } });
  const pos = h("textarea", { style: { ...fieldStyle, height: "90px", resize: "vertical" } });
  const neg = h("textarea", { style: { ...fieldStyle, height: "60px", resize: "vertical" } });
  let current = null;

  async function refresh() {
    const chars = await apiGet("/anim_coord/characters");
    list.innerHTML = "";
    const arr = Array.isArray(chars) ? chars : [];
    if (!arr.length) list.appendChild(h("div", { style: { opacity: "0.6", fontSize: "11px" } }, "no characters yet"));
    for (const c of arr) {
      list.appendChild(button(c.name, () => edit(c.name),
        { display: "block", width: "100%", textAlign: "left", margin: "2px 0" }));
    }
  }

  async function edit(name) {
    const data = await apiGet(`/anim_coord/character?character=${enc(name)}`);
    const layer = (data && data.layer) || {};
    current = name;
    editTitle.textContent = `Editing: ${name}`;
    pos.value = layer.positive_prompt || "";
    neg.value = layer.negative_prompt || "";
    editorWrap.style.display = "block";
  }

  async function create() {
    const name = newName.value.trim();
    if (!name) return;
    const res = await apiPost("/anim_coord/character/create", { character: name });
    if (res.ok) {
      newName.value = "";
      toast("success", "Character ready", res.name);
      await refresh();
      edit(res.name);
    } else {
      toast("error", "Create failed", res.error || "");
    }
  }

  async function save() {
    if (!current) return;
    const res = await apiPost("/anim_coord/character/save", {
      character: current, positive_prompt: pos.value, negative_prompt: neg.value,
    });
    if (res.ok) toast("success", `Saved ${res.name}`, "character.json written");
    else toast("error", "Save failed", res.error || "");
  }

  editorWrap.append(
    editTitle,
    h("div", { style: labelStyle }, "Character positive ({character_prompt} in a positive field)"),
    pos,
    h("div", { style: labelStyle }, "Character negative ({character_prompt} in a negative field)"),
    neg,
    h("div", { style: { marginTop: "6px" } }, button("Save character", save, { fontWeight: "600" })),
    h("div", { style: { fontSize: "10px", opacity: "0.6", marginTop: "4px" } },
      "The identity layer. FLUX.2 Klein ignores negatives — the negative is for the Wan path."),
  );

  root.append(
    h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, [
      newName, button("Create", create),
    ]),
    h("div", { style: labelStyle }, "Characters"),
    list,
    h("div", { style: { textAlign: "right" } }, button("Refresh", refresh, { fontSize: "11px" })),
    editorWrap,
  );
  refresh();
  return root;
}

// --- Thumbnail cache (coverage grid) --------------------------------------- //
// Keyed by "character|kind|id|direction". Value is a data-uri string on hit,
// null when the route returned 404/error. Cleared on execution events so
// newly-rendered cells pick up fresh images on the next grid refresh.
const _thumbCache = new Map();

async function fetchThumb(character, kind, id, direction) {
  const key = `${character}|${kind}|${id}|${direction}`;
  if (_thumbCache.has(key)) return _thumbCache.get(key);
  const url = `/anim_coord/thumb?character=${enc(character)}&kind=${enc(kind)}&id=${enc(id)}&direction=${enc(direction)}`;
  try {
    const res = await api.fetchApi(url);
    if (!res.ok) { _thumbCache.set(key, null); return null; }
    const data = await res.json();
    const uri = data.data_uri || null;
    _thumbCache.set(key, uri);
    return uri;
  } catch (e) {
    console.warn(TAG, "thumb fetch failed", url, e);
    _thumbCache.set(key, null);
    return null;
  }
}

// --- Coverage section (live status dashboard) ------------------------------- //
function coverageSection() {
  const root = h("div", { style: { padding: PAD } });
  const charPick = h("select", { style: fieldStyle });
  const manifestPick = h("select", { style: fieldStyle });
  const summary = h("div", { style: { fontSize: "12px", margin: "8px 0", fontWeight: "600" } });
  const grid = h("div", {});

  async function loadPickers() {
    const [chars, mans] = await Promise.all([
      apiGet("/anim_coord/characters"), apiGet("/anim_coord/manifests"),
    ]);
    charPick.innerHTML = "";
    charPick.appendChild(h("option", { value: "" }, NO_CHARACTER));
    for (const c of (Array.isArray(chars) ? chars : [])) {
      charPick.appendChild(h("option", { value: c.name }, c.name));
    }
    manifestPick.innerHTML = "";
    for (const n of ((mans && mans.manifests) || [])) {
      manifestPick.appendChild(h("option", { value: n }, n));
    }
  }

  async function refresh() {
    const character = charPick.value;
    const manifest = manifestPick.value || "default.json";
    grid.innerHTML = "";
    if (!character) { summary.textContent = "select a character"; return; }
    const opts = await apiGet(
      `/anim_coord/options?manifest=${enc(manifest)}&character=${enc(character)}`);
    if (!Array.isArray(opts)) { summary.textContent = "failed to load"; return; }
    const counts = { generated: 0, ready: 0, stale: 0, blocked: 0 };
    for (const o of opts) counts[o.status] = (counts[o.status] || 0) + 1;
    summary.textContent = STATUS_ORDER
      .map((s) => `${GLYPH[s]} ${s} ${counts[s] || 0}`).join("   ");

    // Group by kind -> category -> id, render a row of direction glyphs per id.
    const byKey = {};
    for (const o of opts) {
      const k = `${o.kind} ${o.category || "(none)"} ${o.id}`;
      (byKey[k] ||= []).push(o);
    }
    let lastHead = "";
    for (const k of Object.keys(byKey).sort()) {
      const [kind, category, id] = k.split(" ");
      const head = `${kind} · ${category}`;
      if (head !== lastHead) {
        grid.appendChild(h("div", {
          style: { fontSize: "11px", opacity: "0.7", margin: "8px 0 2px", borderBottom: "1px solid var(--border-color,#333)" },
        }, head));
        lastHead = head;
      }
      const row = h("div", { style: { display: "flex", alignItems: "center", gap: "4px", margin: "1px 0" } });
      row.appendChild(h("span", { style: { fontSize: "11px", width: "150px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: id }, id));
      const cells = byKey[k].slice().sort((a, b) => a.direction.localeCompare(b.direction));
      for (const o of cells) {
        const blocked = (o.blocked_by || []).join(", ");
        const title = `${o.id} @ ${o.direction} — ${o.status}${blocked ? ` (needs ${blocked})` : ""}`;
        const glyphEl = h("span", {}, GLYPH[o.status] || "·");
        const imgEl = h("img", {
          loading: "lazy",
          style: {
            width: "24px", height: "24px", objectFit: "cover",
            display: "none", borderRadius: "2px", verticalAlign: "middle",
          },
        });
        const cellEl = h("span", {
          style: { fontSize: "12px", cursor: "default", display: "inline-block" },
          title,
        }, [glyphEl, imgEl]);
        if (o.status !== "blocked") {
          fetchThumb(character, o.kind, o.id, o.direction).then((uri) => {
            if (uri) {
              imgEl.src = uri;
              imgEl.style.display = "inline-block";
              glyphEl.style.display = "none";
            }
          }).catch(() => {});
        }
        row.appendChild(cellEl);
      }
      grid.appendChild(row);
    }
  }

  charPick.onchange = refresh;
  manifestPick.onchange = refresh;
  root.append(
    h("div", { style: labelStyle }, "Character"), charPick,
    h("div", { style: labelStyle }, "Manifest"), manifestPick,
    h("div", { style: { textAlign: "right", margin: "6px 0" } },
      button("Refresh", refresh, { fontSize: "11px" })),
    summary, grid,
  );
  loadPickers().then(refresh);
  // Live refresh: re-pull status after any graph run (a writer just unlocked work).
  const onExec = () => { _thumbCache.clear(); if (charPick.value) refresh(); };
  api.addEventListener("execution_success", onExec);
  api.addEventListener("executed", onExec);
  root.__cleanup = () => {
    api.removeEventListener("execution_success", onExec);
    api.removeEventListener("executed", onExec);
  };
  return root;
}

// --- panel shell with a small tab bar --------------------------------------- //
function buildPanel(el) {
  el.innerHTML = "";
  const root = h("div", { style: { color: "var(--fg-color, #ccc)", height: "100%", overflow: "auto" } });
  const tabBar = h("div", {
    style: { display: "flex", borderBottom: "1px solid var(--border-color, #333)", position: "sticky", top: "0", background: "var(--comfy-menu-bg, #202020)", zIndex: "1" },
  });
  const content = h("div", {});
  const tabs = {
    Manifest: manifestSection,
    Characters: charactersSection,
    Coverage: coverageSection,
  };
  let activeCleanup = null;
  function show(name) {
    if (activeCleanup) { activeCleanup(); activeCleanup = null; }
    content.innerHTML = "";
    const node = tabs[name]();
    activeCleanup = node.__cleanup || null;
    content.appendChild(node);
    for (const b of tabBar.children) {
      b.style.borderBottom = b.textContent === name
        ? "2px solid var(--p-primary-color, #4af)" : "2px solid transparent";
      b.style.opacity = b.textContent === name ? "1" : "0.65";
    }
  }
  for (const name of Object.keys(tabs)) {
    tabBar.appendChild(h("button", {
      style: { ...btnStyle, border: "none", borderRadius: "0", background: "transparent", flex: "1", padding: "8px 4px" },
      onClick: () => show(name),
    }, name));
  }
  root.append(tabBar, content);
  el.appendChild(root);
  show("Manifest");
  return () => { if (activeCleanup) activeCleanup(); };
}

app.registerExtension({
  name: "andypack.manifestPanel",
  async setup() {
    if (!app.extensionManager || !app.extensionManager.registerSidebarTab) {
      console.warn(TAG, "registerSidebarTab unavailable — sidebar panel disabled");
      return;
    }
    app.extensionManager.registerSidebarTab({
      id: "andypackManifest",
      icon: "pi pi-images",
      title: "Andypack",
      tooltip: "Manage the animation manifest and character files",
      type: "custom",
      render: (el) => buildPanel(el),
    });
    console.debug(TAG, "sidebar tab registered");
  },
});
