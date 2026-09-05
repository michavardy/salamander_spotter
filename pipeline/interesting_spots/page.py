"""The single-page UI, served as a string from ``server.py``.

Vanilla JS, no build step, no external assets — the whole app is this one page plus the
JSON endpoints in ``server.py``. The photo and each selected spot's green overlay are
stacked <img>s inside a size-to-content stage, so overlays line up with the photo at any
zoom without letterbox maths.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interesting Spot Selector</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #14171c; color: #e6e9ef; display: flex; flex-direction: column;
    min-height: 100vh;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 10px 16px; background: #1c2027; border-bottom: 1px solid #2a2f38;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  .sid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #8fd39a; }
  .muted { color: #9aa3b2; }
  .grow { flex: 1; }
  .pill {
    background: #232833; border: 1px solid #2f3540; border-radius: 999px;
    padding: 3px 10px; font-size: 12px; white-space: nowrap;
  }
  main { flex: 1; display: flex; align-items: center; justify-content: center; padding: 16px; }
  .stage {
    position: relative; display: inline-block; line-height: 0;
    box-shadow: 0 8px 40px rgba(0,0,0,.5); border-radius: 6px; overflow: hidden;
    background: #0c0e12;
  }
  #photo { display: block; max-height: 78vh; max-width: 92vw; cursor: crosshair; }
  .overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .legend { display: inline-flex; align-items: center; gap: 5px; color: #9aa3b2; font-size: 12px; }
  .sw { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .sw-green { background: rgba(0,210,0,.95); }
  footer {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 10px 16px; background: #1c2027; border-top: 1px solid #2a2f38;
  }
  button {
    font: inherit; color: #e6e9ef; background: #2a3038; border: 1px solid #39414d;
    border-radius: 7px; padding: 7px 14px; cursor: pointer;
  }
  button:hover { background: #333b45; }
  button:disabled { opacity: .4; cursor: default; }
  button.primary { background: #24603a; border-color: #2f7a4a; }
  button.primary:hover { background: #2b7044; }
  kbd {
    font-family: ui-monospace, monospace; font-size: 11px; background: #12151a;
    border: 1px solid #333; border-bottom-width: 2px; border-radius: 4px; padding: 0 5px;
  }
  .sel-list { font-family: ui-monospace, monospace; color: #8fd39a; }
  .toast {
    position: fixed; bottom: 74px; left: 50%; transform: translateX(-50%);
    background: #232833; border: 1px solid #39414d; border-radius: 8px;
    padding: 8px 14px; opacity: 0; transition: opacity .15s; pointer-events: none;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Interesting Spot Selector</h1>
  <span class="pill">folder <span class="sid" id="folder">…</span></span>
  <span class="grow"></span>
  <span class="pill" id="progress">…</span>
  <span class="pill" id="labeledPill">…</span>
</header>

<main>
  <div class="stage" id="stage">
    <img id="photo" alt="salamander (purple/segmentation image)" draggable="false" />
  </div>
</main>

<footer>
  <button id="prevBtn">◀ Prev <kbd>←</kbd></button>
  <button id="nextBtn">Next <kbd>→</kbd> ▶</button>
  <button id="unlabeledBtn" class="primary">Next unlabeled <kbd>U</kbd></button>
  <span class="grow"></span>
  <span class="legend">click a magenta spot to mark it <i class="sw sw-green"></i>interesting</span>
  <span class="muted">· <b id="sidLabel" class="sid">…</b> ·</span>
  <span class="sel-list" id="selList">—</span>
</footer>

<div class="toast" id="toast"></div>

<script>
const state = { images: [], index: 0, meta: null, overlays: new Map(), toastTimer: null };
const $ = (id) => document.getElementById(id);
const photo = $("photo"), stage = $("stage");

async function jget(url) { const r = await fetch(url); return r.json(); }
async function jpost(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  return r.json();
}

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => t.classList.remove("show"), 1200);
}

function clearOverlays() {
  for (const el of state.overlays.values()) el.remove();
  state.overlays.clear();
}
function addOverlay(spotId) {
  if (state.overlays.has(spotId)) return;
  const img = document.createElement("img");
  img.className = "overlay";
  img.src = `/overlay?sid=${encodeURIComponent(state.meta.sid)}&spot=${spotId}`;
  stage.appendChild(img);
  state.overlays.set(spotId, img);
}
function removeOverlay(spotId) {
  const el = state.overlays.get(spotId);
  if (el) { el.remove(); state.overlays.delete(spotId); }
}

function renderSelection(sel) {
  $("selList").textContent = sel.length ? sel.join(", ") : "—";
}
function renderHeader() {
  const m = state.meta;
  $("progress").textContent = `${m.index + 1} / ${m.total}`;
  $("labeledPill").textContent = `${m.labeledCount} labeled`;
  $("sidLabel").textContent = m.sid;
  $("prevBtn").disabled = m.index <= 0;
  $("nextBtn").disabled = m.index >= m.total - 1;
}

async function loadIndex(i) {
  const m = await jget(`/api/image?i=${i}`);
  state.index = m.index; state.meta = m;
  clearOverlays();
  photo.src = `/photo?sid=${encodeURIComponent(m.sid)}`;
  renderHeader();
  renderSelection(m.selected);
  for (const id of m.selected) addOverlay(id);
}

photo.addEventListener("click", async (e) => {
  if (!state.meta || !photo.naturalWidth) return;
  const rect = photo.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / rect.width * photo.naturalWidth);
  const y = Math.floor((e.clientY - rect.top) / rect.height * photo.naturalHeight);
  const res = await jpost("/api/click", { sid: state.meta.sid, x, y });
  if (res.spotId === null || res.spotId === undefined) { toast("no spot here"); return; }
  if (res.selected) addOverlay(res.spotId); else removeOverlay(res.spotId);
  state.meta.selected = res.selection;
  state.meta.labeledCount = res.labeledCount;
  renderSelection(res.selection);
  $("labeledPill").textContent = `${res.labeledCount} labeled`;
  // reflect labeled state for "next unlabeled" bookkeeping
  const img = state.images[state.index]; if (img) img.labeled = res.selection.length > 0;
  toast(`spot ${res.spotId} ${res.selected ? "selected" : "unselected"}`);
});

function go(delta) {
  const i = state.index + delta;
  if (i >= 0 && i < state.meta.total) loadIndex(i);
}
async function goUnlabeled() {
  const res = await jget(`/api/next-unlabeled?i=${state.index}`);
  if (res.index === null || res.index === undefined) { toast("all images labeled 🎉"); return; }
  loadIndex(res.index);
}

$("prevBtn").addEventListener("click", () => go(-1));
$("nextBtn").addEventListener("click", () => go(1));
$("unlabeledBtn").addEventListener("click", goUnlabeled);

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { go(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { go(1); e.preventDefault(); }
  else if (e.key === "u" || e.key === "U") { goUnlabeled(); e.preventDefault(); }
});

(async function boot() {
  const idx = await jget("/api/index");
  state.images = idx.images;
  $("folder").textContent = idx.folder;
  if (!idx.total) { document.querySelector("main").innerHTML =
      '<p class="muted">No images with spots found in this folder.</p>'; return; }
  await loadIndex(idx.startIndex);
})();
</script>
</body>
</html>
"""
