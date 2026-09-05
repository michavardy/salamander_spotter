"""Single-page UI for the correspondence labeller. No framework, no CDN — one string."""

INDEX_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>spot correspondence</title>
<style>
  :root { --bg:#12131a; --fg:#e8e8ee; --dim:#8b8fa3; --acc:#59c36a; --warn:#e0574a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui,sans-serif; }
  header { display:flex; gap:16px; align-items:center; padding:8px 14px;
           border-bottom:1px solid #262838; position:sticky; top:0; background:var(--bg); z-index:5; }
  header b { font-size:15px; }
  .sp { flex:1; }
  .pill { background:#1d1f2b; border:1px solid #2c2f40; border-radius:99px; padding:2px 10px;
          color:var(--dim); font-size:12px; }
  .pill.on { color:var(--acc); border-color:var(--acc); }
  .pill.miss { color:var(--warn); border-color:var(--warn); }
  button { background:#1d1f2b; color:var(--fg); border:1px solid #343850; border-radius:6px;
           padding:5px 11px; cursor:pointer; font-size:13px; }
  button:hover { border-color:#5a6088; }
  button.primary { background:#20452a; border-color:var(--acc); }
  #wrap { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:10px; }
  .pane { position:relative; }
  .pane h3 { margin:0 0 6px; font-size:13px; color:var(--dim); font-weight:500; }
  .stage { position:relative; line-height:0; border:1px solid #262838; border-radius:6px;
           overflow:hidden; }
  .stage img.base { width:100%; height:auto; display:block; cursor:crosshair; }
  .stage img.ov { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .tag { position:absolute; transform:translate(-50%,-50%); font:700 12px/1 system-ui;
         color:#000; background:#fff; border-radius:99px; padding:2px 6px; pointer-events:none;
         box-shadow:0 0 0 1.5px #0009; }
  footer { padding:6px 14px; color:var(--dim); font-size:12px; border-top:1px solid #262838; }
  kbd { background:#1d1f2b; border:1px solid #343850; border-radius:4px; padding:1px 5px;
        font:11px monospace; }
</style>
<header>
  <b id="lbl">—</b>
  <span class="pill" id="pos"></span>
  <span class="pill" id="links"></span>
  <span class="pill" id="prog"></span>
  <span class="pill" id="mode">LINK mode</span>
  <span class="sp"></span>
  <button onclick="undo()">Undo <kbd>U</kbd></button>
  <button onclick="toggleMode()">Miss mode <kbd>M</kbd></button>
  <button class="primary" onclick="markDone()">Done + next <kbd>Enter</kbd></button>
  <button onclick="go(idx-1)">←</button>
  <button onclick="go(idx+1)">→</button>
</header>
<div id="wrap">
  <div class="pane"><h3 id="ha"></h3><div class="stage" id="sa"></div></div>
  <div class="pane"><h3 id="hb"></h3><div class="stage" id="sb"></div></div>
</div>
<footer>
  Click a spot on the <b>left</b>, then its partner on the <b>right</b> — same colour = same
  physical spot. <kbd>M</kbd> then clicking left marks a spot <span style="color:var(--warn)">
  visible in the right photo but never extracted there</span> (that is the extraction-recall
  signal). <kbd>U</kbd> undo · <kbd>Enter</kbd> done+next · <kbd>←</kbd>/<kbd>→</kbd> navigate.
  Everything saves on every click.
</footer>
<script>
let idx = 0, meta = null, pending = null, missMode = false;

async function load(i) {
  const r = await fetch('/api/pair?i=' + i);
  if (!r.ok) return;
  meta = await r.json(); idx = meta.index; pending = null;
  document.getElementById('lbl').textContent = meta.label;
  document.getElementById('pos').textContent = (idx + 1) + ' / ' + meta.total;
  document.getElementById('prog').textContent = meta.doneCount + ' pairs done · ' +
      meta.linkCount + ' links total';
  document.getElementById('ha').textContent = meta.a.sid + '  ·  ' + meta.a.nSpots + ' spots';
  document.getElementById('hb').textContent = meta.b.sid + '  ·  ' + meta.b.nSpots + ' spots';
  render();
}

function render() {
  document.getElementById('links').textContent = meta.links.length + ' links · ' +
      meta.misses.length + ' missing';
  draw('sa', meta.a, 'a'); draw('sb', meta.b, 'b');
}

function draw(elId, side, which) {
  const st = document.getElementById(elId);
  st.innerHTML = '';
  const img = new Image();
  img.className = 'base';
  img.src = '/photo?sid=' + encodeURIComponent(side.sid);
  img.onclick = (e) => click(e, side, which);
  st.appendChild(img);
  meta.links.forEach((l, i) => {
    const spot = which === 'a' ? l[0] : l[1];
    add(st, side.sid, spot, 'link', i, String(i + 1));
  });
  if (which === 'a') meta.misses.forEach(m => add(st, side.sid, m, 'miss', 0, '×'));
  if (pending && which === 'a') add(st, side.sid, pending, 'pending', 0, '?');
}

function add(st, sid, spot, kind, i, tag) {
  const o = new Image();
  o.className = 'ov';
  o.src = '/overlay?sid=' + encodeURIComponent(sid) + '&spot=' + spot +
          '&kind=' + kind + '&idx=' + i;
  st.appendChild(o);
}

async function click(e, side, which) {
  const r = e.target.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) / r.width * side.width);
  const y = Math.round((e.clientY - r.top) / r.height * side.height);
  const hit = await post('/api/spot-at', {sid: side.sid, x, y});
  if (hit.spotId === null) return;
  if (which === 'a') {
    if (missMode) { meta = Object.assign(meta, await post('/api/miss',
        {key: meta.key, spot: hit.spotId})); pending = null; }
    else pending = hit.spotId;
  } else {
    if (pending === null) return;
    meta = Object.assign(meta, await post('/api/link',
        {key: meta.key, a: pending, b: hit.spotId}));
    pending = null;
  }
  render();
}

async function post(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify(body)});
  return r.json();
}
async function undo() { meta = Object.assign(meta, await post('/api/undo', {key: meta.key}));
                        pending = null; render(); }
async function markDone() {
  await post('/api/done', {key: meta.key, done: true});
  const r = await fetch('/api/next-undone?i=' + idx);
  const j = await r.json();
  load(j.index === null ? idx : j.index);
}
function toggleMode() {
  missMode = !missMode;
  const el = document.getElementById('mode');
  el.textContent = missMode ? 'MISS mode' : 'LINK mode';
  el.className = 'pill ' + (missMode ? 'miss' : '');
}
function go(i) { if (meta && i >= 0 && i < meta.total) load(i); }

addEventListener('keydown', ev => {
  if (ev.key === 'ArrowLeft') go(idx - 1);
  else if (ev.key === 'ArrowRight') go(idx + 1);
  else if (ev.key === 'u' || ev.key === 'U') undo();
  else if (ev.key === 'm' || ev.key === 'M') toggleMode();
  else if (ev.key === 'Enter') markDone();
  else if (ev.key === 'Escape') { pending = null; render(); }
});

fetch('/api/index').then(r => r.json()).then(j => load(j.startIndex));
</script>
"""
