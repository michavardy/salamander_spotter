"""The single page — markup, CSS and JS as one string.

All state and every interaction live here: click -> ``spot_id`` is an SVG hit test (the spot
outlines are ``<path>`` elements, so the browser does the point-in-polygon), overlays are SVG
layers over an ``<img>``, and the whole review record is mutated in memory and posted back as a
patch ~300 ms later. The server only hands over data and writes the file.

Layout, colours, glyphs and every flow this implements are specified in docs/preprocessing_ui.md.
"""
from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>preprocessing review</title>
<style>
/* ---------- palette (docs/preprocessing_ui.md#palette) ---------- */
:root{
  --bg:#161616; --panel:#1F1F21; --chip:#333333; --card:#3A3A3C; --line:#2E2E30;
  --ink:#EDEDED; --ink2:#A8A8A8; --ink3:#767676;
  /* --human / --machine are the SAME hexes the arc table uses, so the table, the pending-spot
     highlight, the miss stub and the arcs are one blue and one red, not two of each. */
  --identity:#FFFF00; --human:#3987e5; --machine:#e66767; --machine-ink:#BA4040;
  --reject:#800000; --accept:#008000; --teal:#008080; --anatomy:#FFD24A;
  --purple:#8A2BE2; --del:#5A5A5A; --magenta:#FF00FF;
  --prev:#893332; --next:#367950; --steel:#6495AD;
  --i1:#CFAC6B; --i2:#BE9450; --i3:#A87A33; --i4:#8B5C1B; --i5:#6E480F;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);
  font:13px/1.45 Arial,Helvetica,"Segoe UI",sans-serif;overflow:hidden}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}
input,select{font:inherit}
#app{display:grid;height:100vh;
  grid-template-columns:250px 1fr;grid-template-rows:38px 1fr 46px;
  grid-template-areas:"top top" "side canvas" "bot bot"}

/* ---------- top bar ---------- */
#topbar{grid-area:top;display:flex;align-items:center;gap:14px;padding:0 10px;
  border-bottom:1px solid var(--line);background:var(--bg)}
#prog{min-width:210px}
#progBar{height:5px;background:#242426;border-radius:3px;overflow:hidden}
#progFill{height:100%;background:var(--steel);width:0}
#progTxt{font-size:11px;color:var(--ink2)}
.pipe{color:var(--ink3)}
#indLabel{color:var(--identity);font-weight:bold;letter-spacing:.4px}
.topbtn{width:21px;height:21px;border-radius:50%;display:inline-grid;place-items:center;
  line-height:0;color:#fff;opacity:.55;transition:opacity .1s,box-shadow .1s;vertical-align:middle}
.topbtn:hover{opacity:1}
.topbtn.on{opacity:1;box-shadow:0 0 0 2px #ffffff55}
.topbtn.rej{background:#B03030}.topbtn.del{background:var(--del)}
#jobs{margin-left:auto;display:none;gap:6px;align-items:center}
.job{font-size:10.5px;border-radius:10px;padding:1px 9px;cursor:pointer;border:1px solid var(--line);
  color:var(--ink2);white-space:nowrap}
.job.running{border-color:var(--identity);color:var(--identity);animation:pulse 1.6s infinite}
.job.ready{border-color:#19C219;color:#8FE08F}
.job.failed{border-color:#E23A3A;color:#E08A8A}
@keyframes pulse{50%{opacity:.55}}
#help{margin-left:10px;color:var(--ink3)}
.fbtn.gen{background:#2E6FB8}
.card.prov{border-color:#C9A227;border-style:dashed}
.badge.prov{background:#5A4A0E;color:#FFE79A}
#banner{position:fixed;top:44px;left:50%;transform:translateX(-50%);z-index:60;max-width:70vw;
  background:#3A1414;border:1px solid #7A2A2A;padding:8px 12px;border-radius:6px;display:none}
#banner button{text-decoration:underline;margin-left:10px;color:#FFD9D9}

/* ---------- sidebar ---------- */
#sidebar{grid-area:side;overflow-y:auto;border-right:1px solid var(--line);padding:8px 10px 20px}
#sidebar h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--ink2);
  margin:14px 0 6px;display:flex;align-items:center;gap:6px}
#sidebar h2:first-child{margin-top:2px}
#topN{width:44px;background:#fff;color:#111;border:0;padding:1px 4px;text-align:center}
table.mt{width:100%;border-collapse:separate;border-spacing:0;font-size:11px;
  font-variant-numeric:tabular-nums}
table.mt th{color:var(--ink3);font-weight:normal;text-align:left;padding:3px 3px;cursor:pointer;
  border-bottom:1px solid #34343A;position:sticky;top:0;background:var(--bg);font-size:10px;
  text-transform:uppercase;letter-spacing:.4px}
table.mt th:nth-child(3),table.mt th:nth-child(4){text-align:center}
table.mt td{padding:3px;white-space:nowrap;border-bottom:1px solid #232326}
td.ss{font-family:"Cascadia Mono",Consolas,monospace;font-size:10.5px;letter-spacing:-.2px}
td.sc{text-align:center;color:var(--ink)}
tr.mrow{cursor:default}
tr.mrow.rejected td.sc{text-decoration:line-through;color:var(--ink3)}
tr.mrow.rejected td.ss{opacity:.6}
tr.mrow:hover{background:#26262B}
tr.mrow.sel{background:#2F2F36}
.ipill{display:block;width:20px;height:16px;border-radius:3px;color:#fff;font-size:11px;
  line-height:16px;text-align:center;margin:0 auto}
.ipill.t{background:var(--accept)}.ipill.f{background:#B03030}
.undo{color:#5E5E62;font-size:11px;padding:0 2px}
.undo:hover{color:#E08A8A}
#mtMore{display:none;width:100%;font-size:10.5px;color:var(--ink3);padding:3px 0;
  border-bottom:1px solid #232326}
#mtMore:hover{color:var(--ink)}
#addRow{display:flex;gap:3px;margin-top:5px}
#addRow input{flex:1;min-width:0;background:#242426;color:var(--ink);border:1px solid var(--line);
  font-size:10px;padding:2px 3px;font-family:"Cascadia Mono",Consolas,monospace}
#addRow button{background:var(--chip);border-radius:3px;font-size:10.5px;padding:2px 6px;
  color:var(--ink2)}
#addRow button:hover{color:#fff}
#arcLegend{margin-top:7px;display:flex;flex-direction:column;gap:2px}
.lgd{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--ink3)}
.msrow{font-size:11px;color:var(--ink2);display:flex;gap:6px;align-items:center}
.msrow button{color:var(--ink3)}
.opt{display:flex;align-items:center;justify-content:space-between;gap:6px;margin:3px 0;
  font-size:11.5px;color:var(--ink2)}
.opt select,.opt input[type=number]{background:#242426;color:var(--ink);border:1px solid var(--line);
  padding:1px 3px}
.opt input[type=range]{width:96px}
.chips{display:flex;flex-direction:column;gap:4px;align-items:flex-start}
.chip{background:var(--chip);border-radius:11px;padding:2px 10px;font-size:11.5px;color:var(--ink)}
.chip.add{color:var(--ink2)}
table.qt{width:100%;border-collapse:collapse;font-size:11px}
table.qt th{color:var(--ink2);font-weight:normal;padding:2px;cursor:pointer;text-align:right}
table.qt th:first-child{text-align:left}
table.qt td{padding:2px;text-align:right;position:relative}
table.qt td:first-child{text-align:left;color:var(--ink2)}
table.qt tr.sel td:first-child{color:var(--identity)}
table.qt tr.meanrow td{color:var(--ink3);border-top:1px solid var(--line);font-style:italic}
.qcell{position:relative;z-index:1}
.qbg{position:absolute;inset:1px;z-index:0;border-radius:2px}
#ramp{display:flex;gap:0;margin-top:6px;align-items:center;font-size:10px;color:var(--ink3)}
#ramp i{width:20px;height:9px;display:inline-block}
#interestState{font-size:10.5px;color:var(--ink3);margin-top:4px}

/* ---------- canvas ---------- */
#canvas{grid-area:canvas;overflow:auto;position:relative;padding:8px}
#cards{display:flex;gap:14px;align-items:flex-start;position:relative;width:max-content;min-width:100%}
#arcs{position:absolute;left:0;top:0;pointer-events:none;overflow:visible;z-index:5}
#arcs path.arc{fill:none;pointer-events:stroke;cursor:pointer}
#arcs path.hit{fill:none;stroke:transparent;stroke-width:14px;pointer-events:stroke;cursor:pointer}
.card{background:var(--card);border-radius:4px;width:290px;flex:0 0 auto;
  border:2px solid var(--line);transition:border-color .12s,box-shadow .12s}
/* The border carries the decision — a reviewer sees the state of the whole family at a glance. */
.card.acc{border-color:#19C219;box-shadow:0 0 0 1px #19C21955}
.card.rej{border-color:#E23A3A;box-shadow:0 0 0 1px #E23A3A55}
.card.notrain{border-color:#6A6A6E;border-style:dashed}
.card.active{outline:1px solid var(--identity);outline-offset:1px}
.card.deleted{opacity:.42;border-style:dashed}
.card.pop{animation:pop .32s ease-out}
@keyframes pop{0%{transform:scale(.985)}45%{transform:scale(1.012)}100%{transform:scale(1)}}
.badge.train{background:#0E5D5D;color:#9EE7E7}
.cardTitle{text-align:center;color:var(--identity);padding:4px 6px 2px;font-size:13px;
  display:flex;justify-content:center;gap:6px;align-items:center}
.badge{font-size:9px;background:#4A4A4E;color:var(--ink2);border-radius:8px;padding:0 5px;
  letter-spacing:.4px;text-transform:uppercase}
.photoWrap{position:relative;background:#111;line-height:0}
.photoWrap img{width:100%;display:block}
svg.ov{position:absolute;inset:0;width:100%;height:100%}
/* Only the spot outlines take clicks. The centroid dot, its numeral, the interesting-ring and the
   body layers are decoration drawn ON TOP of them — left hittable, they swallow a click aimed at
   exactly the spot centre, which is the most natural place to aim. */
svg.ov circle.cen,svg.ov text.cnum,svg.ov circle.ring,svg.ov path.outline,svg.ov path.spine,
svg.ov circle.head,svg.ov circle.tail,svg.ov line{pointer-events:none}
svg.ov path.spot{cursor:pointer;stroke-width:1.4}
svg.ov path.spot.hov{stroke:#fff;stroke-width:2.6}
/* A selected spot turns PURPLE and stays that way until it is deselected or a line closes. */
svg.ov path.spot.pend{fill:#B026FF!important;fill-opacity:.85!important;stroke:#F0C8FF!important;
  stroke-width:3.4!important}
svg.ov circle.cen{fill:#00C000}
/* NO font-size / stroke here: a CSS declaration would beat the per-image presentation attributes
   the script sets (they scale with the photo's own pixel size). */
svg.ov text.cnum{fill:#00E000;paint-order:stroke;stroke:#000}
svg.ov circle.ring{fill:none;stroke:#00E000;stroke-width:1.6}
svg.ov path.outline{fill:none}
svg.ov path.spine{fill:none;stroke:#00D2FF;opacity:.95}
svg.ov circle.head{fill:#00E000;stroke:#003300;stroke-width:1.5}
svg.ov circle.tail{fill:#FF2A2A;stroke:#3A0000;stroke-width:1.5}
/* A normalized card is a tall thin strip, so it is narrower than a photo card — which also puts
   more animals side by side, the point of comparing in a pose-free frame. */
.photoWrap.norm{background:#0E0E0E;aspect-ratio:200/520}
.photoWrap.norm svg.ov{position:relative}
#cards.normmode .card{width:196px}
.cardFoot{background:var(--panel);border-radius:0 0 4px 4px;padding:5px 6px}
.fr1{display:flex;align-items:center;gap:5px;border-bottom:1px solid #2A2A2C;padding-bottom:5px}
/* Unselected buttons are dim AND desaturated; a selected one is full colour with a ring, so the
   toggle state is legible without reading the icon. */
.fbtn{width:23px;height:23px;border-radius:50%;display:grid;place-items:center;font-size:12px;
  line-height:1;color:#fff;opacity:.38;filter:grayscale(.55);
  transition:opacity .12s,filter .12s,box-shadow .12s,transform .12s}
.fbtn:hover{opacity:.9;filter:grayscale(0)}
.fbtn.on{opacity:1;filter:none;box-shadow:0 0 0 2px #ffffff70;transform:scale(1.08)}
.fbtn.acc{background:var(--accept)}.fbtn.rej{background:#B03030}
.fbtn.tr{background:var(--teal)}.fbtn.del{background:var(--del)}
.fbtn.an{background:var(--anatomy);color:#222}.fbtn.pu{background:var(--purple)}
.stem{flex:1;text-align:center;font-size:11.5px;color:var(--ink)}
.fr2{display:flex;flex-wrap:wrap;gap:4px;padding:5px 0;align-items:center}
.rchip{background:var(--chip);border-radius:10px;padding:1px 8px;font-size:10.5px;color:var(--ink2)}
.rchip.on{background:#6A2222;color:#fff}
.miss{font-size:10.5px;color:var(--ink2);border:1px solid var(--line);border-radius:10px;
  padding:1px 8px;margin-left:auto}
.miss.arm{background:var(--human);color:#fff;border-color:var(--human)}
.fr3{display:grid;grid-template-columns:repeat(5,1fr);gap:3px;font-size:10px;color:var(--ink2);
  cursor:pointer;padding-top:3px}
.fr3 b{display:block;color:var(--ink);font-size:11.5px;font-weight:normal}
.dlt{display:block;font-size:9px}
.dlt.up{color:#6FBF6F}.dlt.dn{color:#D98080}
.q04{color:var(--ink3)}.q04.ok{color:#7BC47B}.q04.no{color:#E08A8A}
.fr4{font-size:9.5px;color:var(--ink3);text-align:right;padding-top:3px}
.qpanel{display:none;background:#191919;border-top:1px solid var(--line);padding:5px 6px;
  max-height:230px;overflow:auto}
.qpanel.open{display:block}
.qpanel table{width:100%;border-collapse:collapse;font-size:10.5px}
.qpanel td{padding:1px 2px;color:var(--ink2)}.qpanel td.v{text-align:right;color:var(--ink)}
.qpanel h4{margin:5px 0 2px;font-size:10px;text-transform:uppercase;color:var(--ink3)}
.qpanel textarea{width:100%;background:#242426;color:var(--ink);border:1px solid var(--line);
  font:inherit;font-size:11px;min-height:38px}

/* ---------- bottom bar ---------- */
#bot{grid-area:bot;display:flex;align-items:center;justify-content:center;gap:18px;
  border-top:1px solid var(--line)}
.nav{padding:6px 16px;border-radius:6px;color:#fff;font-weight:bold;font-size:12px;
  letter-spacing:.5px}
.nav small{display:block;font-weight:normal;font-size:10px;opacity:.8;letter-spacing:0}
.nav.prev{background:var(--prev)}.nav.next{background:var(--next)}
#save{font-size:11px;color:var(--ink2);min-width:190px;text-align:center}
#save.err{color:#E08A8A}#save.busy{color:var(--identity)}

/* ---------- tooltip ---------- */
#tip{position:fixed;z-index:70;display:none;gap:6px;align-items:flex-start;pointer-events:none}
#tipBox{background:var(--panel);padding:5px 8px;font-size:11.5px;white-space:nowrap;
  box-shadow:0 2px 10px #000a}
#tipBox .sp{color:var(--identity)}
#tipBox .hint{color:var(--ink3);font-size:10px;margin-top:2px}
/* Ctrl held: arcs stop intercepting, so the spots underneath are clickable. It has to name the
   paths — #arcs is already pointer-events:none, and a child that sets `stroke` overrides a parent
   `none`, so a rule on the container alone does nothing. */
body.ctrl #arcs path.hit,body.ctrl #arcs path.arc{pointer-events:none}
#tipShape{background:#000;width:64px;height:64px;flex:0 0 auto}
#modal{position:fixed;inset:0;background:#000b;z-index:80;display:none;place-items:center}
#modal.open{display:grid}
#modalBox{background:var(--panel);padding:16px 20px;max-width:560px;border-radius:6px;
  max-height:80vh;overflow:auto}
#modalBox h3{margin:0 0 8px}
#modalBox kbd{background:var(--chip);border-radius:3px;padding:0 4px;font-family:inherit}
#modalBox table{border-collapse:collapse;font-size:12px}
#modalBox td{padding:2px 8px 2px 0;color:var(--ink2)}
</style>
</head>
<body>
<div id="app">
  <header id="topbar">
    <div id="prog"><div id="progTxt">loading…</div><div id="progBar"><div id="progFill"></div></div></div>
    <div>Individual ID: <span id="indLabel">—</span>
      <button class="topbtn rej" id="indReject"
        title="Reject this whole animal — cascades to every photo still unreviewed">
        <svg width="12" height="12" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8"
          stroke="#fff" stroke-width="2.4" stroke-linecap="round" fill="none"/></svg></button>
      <button class="topbtn del" id="indDelete"
        title="This label is the SAME animal as another — merge into it and soft-delete this one">
        <svg width="13" height="13" viewBox="0 0 16 16">
          <circle cx="5.6" cy="8" r="3.1" fill="none" stroke="#fff" stroke-width="1.5"/>
          <circle cx="10.4" cy="8" r="3.1" fill="none" stroke="#fff" stroke-width="1.5"/>
          <path d="M8 5.4v5.2" stroke="#fff" stroke-width="1.3"/></svg></button>
    </div>
    <span class="pipe">|</span><div>Dataset: <span id="dsName" style="color:#8FD08F"></span></div>
    <span class="pipe">|</span><div id="qualHead" style="color:var(--ink2)"></div>
    <div id="jobs"></div>
    <button id="help">&#9432; Help</button>
  </header>

  <aside id="sidebar">
    <h2>Matches <span style="margin-left:auto;color:var(--ink3)">Top</span>
      <input id="topN" type="number" min="0" max="40" value="5"></h2>
    <table class="mt"><thead><tr>
      <th data-sort="a">SSID</th><th data-sort="b">SSID match</th>
      <th data-sort="score">Score</th><th data-sort="accepted">Acc</th><th></th>
    </tr></thead><tbody id="mtBody"></tbody></table>
    <button id="mtMore"></button>
    <div id="mtEmpty" style="color:var(--ink3);font-size:11px"></div>
    <div id="addRow">
      <input id="addA" placeholder="AC-3-1-14" spellcheck="false">
      <input id="addB" placeholder="AC-3-2-15" spellcheck="false">
      <button id="addGo" title="assert this match by typing two SSIDs">+ add</button>
    </div>
    <div id="arcLegend"></div>
    <h2 id="missHead">Misses</h2>
    <div id="missList" style="display:flex;flex-direction:column;gap:3px"></div>

    <h2>Options &#9881;</h2>
    <div id="options"></div>

    <h2>Rejection Reasons</h2>
    <div class="chips" id="reasonChips"></div>

    <h2>Quality</h2>
    <table class="qt"><thead><tr id="qtHead"></tr></thead><tbody id="qtBody"></tbody></table>
    <div id="ramp"></div>
    <div id="interestState"></div>
  </aside>

  <main id="canvas">
    <div id="cards"></div>
    <svg id="arcs"></svg>
  </main>

  <footer id="bot">
    <button class="nav prev" id="prev">&laquo; PREV INDIVIDUAL<small id="prevLbl"></small></button>
    <div id="save">&#9729; ready</div>
    <button class="nav next" id="next">NEXT INDIVIDUAL &raquo;<small id="nextLbl"></small></button>
  </footer>
</div>

<div id="tip"><div id="tipBox"></div><svg id="tipShape" viewBox="0 0 64 64"></svg></div>
<div id="banner"></div>
<div id="modal"><div id="modalBox"></div></div>

<script>
"use strict";
// ---------------------------------------------------------------- constants
const RAMP = ["#CFAC6B", "#BE9450", "#A87A33", "#8B5C1B", "#6E480F"];   // interest, 5 bins
const OUTLINE_HI = "#FFB454";

// Arc provenance. Three hues, validated all-pairs on the dark shell (scripts/validate_palette.js:
// normal-vision ΔE 20.9, CVD 6.5 — the warn band, which the per-matcher DASH covers as secondary
// encoding). A fourth hue could not clear the normal-vision floor against machine red, so `logreg`
// keeps machine red and is dotted — semantically right, since its arcs ARE raw pairs it scored.
const ARC = {
  human:          { color: "#3987e5", dash: "",      width: 3.0, label: "human" },
  raw:            { color: "#e66767", dash: "7 4",   width: 2.0, label: "raw" },
  strict_hand_pos:{ color: "#199e70", dash: "",      width: 2.0, label: "strict_hand_pos (1:1)" },
  logreg:         { color: "#e66767", dash: "2 4",   width: 2.0, label: "logreg (raw pairs)" },
  rejected:       { color: "#B03A3A", dash: "3 5",   width: 2.0, label: "rejected" },
};
const arcStyle = (m) => m.accepted === false ? ARC.rejected
  : (m.proposed_by === "human" ? ARC.human : (ARC[m.method] || ARC.raw));
const Q_COLS = [["overall_quality","Qual"],["blur_quality","Sharp"],
                ["spot_extraction_quality","Spots"],["body_extraction_quality","Body"]];

// The two re-extraction buttons draw the pipeline they re-run: the anatomy disc IS a tiny anatomy
// (green head dot, grey spine, red tail dot) and the purple disc carries a white salamander.
const ICON = {
  anatomy: '<svg width="15" height="15" viewBox="0 0 16 16">' +
    '<path d="M8 2.5 C5 5.5 11 8.5 8 13.5" fill="none" stroke="#8A8A8A" stroke-width="1.6"/>' +
    '<circle cx="8" cy="2.5" r="2" fill="#008000"/><circle cx="8" cy="13.5" r="2" fill="#C00000"/></svg>',
  salamander: '<svg width="15" height="15" viewBox="0 0 16 16">' +
    '<path d="M8 1.6 C6.6 3.4 6.6 5 7.4 6.4 C8.4 8 8.4 9.6 7.2 11 C6.2 12.2 6.4 13.4 8 14.4 ' +
    'C7.2 12.8 7.6 12 8.6 10.8 C9.9 9.2 9.9 7.4 8.9 5.9 C8.2 4.8 8.2 3.4 8 1.6 Z" fill="#fff"/>' +
    '<path d="M7.4 5.2 L4.8 3.9 M8.8 5.6 L11.4 4.2 M7.5 10 L4.9 11.6 M8.8 9.8 L11.5 11.4" ' +
    'stroke="#fff" stroke-width="1.3" stroke-linecap="round"/></svg>',
  regen: '<svg width="14" height="14" viewBox="0 0 16 16">' +
    '<path d="M13 8a5 5 0 1 1-1.7-3.8" fill="none" stroke="#fff" stroke-width="1.7" ' +
    'stroke-linecap="round"/><path d="M12.6 1.6v3.2H9.4" fill="none" stroke="#fff" ' +
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>' +
    '<circle cx="8" cy="8" r="1.6" fill="#fff"/></svg>',
  bin: '<svg width="13" height="13" viewBox="0 0 16 16">' +
    '<path d="M3.5 4.5h9l-.9 9.2a1 1 0 0 1-1 .9H5.4a1 1 0 0 1-1-.9L3.5 4.5Z" fill="#E6E6E6"/>' +
    '<path d="M2.6 3.4h10.8M6.4 3.4V2.2h3.2v1.2" stroke="#E6E6E6" stroke-width="1.3" ' +
    'stroke-linecap="round" fill="none"/></svg>',
};
const $ = (s, r) => (r || document).querySelector(s);
const el = (tag, cls, txt) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const svgEl = (tag) => document.createElementNS("http://www.w3.org/2000/svg", tag);
const nowISO = () => new Date().toISOString().slice(0, 19);
const fmt = (v, nd) => (v == null || Number.isNaN(v)) ? "–" : Number(v).toFixed(nd == null ? 2 : nd);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const S = {
  boot: null, review: null, order: [], summaries: {}, idx: 0, label: "",
  geo: null, rec: null, machine: null, rev: 0,
  pending: null,          // {sid, spot} — first click of a match / origin of a miss
  chain: [],              // members of the group being built
  missArm: false,         // M pressed: next card click records a miss
  activeSid: null, hoverCard: null, selEdge: null, sortKey: "score", sortDir: -1,
  saveTimer: null, inflight: false, dirty: false,
};

// ---------------------------------------------------------------- helpers
const isSynth = (sid) => { const t = sid.split("_").pop(); return /^g\d+$/.test(t); };
const ssidOf = (sid, spot) => { const p = sid.split("_"); p[0] = p[0].toUpperCase();
  return p.join("-") + "-" + String(spot).padStart(2, "0"); };
const edgeKey = (sidA, spotA, sidB, spotB) => {
  const a = sidA + "#" + spotA, b = sidB + "#" + spotB;
  return a < b ? a + "|" + b : b + "|" + a;
};
const imgOf = (sid) => (S.geo ? S.geo.images.find((i) => i.sid === sid) : null);
const spotOf = (sid, spot) => { const im = imgOf(sid);
  return im ? im.spots.find((s) => s.id === spot) : null; };
const interestBin = (v) => (v == null ? null : RAMP[clamp(Math.floor(v * 5), 0, 4)]);

function ensureRecord(label) {
  const inds = S.review.individuals;
  if (!inds[label]) {
    inds[label] = { done: false, decision: "unreviewed", reasons: [], reviewed_at: null,
      duplicate_of: null, deleted: false, deleted_reason: null, note: "",
      images: {}, matches: [], misses: [] };
  }
  const rec = inds[label];
  for (const k of ["images", "matches", "misses", "reasons"]) if (!rec[k]) rec[k] = (k === "images" ? {} : []);
  return rec;
}

function ensureImage(sid) {
  const im = imgOf(sid);
  if (!S.rec.images[sid]) {
    S.rec.images[sid] = { is_synthetic: im ? im.is_synthetic : isSynth(sid),
      decision: "unreviewed", eval_ok: im && im.is_synthetic ? false : true, train_ok: true,
      split: null, reasons: [], cascaded: false, deleted: false, deleted_reason: null,
      duplicate_of: null, flagged: false, note: "", quality_at_review: null,
      n_spots: im ? im.spots.length : null, reviewed_at: null };
  }
  return S.rec.images[sid];
}

/** Snapshot of the numbers the reviewer actually saw — the review is unauditable without it. */
function qualitySnapshot(sid) {
  const im = imgOf(sid); if (!im) return null;
  const out = { passes_q04: im.passes_q04 };
  for (const [k] of Q_COLS) out[k] = im.quality[k];
  out.lighting_quality = im.quality.lighting_quality;
  out.spots_outside_frac = im.quality.spots_outside_frac;
  return out;
}

/** Machine edges -> match records, deduped BY EDGE so two matchers never double-count. */
function mergeMachine() {
  if (!S.machine) return;
  const byEdge = new Map();
  for (const m of S.rec.matches) {
    for (const e of edgesOfMatch(m)) byEdge.set(e.key, m);
  }
  let n = 0;
  const dismissed = new Set(S.rec.dismissed || []);
  for (const e of S.machine.edges) {
    const key = edgeKey(e.a.image, e.a.spot, e.b.image, e.b.spot);
    if (dismissed.has(key)) continue;
    const hit = byEdge.get(key);
    if (hit) {
      if (hit.proposed_by === "algorithm" && hit.method !== S.machine.method) {
        hit.also_proposed_by = hit.also_proposed_by || [];
        if (!hit.also_proposed_by.includes(S.machine.method)) hit.also_proposed_by.push(S.machine.method);
      }
      continue;
    }
    const mk = (side) => { const sp = spotOf(side.image, side.spot);
      return { ssid: side.ssid, image: side.image, spot: side.spot,
               xy: sp ? sp.xy : null,
               axis: sp ? { axis_t: sp.axis_t, axis_side: sp.side, bin: sp.bin } : null }; };
    const rec = { match_id: S.label + "-a" + String(S.rec.matches.length + 1).padStart(2, "0"),
      proposed_by: "algorithm", method: S.machine.method, also_proposed_by: [],
      accepted: true, rejected_note: null, interesting: false, stale: false,
      members: [mk(e.a), mk(e.b)],
      edges: [{ a: e.a.ssid, b: e.b.ssid, score: e.score }],
      created_at: nowISO(), reviewed_at: null };
    S.rec.matches.push(rec); byEdge.set(key, rec); n++;
  }
  return n;
}

function edgesOfMatch(m) {
  const out = [];
  const bySsid = new Map((m.members || []).map((x) => [x.ssid, x]));
  for (const e of m.edges || []) {
    const a = bySsid.get(e.a), b = bySsid.get(e.b);
    if (!a || !b) continue;
    out.push({ key: edgeKey(a.image, a.spot, b.image, b.spot), a, b, score: e.score, match: m });
  }
  if (!out.length && (m.members || []).length >= 2) {          // chain without explicit edges
    for (let i = 0; i + 1 < m.members.length; i++) {
      const a = m.members[i], b = m.members[i + 1];
      out.push({ key: edgeKey(a.image, a.spot, b.image, b.spot), a, b, score: null, match: m });
    }
  }
  return out;
}

const allEdges = () => S.rec ? S.rec.matches.flatMap(edgesOfMatch) : [];

// ---------------------------------------------------------------- save queue
const LSKEY = () => "prep:" + (S.boot ? S.boot.dataset : "?") + ":pending";

function markDirty() {
  S.dirty = true;
  try { localStorage.setItem(LSKEY(), JSON.stringify(
    { label: S.label, record: S.rec, rev: S.rev, at: nowISO() })); } catch (e) { /* quota */ }
  setSave("busy", "☁ saving…");
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(flush, 300);
}

async function flush() {
  if (!S.dirty || S.inflight || !S.label) return;
  S.inflight = true;
  const body = { rev: S.rev, individual: S.label, record: S.rec,
    ui: S.review.ui, rejection_reasons: S.review.rejection_reasons,
    reprocess_queue: S.review.reprocess_queue, reviewer: S.review.reviewer };
  try {
    const r = await fetch("/api/edit", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json();
    if (r.status === 409) { S.inflight = false; conflict(data); return; }
    if (!r.ok) throw new Error(data.error || r.status);
    S.rev = data.rev; S.review.rev = data.rev; S.review.counts = data.counts;
    S.dirty = false;
    localStorage.removeItem(LSKEY());
    setSave("", "☁ AUTO-SAVE [" + data.updated_at.slice(11) + "]");
    renderTop();
  } catch (err) {
    setSave("err", "⚠ save failed — retrying (" + err.message + ")");
    setTimeout(flush, 2000);
  } finally {
    S.inflight = false;
    if (S.dirty) { clearTimeout(S.saveTimer); S.saveTimer = setTimeout(flush, 400); }
  }
}

function setSave(cls, txt) { const n = $("#save"); n.className = cls; n.textContent = txt; }

/** A `rev` mismatch means another tab or reviewer wrote. Never silently win. */
function conflict(data) {
  showBanner("Another tab or reviewer wrote to review.json (server rev " + data.rev +
    ", this page has " + S.rev + "). Your unsaved edits to " + S.label + " are still on screen.",
    [["Reload from disk (discard mine)", () => location.reload()],
     ["Keep mine and overwrite", () => { S.rev = data.rev; S.review.rev = data.rev;
        hideBanner(); markDirty(); }]]);
  setSave("err", "⚠ conflict — not saved");
}

function showBanner(msg, actions) {
  const b = $("#banner"); b.textContent = msg;
  for (const [label, fn] of (actions || [])) {
    const btn = el("button", null, label); btn.onclick = fn; b.appendChild(btn);
  }
  b.style.display = "block";
}
const hideBanner = () => { $("#banner").style.display = "none"; };

// ---------------------------------------------------------------- boot
async function boot() {
  const r = await fetch("/api/bootstrap");
  S.boot = await r.json();
  S.review = S.boot.review;
  S.rev = S.review.rev;
  S.order = S.review.review_order || [];
  for (const s of S.boot.individuals) S.summaries[s.label] = s;
  $("#dsName").textContent = S.boot.dataset;
  $("#topN").value = S.review.ui.top_n_matches;
  S.norm = !!S.review.ui.normalized_view;
  renderOptions(); renderReasons(); renderRamp();
  // ?label=ac_3 opens (and links to) one animal; otherwise resume after the last edited one.
  const want = new URLSearchParams(location.search).get("label");
  const resume = (want && S.order.includes(want) ? want : null) ||
    (S.review.cursor && S.review.cursor.resume_at) || S.order[0];
  const start = S.order.indexOf(resume);
  await goto(start >= 0 ? start : 0);
  restorePending();
}

/** A crash between mutation and POST leaves the queue in localStorage — offer it back. */
function restorePending() {
  let raw; try { raw = localStorage.getItem(LSKEY()); } catch (e) { return; }
  if (!raw) return;
  let p; try { p = JSON.parse(raw); } catch (e) { return; }
  if (!p || !p.label) return;
  showBanner("Unsaved edits to " + p.label + " from " + p.at + " were found in this browser.",
    [["Restore them", async () => { hideBanner();
        const i = S.order.indexOf(p.label); if (i >= 0) await goto(i);
        S.review.individuals[p.label] = p.record; S.rec = p.record; renderAll(); markDirty(); }],
     ["Discard", () => { localStorage.removeItem(LSKEY()); hideBanner(); }]]);
}

async function goto(idx) {
  if (!S.order.length) return;
  await flush();
  S.idx = clamp(idx, 0, S.order.length - 1);
  S.label = S.order[S.idx];
  S.pending = null; S.missArm = false; S.selEdge = null; S.activeSid = null;
  const ui = S.review.ui;
  const q = new URLSearchParams({ label: S.label, matcher: ui.matcher,
    top_n: ui.top_n_matches, cutoff: ui.match_cutoff });
  const r = await fetch("/api/individual?" + q);
  if (!r.ok) { showBanner("could not load " + S.label + ": " + r.status); return; }
  S.geo = await r.json();
  S.machine = S.geo.machine;
  S.rec = ensureRecord(S.label);
  mergeMachine();                        // in memory only — navigation never writes
  history.replaceState(null, "", "?label=" + encodeURIComponent(S.label));
  renderAll();
  if ((S.geo.jobs || []).some((j) => j.state === "queued" || j.state === "running")) pollJobs();
}

const nextUnreviewed = () => {
  for (let k = 1; k <= S.order.length; k++) {
    const i = (S.idx + k) % S.order.length;
    const rec = S.review.individuals[S.order[i]];
    if (!rec || !rec.done) return i;
  }
  return S.idx;
};

// ---------------------------------------------------------------- render
function renderAll() { renderTop(); renderCards(); renderSidebar(); renderBottom(); renderJobs(); }

function renderTop() {
  const c = S.review.counts || {};
  const done = c.individuals_done || 0, tot = c.individuals_total || S.order.length;
  $("#progTxt").textContent = "Progress: " + done + " / " + tot + " Total   (" +
    (S.idx + 1) + " in order)";
  $("#progFill").style.width = (tot ? (100 * done / tot) : 0) + "%";
  $("#indLabel").textContent = S.label.toUpperCase();
  const sm = S.summaries[S.label] || {};
  const rec = S.rec || {};
  $("#qualHead").textContent = "mean quality " + fmt(sm.mean_quality) +
    (rec.deleted ? "  • DELETED (dup of " + (rec.duplicate_of || "?") + ")" : "") +
    (rec.decision === "reject" ? "  • REJECTED" : "");
  $("#indReject").classList.toggle("on", rec.decision === "reject");
}

function renderBottom() {
  const p = S.order[S.idx - 1], n = S.order[S.idx + 1];
  $("#prevLbl").textContent = p ? "(" + p + ")" : "—";
  $("#nextLbl").textContent = n ? "(" + n + ")" : "—";
}

function renderCards() {
  const wrap = $("#cards"); wrap.textContent = "";
  wrap.classList.toggle("normmode", !!S.norm);
  for (const im of S.geo.images) wrap.appendChild(buildCard(im));
  S.popSid = null;
  requestAnimationFrame(renderArcs);
}

function buildCard(im) {
  const rimg = ensureImage(im.sid);
  // The card border IS the decision: green accepted, red rejected, turquoise train-only.
  const state = rimg.deleted ? "deleted" : rimg.decision === "accept" ? "acc"
    : rimg.decision === "reject" ? "rej" : (rimg.train_ok ? "" : "notrain");
  // Pop only the card whose decision just changed, not every accepted card on every re-render.
  const card = el("div", "card " + state + (S.popSid === im.sid ? " pop" : "") +
    (im.provisional ? " prov" : ""));
  card.dataset.sid = im.sid;
  if (im.sid === S.activeSid) card.classList.add("active");

  const title = el("div", "cardTitle"); title.appendChild(el("span", null, im.sid));
  if (im.is_synthetic) title.appendChild(el("span", "badge", "gen"));
  if (im.provisional) {
    const b = el("span", "badge prov", "new");
    b.title = "Just regenerated. Served from " + (S.geo.regen_dir || "images/regen_…") +
      " — NOT in the packaged dataset yet. Judge it here; fold it in when you are happy:\n" +
      "pixi run fold-synth --synth regen_" + S.boot.dataset + " --into " + S.boot.images_folder;
    title.appendChild(b);
  }
  if (rimg.flagged) title.appendChild(el("span", "badge", "flag"));
  if (rimg.train_ok) title.appendChild(el("span", "badge train", "train"));
  card.appendChild(title);

  if (S.norm) { card.appendChild(buildNormalized(im)); card.appendChild(buildFooter(im, rimg));
    return card; }

  // ---- photo + SVG overlay (the browser's hit test IS the click resolution)
  const wrap = el("div", "photoWrap");
  wrap.dataset.vbw = im.width; wrap.dataset.vbh = im.height;
  const img = el("img"); img.src = "/photo?sid=" + encodeURIComponent(im.sid);
  img.alt = im.sid; img.draggable = false;
  img.style.opacity = S.review.ui.photo_opacity;
  img.onload = renderArcs;
  wrap.appendChild(img);

  const ov = svgEl("svg"); ov.setAttribute("class", "ov");
  ov.setAttribute("viewBox", "0 0 " + im.width + " " + im.height);
  ov.setAttribute("preserveAspectRatio", "none");
  const O = S.review.ui.overlays, ax = im.axis;
  if (ax) {
    if (O.outline && ax.left.length) {
      // Two passes: a dark halo under a light line. A single dark stroke (as in the mockup) is
      // invisible where it matters most — the body edge of a black animal.
      const d = poly(ax.left) + " " + poly(ax.right.slice().reverse()).replace("M", "L");
      for (const [cls, w, col, op] of [["halo", 3.2, "#000000", .55], ["line", 1.3, "#FFFFFF", .8]]) {
        const p = svgEl("path"); p.setAttribute("class", "outline " + cls);
        p.setAttribute("d", d); p.setAttribute("stroke", col);
        p.setAttribute("stroke-width", w * Math.max(1, im.width / 700));
        p.setAttribute("opacity", op); ov.appendChild(p);
      }
    }
    if (O.spine && ax.midline.length) {
      const p = svgEl("path"); p.setAttribute("class", "spine");
      p.setAttribute("stroke-width", 2.4 * Math.max(1, im.width / 700));
      p.setAttribute("d", poly(ax.midline)); ov.appendChild(p);
    }
    if (O.head_tail) {
      for (const [xy, cls] of [[ax.head, "head"], [ax.tail, "tail"]]) {
        if (!xy || xy[0] == null) continue;
        const c = svgEl("circle"); c.setAttribute("class", cls);
        c.setAttribute("cx", xy[0]); c.setAttribute("cy", xy[1]);
        c.setAttribute("r", Math.max(6, im.width / 60)); ov.appendChild(c);
      }
    }
  }
  if (O.spots) {
    const clicks = (S.geo.legacy_clicks || {})[im.sid] || [];
    const r = Math.max(2, im.width / 260);
    for (const sp of im.spots) {
      if (sp.contour.length > 2) {
        const p = svgEl("path");
        p.setAttribute("class", "spot");
        p.setAttribute("d", poly(sp.contour.map((c) => [sp.xy[0] + c[0], sp.xy[1] + c[1]])) + " Z");
        const col = S.review.ui.spot_fill === "interest" ? interestBin(sp.interest) : null;
        p.setAttribute("fill", col || "#EAD7AD");
        p.setAttribute("fill-opacity", S.review.ui.spot_fill_alpha);
        p.setAttribute("stroke", OUTLINE_HI);
        p.setAttribute("stroke-width", 1.8 * Math.max(1, im.width / 700));
        p.dataset.sid = im.sid; p.dataset.spot = sp.id;
        if (S.pending && S.pending.sid === im.sid && S.pending.spot === sp.id)
          p.classList.add("pend");
        ov.appendChild(p);
      }
      const c = svgEl("circle"); c.setAttribute("class", "cen");
      c.setAttribute("cx", sp.xy[0]); c.setAttribute("cy", sp.xy[1]); c.setAttribute("r", r);
      ov.appendChild(c);
      const t = svgEl("text"); t.setAttribute("class", "cnum");
      t.setAttribute("x", sp.xy[0] + r * 1.8); t.setAttribute("y", sp.xy[1] - r);
      t.setAttribute("font-size", Math.max(11, im.width / 26));
      t.setAttribute("stroke-width", Math.max(2, im.width / 300));
      t.textContent = sp.id;
      ov.appendChild(t);
      if (clicks.includes(sp.id)) {                   // human interesting-spot click: a RING
        const g = svgEl("circle"); g.setAttribute("class", "ring");
        g.setAttribute("cx", sp.xy[0]); g.setAttribute("cy", sp.xy[1]);
        g.setAttribute("r", r * 3.4); ov.appendChild(g);
      }
    }
  }
  wrap.appendChild(ov);
  card.appendChild(wrap);
  card.appendChild(buildFooter(im, rimg));
  return card;
}

const poly = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");

// ---------------------------------------------------------------- normalized view
const NVB = [200, 520];                  // viewBox of a normalized card: w, h
const NPAD = 46;                         // top/bottom padding, so head/tail discs sit inside

/** Pose removed: the body axis is a straight vertical line, head (green) always at the top and
 *  tail (red) always at the bottom, with every spot placed at its body-frame coordinate —
 *  y = axis_t (0 head .. 1 tail), x = lateral offset in half-widths. Two photos of a bent animal
 *  become directly comparable, which is the whole point of `axis_t` / `axis_offset` existing. */
function buildNormalized(im) {
  const [W, H] = NVB;
  const wrap = el("div", "photoWrap norm");
  wrap.dataset.vbw = W; wrap.dataset.vbh = H;
  const svg = svgEl("svg"); svg.setAttribute("class", "ov");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("preserveAspectRatio", "none");
  wrap.appendChild(svg);

  const cx = W / 2, y0 = NPAD, y1 = H - NPAD, span = y1 - y0;
  const halfW = (im.quality && im.quality.avg_width_px) ? im.quality.avg_width_px / 2 : null;
  const len = (im.axis && im.axis.length_px) || im.length_px;

  const axis = svgEl("line"); axis.setAttribute("x1", cx); axis.setAttribute("x2", cx);
  axis.setAttribute("y1", y0); axis.setAttribute("y2", y1);
  axis.setAttribute("stroke", "#FFFFFF"); axis.setAttribute("stroke-width", 4);
  svg.appendChild(axis);
  for (const [y, col] of [[y0, "#00C000"], [y1, "#E02020"]]) {
    const c = svgEl("circle"); c.setAttribute("cx", cx); c.setAttribute("cy", y);
    c.setAttribute("r", 15); c.setAttribute("fill", col); svg.appendChild(c);
  }
  if (!len || !im.spots.length) {
    const t = svgEl("text"); t.setAttribute("x", cx); t.setAttribute("y", H / 2);
    t.setAttribute("fill", "#888"); t.setAttribute("font-size", 15);
    t.setAttribute("text-anchor", "middle");
    t.textContent = len ? "no spots" : "no body axis"; svg.appendChild(t);
    return wrap;
  }

  const k = span / len;                          // px -> normalized units: 1 body length = span
  for (const sp of im.spots) {
    if (sp.axis_t == null) { sp._nxy = null; continue; }
    const u = halfW ? clamp((sp.offset || 0) / halfW, -2.6, 2.6) : 0;
    const x = cx + u * (W * 0.17), y = y0 + clamp(sp.axis_t, 0, 1) * span;
    sp._nxy = [x, y];
    if (sp.contour.length > 2) {
      const p = svgEl("path"); p.setAttribute("class", "spot");
      p.setAttribute("d", poly(sp.contour.map((c) => [x + c[0] * k, y + c[1] * k])) + " Z");
      const col = S.review.ui.spot_fill === "interest" ? interestBin(sp.interest) : null;
      p.setAttribute("fill", col || "#EAD7AD");
      p.setAttribute("fill-opacity", Math.min(1, S.review.ui.spot_fill_alpha + .35));
      p.setAttribute("stroke", OUTLINE_HI); p.setAttribute("stroke-width", 1);
      p.dataset.sid = im.sid; p.dataset.spot = sp.id;
      if (S.pending && S.pending.sid === im.sid && S.pending.spot === sp.id) p.classList.add("pend");
      svg.appendChild(p);
    }
    const t = svgEl("text"); t.setAttribute("class", "cnum");
    t.setAttribute("x", x + 7); t.setAttribute("y", y - 5);
    t.setAttribute("font-size", 12); t.setAttribute("stroke-width", 2.5);
    t.textContent = sp.id; svg.appendChild(t);
  }
  return wrap;
}

function buildFooter(im, rimg) {
  const f = el("div", "cardFoot");
  const r1 = el("div", "fr1");
  const mk = (cls, glyph, title, on, fn, icon) => {
    const b = el("button", "fbtn " + cls + (on ? " on" : ""), glyph);
    if (icon) b.innerHTML = icon;
    b.title = title; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b;
  };
  r1.appendChild(mk("acc", "✓", "Accept (A)", rimg.decision === "accept",
    () => setDecision(im.sid, "accept")));
  r1.appendChild(mk("rej", "✗", "Reject — needs a reason (R)", rimg.decision === "reject",
    () => setDecision(im.sid, "reject")));
  r1.appendChild(mk("tr", "T", "Train: include in the training set (T)", rimg.train_ok,
    () => { rimg.train_ok = !rimg.train_ok; touch(im.sid); }));
  r1.appendChild(mk("del", "", "Delete — duplicate of another photo (D)", rimg.deleted,
    () => deleteImage(im.sid), ICON.bin));
  r1.appendChild(el("span", "stem", im.sid));
  r1.appendChild(mk("an", "", "Re-run ANATOMY extraction — head dot, spine, tail dot (queued, never run here)",
    false, () => queueReprocess(im.sid, "anatomy"), ICON.anatomy));
  r1.appendChild(mk("pu", "", "Re-run PURPLE spot extraction (queued, never run here)", false,
    () => queueReprocess(im.sid, "repurple"), ICON.salamander));
  // Regeneration only makes sense FROM a real photo — a synthetic view is not a source.
  if (!im.is_synthetic) r1.appendChild(mk("gen", "", "Regenerate synthetic views from this photo — " +
    "BILLED, and it runs here (generate + purple + anatomy), then the new view appears in this family",
    false, () => regenerate(im.sid), ICON.regen));
  f.appendChild(r1);

  const r2 = el("div", "fr2");
  r2.appendChild(el("span", null, "Reason"));
  for (const reason of S.review.rejection_reasons) {
    if (reason.retired) continue;
    const c = el("button", "rchip" + (rimg.reasons.includes(reason.slug) ? " on" : ""), reason.label);
    c.onclick = (e) => { e.stopPropagation(); toggleReason(im.sid, reason.slug); };
    r2.appendChild(c);
  }
  const miss = el("button", "miss" + (S.missArm && S.pending && S.pending.sid !== im.sid ? " arm" : ""),
    "miss");
  miss.title = "Select a spot, press M (or this), then click the card where it was never extracted";
  miss.onclick = (e) => { e.stopPropagation(); armMiss(); };
  r2.appendChild(miss);
  f.appendChild(r2);

  const r3 = el("div", "fr3");
  for (const [key, label] of Q_COLS) {
    const cell = el("div", null, label);
    cell.appendChild(el("b", null, fmt(im.quality[key])));
    // Normalized against every real photo in the dataset: a bare 0.83 is not a decision.
    const n = (im.quality_norm || {})[key];
    const st = (S.boot.q_stats || {})[key];
    if (n && n.d != null) {
      const d = el("span", "dlt " + (n.d >= 0 ? "up" : "dn"),
        (n.d >= 0 ? "+" : "") + n.d.toFixed(2));
      d.title = label + " " + fmt(im.quality[key]) + "  vs dataset mean " + fmt(st && st.mean) +
        (n.pct == null ? "" : "  ·  " + Math.round(n.pct * 100) + "th percentile of " +
          (st ? st.n : "?") + " real photos") + (n.z == null ? "" : "  ·  z " + n.z);
      cell.appendChild(d);
    }
    r3.appendChild(cell);
  }
  const q = el("div", null, "q0.4");
  const known = im.quality_known;
  const badge = el("b", "q04 " + (im.passes_q04 ? "ok" : "no"),
    im.is_synthetic ? "✓gen" : (!known ? "?" : (im.passes_q04 ? "✓" : "✗")));
  badge.title = im.is_synthetic ? "kept by the gate because it is synthetic, not because it scored"
    : (!known ? "no image_quality row — the gate keeps it, unverified"
              : "overall_quality " + fmt(im.quality.overall_quality) + " vs gate " + S.boot.q_gate);
  q.appendChild(badge); r3.appendChild(q);
  r3.onclick = (e) => { e.stopPropagation();
    const p = e.currentTarget.parentElement.querySelector(".qpanel");
    p.classList.toggle("open"); };
  f.appendChild(r3);

  f.appendChild(el("div", "fr4", "Last Edit: " + (rimg.reviewed_at || "—")));
  f.appendChild(buildQPanel(im, rimg));
  return f;
}

function buildQPanel(im, rimg) {
  const p = el("div", "qpanel");
  const add = (title, keys) => {
    p.appendChild(el("h4", null, title));
    const t = el("table");
    const rows = keys.map((k) => [k, im.quality[k]]).filter((r) => r[1] != null)
      .sort((a, b) => a[1] - b[1]);                    // worst first — why is it low?
    for (const [k, v] of rows) {
      const tr = el("tr"); tr.appendChild(el("td", null, k));
      tr.appendChild(el("td", "v", fmt(v, 3))); t.appendChild(tr);
    }
    p.appendChild(t);
  };
  add("scores (worst first)", S.boot.score_fields);
  add("raw markers", S.boot.raw_fields);
  const axh = el("h4", null, "axis trust"); p.appendChild(axh);
  const at = el("table");
  for (const [k, v] of Object.entries(im.axis_meta || {})) {
    const tr = el("tr"); tr.appendChild(el("td", null, k));
    tr.appendChild(el("td", "v", v == null ? "–" : String(v))); at.appendChild(tr);
  }
  p.appendChild(at);

  p.appendChild(el("h4", null, "note"));
  const ta = el("textarea"); ta.value = rimg.note || "";
  ta.oninput = () => { rimg.note = ta.value; touch(im.sid); };
  ta.onclick = (e) => e.stopPropagation();
  p.appendChild(ta);
  const flag = el("button", "rchip" + (rimg.flagged ? " on" : ""), "⚠ needs a second opinion");
  flag.onclick = (e) => { e.stopPropagation(); rimg.flagged = !rimg.flagged; touch(im.sid); };
  p.appendChild(flag);
  p.onclick = (e) => e.stopPropagation();
  return p;
}

// ---------------------------------------------------------------- arcs + misses
function renderArcs() {
  const svg = $("#arcs"), cards = $("#cards");
  svg.textContent = "";
  if (!S.geo) return;
  const box = cards.getBoundingClientRect();
  svg.setAttribute("width", cards.scrollWidth); svg.setAttribute("height", cards.scrollHeight);
  const defs = svgEl("defs");
  defs.innerHTML = '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">' +
    '<feDropShadow dx="0" dy="0" stdDeviation="2.5" flood-color="#fff" flood-opacity=".55"/></filter>';
  svg.appendChild(defs);
  if (!S.review.ui.overlays.arcs) return;

  /** A spot's position in #cards coordinates — works in photo view and normalized view alike,
   *  because both write the card's own viewBox size onto the wrap. */
  const at = (sid, spot) => {
    const card = cards.querySelector('.card[data-sid="' + cssEsc(sid) + '"]');
    const sp = spotOf(sid, spot);
    if (!card || !sp) return null;
    const wrap = card.querySelector(".photoWrap");
    const r = wrap.getBoundingClientRect();
    if (!r.width) return null;
    const vw = Number(wrap.dataset.vbw), vh = Number(wrap.dataset.vbh);
    const xy = S.norm ? sp._nxy : sp.xy;
    if (!xy || !vw || !vh) return null;
    return [r.left - box.left + xy[0] * r.width / vw, r.top - box.top + xy[1] * r.height / vh];
  };

  const seen = new Map();                  // per card-pair counter, so score labels do not pile up
  for (const e of allEdges()) {
    const p1 = at(e.a.image, e.a.spot), p2 = at(e.b.image, e.b.spot);
    if (!p1 || !p2) continue;
    const pk = e.a.image + "|" + e.b.image;
    const rank = seen.get(pk) || 0; seen.set(pk, rank + 1);
    const rejected = e.match.accepted === false;
    const st = arcStyle(e.match);
    const d = bez(p1, p2);
    const path = svgEl("path"); path.setAttribute("class", "arc"); path.setAttribute("d", d);
    path.setAttribute("stroke", st.color);
    path.setAttribute("stroke-width", st.width);
    if (st.dash) path.setAttribute("stroke-dasharray", st.dash);
    if (rejected) path.setAttribute("opacity", ".85");

    // Fat invisible sibling so a 2 px line is easy to hit; hovering POPS the arc and tells you
    // what it is. Holding Ctrl passes clicks through to the spots underneath.
    const hit = svgEl("path"); hit.setAttribute("class", "hit"); hit.setAttribute("d", d);
    const pop = (on) => {
      path.setAttribute("stroke-width", on ? st.width + 2.2 : st.width);
      path.setAttribute("filter", on ? "url(#glow)" : "");
      S.selEdge = on ? e.key : null;
    };
    hit.onmouseenter = (ev) => { pop(true); showArcTip(ev, e); };
    hit.onmousemove = (ev) => moveTip(ev);
    hit.onmouseleave = () => { pop(false); hideTip(); };
    hit.onclick = () => toggleMatch(e.match);
    path.style.pointerEvents = "none";
    if (S.selEdge === e.key) pop(true);
    svg.appendChild(path); svg.appendChild(hit);

    // Label at the curve's own midpoint, staggered per rank — several arcs share every gutter.
    const mid = bezMid(p1, p2);
    mid[0] += (rank % 3 - 1) * 26;
    mid[1] += (rank % 4) * 13 - 22;
    if (e.score != null) {
      const t = svgEl("text"); t.setAttribute("x", mid[0]); t.setAttribute("y", mid[1]);
      t.setAttribute("fill", rejected ? "#C08080" : "var(--machine-ink)");
      t.setAttribute("font-size", "11"); t.setAttribute("text-anchor", "middle");
      t.setAttribute("paint-order", "stroke"); t.setAttribute("stroke", "#000");
      t.setAttribute("stroke-width", "3");
      t.textContent = e.score.toFixed(2); svg.appendChild(t);
    }
    if (rejected) {
      const x = svgEl("text"); x.setAttribute("x", mid[0]); x.setAttribute("y", mid[1] + 4);
      x.setAttribute("fill", "var(--reject)"); x.setAttribute("font-size", "20");
      x.setAttribute("text-anchor", "middle"); x.setAttribute("font-weight", "bold");
      x.textContent = "✕"; svg.appendChild(x);
    }
  }

  // misses: a blue DASHED STUB to an open circle — deliberately not an arc (nothing at the far end)
  for (const ms of S.rec.misses || []) {
    const p1 = at(ms.image, ms.spot);
    const card = cards.querySelector('.card[data-sid="' + cssEsc(ms.partner) + '"]');
    if (!p1 || !card) continue;
    const wrap = card.querySelector(".photoWrap");
    const r = wrap.getBoundingClientRect();
    const vw = Number(wrap.dataset.vbw), vh = Number(wrap.dataset.vbh);
    // partner_xy is in SOURCE pixels; in the normalized view there is no such place, so aim at
    // the middle of the axis instead of pretending to know where it was.
    const xy = (!S.norm && ms.partner_xy && ms.partner_xy[0] != null) ? ms.partner_xy
      : [vw / 2, vh / 2];
    const p2 = [r.left - box.left + xy[0] * r.width / vw,
                r.top - box.top + xy[1] * r.height / vh];
    const path = svgEl("path"); path.setAttribute("class", "arc");
    path.setAttribute("d", bez(p1, p2)); path.setAttribute("stroke", ARC.human.color);
    path.setAttribute("stroke-width", "2"); path.setAttribute("stroke-dasharray", "3 4");
    path.onclick = () => removeMiss(ms); svg.appendChild(path);
    const c = svgEl("circle"); c.setAttribute("cx", p2[0]); c.setAttribute("cy", p2[1]);
    c.setAttribute("r", 7); c.setAttribute("fill", "none");
    c.setAttribute("stroke", ARC.human.color); c.setAttribute("stroke-width", "2");
    svg.appendChild(c);
  }
}

const sagOf = (p1, p2) => Math.min(70, Math.abs((p2[0] - p1[0]) / 2) * .5 + 18);
const bez = (p1, p2) => { const dx = (p2[0] - p1[0]) / 2, sag = sagOf(p1, p2);
  return "M" + p1[0] + " " + p1[1] + " C" + (p1[0] + dx) + " " + (p1[1] + sag) + " " +
    (p2[0] - dx) + " " + (p2[1] + sag) + " " + p2[0] + " " + p2[1]; };
/** The cubic's own midpoint (t = 0.5) — the straight-line midpoint misses a sagging arc. */
const bezMid = (p1, p2) => { const dx = (p2[0] - p1[0]) / 2, sag = sagOf(p1, p2);
  const c1 = [p1[0] + dx, p1[1] + sag], c2 = [p2[0] - dx, p2[1] + sag];
  return [(p1[0] + 3 * c1[0] + 3 * c2[0] + p2[0]) / 8, (p1[1] + 3 * c1[1] + 3 * c2[1] + p2[1]) / 8]; };
const cssEsc = (s) => s.replace(/"/g, '\\"');

// ---------------------------------------------------------------- sidebar
function renderSidebar() { renderMatchTable(); renderMisses(); renderQualityTable(); renderTop(); }

function renderMatchTable() {
  const body = $("#mtBody"); body.textContent = "";
  let rows = allEdges();
  const key = S.sortKey || "score", dir = S.sortDir == null ? -1 : S.sortDir;
  rows.sort((x, y) => {
    const pick = (e) => key === "score" ? (e.score == null ? 2 : e.score)
      : key === "accepted" ? (e.match.accepted === false ? 0 : 1)
      : key === "a" ? e.a.ssid : e.b.ssid;
    const a = pick(x), b = pick(y);
    return (a > b ? 1 : a < b ? -1 : 0) * dir;
  });
  const cap = S.showAllMatches ? rows.length : 12;
  for (const e of rows.slice(0, cap)) {
    const st = arcStyle(e.match);
    const tr = el("tr", "mrow" + (e.match.accepted === false ? " rejected" : "") +
      (S.selEdge === e.key ? " sel" : ""));
    const a = el("td", "ss", e.a.ssid), b = el("td", "ss", e.b.ssid);
    a.style.color = st.color; b.style.color = st.color;
    tr.appendChild(a); tr.appendChild(b);
    tr.appendChild(el("td", "sc", e.score == null ? "–" : e.score.toFixed(2)));

    const acc = el("td");
    const pill = el("button", "ipill " + (e.match.accepted === false ? "f" : "t"),
      e.match.accepted === false ? "✗" : "✓");
    pill.title = "accepted — click to " + (e.match.accepted === false ? "restore" : "reject") +
      " (rejections are kept as hard negatives)";
    pill.onclick = (ev) => { ev.stopPropagation(); toggleMatch(e.match); };
    acc.appendChild(pill); tr.appendChild(acc);

    const und = el("td");
    const x = el("button", "undo", "✕");
    x.title = e.match.proposed_by === "human"
      ? "remove this match (undo — it was never a machine proposal)"
      : "dismiss this proposal: drops the row and stops this matcher re-proposing the edge.\n" +
        "To keep it as a hard negative, use the ✓/✗ pill instead.";
    x.onclick = (ev) => { ev.stopPropagation(); removeMatch(e.match, e.key); };
    und.appendChild(x); tr.appendChild(und);

    tr.title = (e.match.proposed_by === "human" ? "human" : e.match.method || "algorithm") +
      (e.match.also_proposed_by && e.match.also_proposed_by.length
        ? " (+ " + e.match.also_proposed_by.join(", ") + ")" : "") +
      (e.score == null ? "" : ", score " + e.score.toFixed(3)) +
      ", proposed " + (e.match.created_at || "?");
    tr.onmouseenter = () => { S.selEdge = e.key; tr.classList.add("sel"); renderArcs(); };
    tr.onmouseleave = () => { S.selEdge = null; tr.classList.remove("sel"); renderArcs(); };
    body.appendChild(tr);
  }
  const more = $("#mtMore");
  more.textContent = rows.length > 12
    ? (S.showAllMatches ? "Show fewer" : "Show all " + rows.length) : "";
  more.style.display = rows.length > 12 ? "block" : "none";
  more.onclick = () => { S.showAllMatches = !S.showAllMatches; renderMatchTable(); };
  const note = S.machine && S.machine.note ? S.machine.note : "";
  $("#mtEmpty").textContent = rows.length ? note
    : (note || "no matches — click a spot, then a spot on another card");
  renderArcLegend();
}

/** Type two SSIDs to assert a match without hunting for the spots on the photo. */
function addMatchByTyping() {
  const inA = $("#addA"), inB = $("#addB");
  const parse = (raw) => {
    const s = (raw || "").trim();
    if (!s) return null;
    const m = s.replace(/-/g, "_").match(/^(.*)_(\d+)$/);       // AC-3-1-14 -> ac_3_1 + 14
    if (!m) return null;
    const sid = m[1].toLowerCase(), spot = Number(m[2]);
    return spotOf(sid, spot) ? { sid, spot } : null;
  };
  const a = parse(inA.value), b = parse(inB.value);
  if (!a || !b) { showBanner("Type two SSIDs of spots in THIS family, e.g. AC-3-1-14 and " +
    "AC-3-2-15 (an unknown spot is refused rather than invented).", [["ok", hideBanner]]); return; }
  if (a.sid === b.sid) { showBanner("A match links two DIFFERENT photos.", [["ok", hideBanner]]);
    return; }
  S.pending = a;
  connect(a, b);
  inA.value = ""; inB.value = "";
}

function renderArcLegend() {
  const box = $("#arcLegend"); box.textContent = "";
  const used = new Set(allEdges().map((e) => {
    const m = e.match;
    return m.accepted === false ? "rejected" : (m.proposed_by === "human" ? "human" : (m.method || "raw"));
  }));
  for (const key of ["human", "raw", "strict_hand_pos", "logreg", "rejected"]) {
    if (!used.has(key)) continue;
    const row = el("div", "lgd");
    const s = svgEl("svg"); s.setAttribute("width", 26); s.setAttribute("height", 8);
    const p = svgEl("line"); p.setAttribute("x1", 0); p.setAttribute("y1", 4);
    p.setAttribute("x2", 26); p.setAttribute("y2", 4);
    p.setAttribute("stroke", ARC[key].color); p.setAttribute("stroke-width", ARC[key].width);
    if (ARC[key].dash) p.setAttribute("stroke-dasharray", ARC[key].dash);
    s.appendChild(p); row.appendChild(s);
    row.appendChild(el("span", null, ARC[key].label));
    box.appendChild(row);
  }
}

function renderMisses() {
  const box = $("#missList"); box.textContent = "";
  const list = S.rec.misses || [];
  $("#missHead").textContent = "Misses (" + list.length + ")";
  for (const ms of list) {
    const row = el("div", "msrow");
    row.appendChild(el("span", null, ssidOf(ms.image, ms.spot) + " → " + ms.partner));
    const x = el("button", null, "✕"); x.title = "remove this miss";
    x.onclick = () => removeMiss(ms); row.appendChild(x);
    box.appendChild(row);
  }
}

function renderOptions() {
  const box = $("#options"); box.textContent = "";
  const ui = S.review.ui;
  const row = (label, node) => { const d = el("div", "opt");
    d.appendChild(el("span", null, label)); d.appendChild(node); box.appendChild(d); return d; };

  const view = el("div");
  for (const [v, label] of [[false, "photo"], [true, "normalized"]]) {
    const b = el("button", "rchip" + (S.norm === v ? " on" : ""), label);
    b.title = v ? "Pose removed: head green at the top, tail red at the bottom, every spot at its " +
      "body-frame position (axis_t, lateral offset). Two photos of a bent animal become comparable."
      : "The photograph with overlays";
    b.onclick = () => setNormalized(v);
    view.appendChild(b);
  }
  row("View (N)", view);

  const sel = el("select");
  for (const m of S.boot.matchers) { const o = el("option", null, m); o.value = m;
    if (m === ui.matcher) o.selected = true; sel.appendChild(o); }
  sel.onchange = async () => { ui.matcher = sel.value; markDirty(); await reloadMachine(); };
  row("Matcher", sel);

  const cut = el("input"); cut.type = "range"; cut.min = 0; cut.max = 1; cut.step = 0.02;
  cut.value = ui.match_cutoff;
  cut.oninput = async () => { ui.match_cutoff = Number(cut.value); markDirty();
    cut.title = cut.value; await reloadMachine(); };
  row("Score cutoff", cut);

  const hover = el("div");
  for (const [f, label] of [["ssid", "SSID"], ["interest", "Interest"], ["spot", "Spot"],
                            ["head_dist", "Head"]]) {
    const b = el("button", "rchip" + (ui.tooltip_fields.includes(f) ? " on" : ""), label);
    b.onclick = () => { const i = ui.tooltip_fields.indexOf(f);
      if (i >= 0) ui.tooltip_fields.splice(i, 1); else ui.tooltip_fields.push(f);
      markDirty(); renderOptions(); };
    hover.appendChild(b);
  }
  row("On hover", hover);

  const lay = el("div");
  for (const [k, label] of [["spots", "spots"], ["outline", "outline"], ["spine", "spine"],
                            ["head_tail", "head/tail"], ["arcs", "arcs"]]) {
    const b = el("button", "rchip" + (ui.overlays[k] ? " on" : ""), label);
    b.onclick = () => { ui.overlays[k] = !ui.overlays[k]; markDirty(); renderOptions();
      renderCards(); };
    lay.appendChild(b);
  }
  row("Overlays", lay);

  const op = el("input"); op.type = "range"; op.min = .15; op.max = 1; op.step = .05;
  op.value = ui.photo_opacity;
  op.oninput = () => { ui.photo_opacity = Number(op.value); markDirty();
    for (const i of document.querySelectorAll(".photoWrap img")) i.style.opacity = op.value; };
  row("Photo opacity", op);

  const fill = el("select");
  for (const v of ["interest", "flat"]) { const o = el("option", null, v); o.value = v;
    if (ui.spot_fill === v) o.selected = true; fill.appendChild(o); }
  fill.onchange = () => { ui.spot_fill = fill.value; markDirty(); renderCards(); };
  row("Spot fill", fill);

  const al = el("input"); al.type = "range"; al.min = .05; al.max = 1; al.step = .05;
  al.value = ui.spot_fill_alpha;
  al.oninput = () => { ui.spot_fill_alpha = Number(al.value); markDirty();
    for (const p of document.querySelectorAll("svg.ov path.spot"))
      p.setAttribute("fill-opacity", al.value); };
  row("Fill alpha", al);

  const who = el("input"); who.type = "text"; who.size = 8; who.value = S.review.reviewer || "";
  who.style.cssText = "background:#242426;color:inherit;border:1px solid var(--line)";
  who.onchange = () => { S.review.reviewer = who.value; markDirty(); };
  row("Reviewer", who);

  const exp = el("button", "chip", "export legacy + queue CSVs");
  exp.onclick = async () => { await flush();
    const r = await fetch("/api/export", { method: "POST" }); const d = await r.json();
    showBanner("exported " + d.interesting_labels + " interesting-spot labels over " +
      d.interesting_images + " images", [["ok", hideBanner]]); };
  box.appendChild(exp);
}

function setNormalized(on) {
  S.norm = !!on;
  S.review.ui.normalized_view = S.norm;
  markDirty(); renderOptions(); renderCards();
}

async function reloadMachine() {
  const ui = S.review.ui;
  const q = new URLSearchParams({ label: S.label, matcher: ui.matcher,
    top_n: ui.top_n_matches, cutoff: ui.match_cutoff });
  const r = await fetch("/api/individual?" + q);
  if (!r.ok) return;
  const geo = await r.json();
  S.machine = geo.machine;
  mergeMachine();
  renderMatchTable(); renderArcs();
}

function renderReasons() {
  const box = $("#reasonChips"); box.textContent = "";
  for (const r of S.review.rejection_reasons) {
    if (r.retired) continue;
    box.appendChild(el("div", "chip", r.label + (r.quick_key ? "  " + r.quick_key : "")));
  }
  const add = el("button", "chip add", "+ Add reason");
  add.onclick = () => {
    const label = prompt("New rejection reason (label):"); if (!label) return;
    const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
    if (S.review.rejection_reasons.some((r) => r.slug === slug)) return;
    S.review.rejection_reasons.push({ slug, label, quick_key:
      String(S.review.rejection_reasons.length + 1), added_at: nowISO() });
    markDirty(); renderReasons(); renderCards();
  };
  box.appendChild(add);
}

function renderQualityTable() {
  const head = $("#qtHead"), body = $("#qtBody");
  head.textContent = ""; body.textContent = "";
  head.appendChild(el("th", null, "img"));
  for (const [, label] of Q_COLS) head.appendChild(el("th", null, label));
  head.appendChild(el("th", null, "q0.4"));
  const rows = S.geo.images.slice().sort((a, b) => {
    const av = a.quality.overall_quality, bv = b.quality.overall_quality;
    return (av == null ? -1 : av) - (bv == null ? -1 : bv);       // worst first
  });
  for (const im of rows) {
    const tr = el("tr", im.sid === S.activeSid ? "sel" : "");
    tr.appendChild(el("td", null, im.sid));
    for (const [k] of Q_COLS) {
      const td = el("td");
      const v = im.quality[k];
      const n = (im.quality_norm || {})[k];
      if (v != null) {
        // Neutral GREY ramp (hue means other things here) driven by the DATASET PERCENTILE, so
        // the shade says "bad for this dataset", not "bad on an absolute 0..1 scale".
        const bg = el("div", "qbg");
        const p = n && n.pct != null ? n.pct : clamp(v, 0, 1);
        const g = Math.round(38 + 74 * p);
        bg.style.background = "rgb(" + g + "," + g + "," + g + ")";
        td.appendChild(bg);
      }
      const cell = el("span", "qcell", fmt(v));
      if (n && n.pct != null) cell.title = Math.round(n.pct * 100) + "th percentile  ·  " +
        (n.d >= 0 ? "+" : "") + n.d.toFixed(2) + " vs mean";
      td.appendChild(cell);
      tr.appendChild(td);
    }
    tr.appendChild(el("td", null, im.is_synthetic ? "✓gen"
      : (!im.quality_known ? "?" : (im.passes_q04 ? "✓" : "✗"))));
    tr.onclick = () => { S.activeSid = im.sid; renderCards(); renderQualityTable(); };
    body.appendChild(tr);
  }
  // The dataset itself, as the last row — what "good" means here.
  const st = S.boot.q_stats || {};
  const mean = el("tr", "meanrow");
  mean.appendChild(el("td", null, "dataset mean"));
  for (const [k] of Q_COLS) mean.appendChild(el("td", null, fmt(st[k] && st[k].mean)));
  mean.appendChild(el("td", null, ""));
  mean.title = "mean over the " + ((st.overall_quality && st.overall_quality.n) || "?") +
    " real photos in the dataset (synthetic views excluded)";
  body.appendChild(mean);
}

function renderRamp() {
  const box = $("#ramp"); box.textContent = "";
  box.appendChild(el("span", null, "interest 0"));
  for (const c of RAMP) { const i = el("i"); i.style.background = c; box.appendChild(i); }
  box.appendChild(el("span", null, "1"));
  const st = S.boot.interest;
  $("#interestState").textContent = st.state === "ready" ? ""
    : st.state === "building" ? "interest: computing (one-off, ~4 min) — flat fill until then"
    : st.state === "failed" ? "interest unavailable: " + st.error
    : "interest not cached — run: pixi run preprocess-interest";
}

// ---------------------------------------------------------------- mutations
function touch(sid) {
  if (sid) { const r = ensureImage(sid); r.reviewed_at = nowISO();
    r.quality_at_review = r.quality_at_review || qualitySnapshot(sid); }
  S.rec.reviewed_at = nowISO();
  markDirty(); renderCards(); renderSidebar();
}

function setDecision(sid, decision) {
  const r = ensureImage(sid);
  r.decision = r.decision === decision ? "unreviewed" : decision;   // pressing again clears
  if (r.decision === "reject") {
    r.eval_ok = false;
    if (!r.reasons.length) showBanner(
      "A rejection needs at least one reason — pick a chip (keys 1–4) or the histogram is useless.",
      [["ok", hideBanner]]);
  } else if (r.decision === "accept") {
    r.eval_ok = !(imgOf(sid) || {}).is_synthetic;                   // synthetics: never eval (#11)
    r.train_ok = true;
  } else { r.eval_ok = !(imgOf(sid) || {}).is_synthetic; }
  S.activeSid = sid;
  S.popSid = sid;                                       // the border pops on THIS card only
  maybeDone();
  touch(sid);
}

function toggleReason(sid, slug) {
  const r = ensureImage(sid);
  const i = r.reasons.indexOf(slug);
  if (i >= 0) r.reasons.splice(i, 1); else r.reasons.push(slug);
  touch(sid);
}

function deleteImage(sid) {
  const r = ensureImage(sid);
  if (r.deleted) { r.deleted = false; r.deleted_reason = null; r.duplicate_of = null;
    r.eval_ok = !isSynth(sid); r.train_ok = true; touch(sid); return; }
  const others = S.geo.images.map((i) => i.sid).filter((s) => s !== sid);
  const dup = prompt("Delete " + sid + " as a duplicate.\nWhich photo does it duplicate? " +
    "(the survivor)\n\n" + others.join(", "), others[0] || "");
  if (dup === null) return;
  r.deleted = true; r.deleted_reason = "duplicate"; r.duplicate_of = dup || null;
  r.eval_ok = false; r.train_ok = false;                            // redundant, not bad
  touch(sid);
}

function queueReprocess(sid, kind) {
  let action = kind;
  if (kind === "anatomy") {
    const billed = confirm("Anatomy re-extraction for " + sid + "\n\n" +
      "OK = correct_axis (free, geometric re-tip from the saved mask)\n" +
      "Cancel = reextract_body (BILLED Gemini call, stage 1 + 1b redo)");
    action = billed ? "correct_axis" : "reextract_body";
  }
  const note = prompt("Why? (goes in the queue entry)", "") || "";
  S.review.reprocess_queue.push({ image: sid, action, note, status: "requested",
    requested_at: nowISO(), completed_at: null });
  showBanner("queued " + action + " for " + sid + " — nothing runs from this UI; " +
    "drain it with the pixi task (see docs Flow 5)", [["ok", hideBanner]]);
  touch(sid);
}

// ---------------------------------------------------------------- regeneration
/** Regenerate synthetic views from a real photo. This one DOES spend money, so it asks first and
 *  says exactly how much; the queue-only rule covers the repair actions, not this. */
async function regenerate(sid) {
  const n = Number(prompt("Regenerate synthetic views from " + sid + ".\n\n" +
    "This RUNS now and is BILLED: about 3 Gemini calls per view\n" +
    "(1 generate + 1 purple + 1 anatomy, plus judge re-draws).\n\n" +
    "How many views? (1–4, Cancel to abort)", "1"));
  if (!n || !(n >= 1 && n <= 4)) return;
  try {
    const r = await fetch("/api/generate", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid, n }) });
    const job = await r.json();
    if (!r.ok) throw new Error(job.error || r.status);
    S.jobs = (S.jobs || []).filter((j) => j.id !== job.id).concat([job]);
    renderJobs(); pollJobs();
  } catch (err) {
    showBanner("could not start generation: " + err.message, [["ok", hideBanner]]);
  }
}

let jobTimer = null;
function pollJobs() {
  clearTimeout(jobTimer);
  jobTimer = setTimeout(async () => {
    try {
      const r = await fetch("/api/jobs");
      const data = await r.json();
      const wasRunning = (S.jobs || []).some((j) => j.state === "queued" || j.state === "running");
      S.jobs = data.jobs || [];
      renderJobs();
      const mine = S.jobs.filter((j) => j.label === S.label);
      const ready = mine.some((j) => j.state === "ready" && !j.seen);
      if (ready) {                               // the new view pops into the family view
        for (const j of mine) if (j.state === "ready") j.seen = true;
        await refreshGeo();
        showBanner("new synthetic view ready: " +
          mine.filter((j) => j.state === "ready").flatMap((j) => j.produced).join(", ") +
          " — provisional, shown from images/regen_… until you fold it in.",
          [["ok", hideBanner]]);
      }
      if (S.jobs.some((j) => j.state === "queued" || j.state === "running")) pollJobs();
      else if (wasRunning) renderJobs();
    } catch (e) { pollJobs(); }
  }, 2500);
}

/** Re-fetch the current family's geometry without touching the review record. */
async function refreshGeo() {
  const ui = S.review.ui;
  const q = new URLSearchParams({ label: S.label, matcher: ui.matcher,
    top_n: ui.top_n_matches, cutoff: ui.match_cutoff });
  const r = await fetch("/api/individual?" + q);
  if (!r.ok) return;
  S.geo = await r.json();
  S.machine = S.geo.machine;
  mergeMachine();
  renderAll();
}

function renderJobs() {
  const box = $("#jobs"); box.textContent = "";
  const jobs = (S.jobs || []).filter((j) => j.state !== "ready" || j.label === S.label);
  if (!jobs.length) { box.style.display = "none"; return; }
  box.style.display = "flex";
  for (const j of jobs) {
    const chip = el("div", "job " + j.state);
    chip.appendChild(el("span", null, j.label + " · " +
      (j.state === "running" ? j.step || "working" : j.state) +
      (j.state === "running" ? " (" + j.steps_done + "/" + j.steps_total + ")" : "") +
      (j.state === "ready" ? " · " + (j.produced.join(", ") || "nothing") : "")));
    if (j.error) chip.title = j.error;
    chip.onclick = () => showJobLog(j);
    box.appendChild(chip);
  }
}

function showJobLog(j) {
  const box = $("#modalBox"); box.textContent = "";
  box.appendChild(el("h3", null, "Regeneration " + j.id + " — " + j.label + " (" + j.state + ")"));
  box.appendChild(el("p", null, "source " + j.source + " · " + j.n + " view(s) · " +
    (j.mock ? "MOCK (no billed calls)" : "~" + j.billed_calls + " billed Gemini calls") +
    " · expecting " + j.expect.join(", ")));
  if (j.error) box.appendChild(el("p", "hint", j.error));
  const pre = el("pre"); pre.textContent = (j.log || []).join("\n");
  pre.style.cssText = "white-space:pre-wrap;font-size:10.5px;max-height:50vh;overflow:auto;" +
    "background:#111;padding:8px;color:#B8B8B8";
  box.appendChild(pre);
  $("#modal").classList.add("open");
}

function setIndividualDecision() {
  const rec = S.rec;
  rec.decision = rec.decision === "reject" ? "unreviewed" : "reject";
  if (rec.decision === "reject") {
    if (!rec.reasons.length) {
      const slug = prompt("Reject the whole animal. Reason slug? (" +
        S.review.rejection_reasons.map((r) => r.slug).join(", ") + ")", "blurry");
      if (!slug) { rec.decision = "unreviewed"; return; }
      rec.reasons = [slug];
    }
    for (const im of S.geo.images) {                   // cascades to UNREVIEWED images only
      const r = ensureImage(im.sid);
      if (r.decision === "unreviewed") { r.decision = "reject"; r.eval_ok = false;
        r.cascaded = true; r.reasons = rec.reasons.slice(); r.reviewed_at = nowISO();
        r.quality_at_review = r.quality_at_review || qualitySnapshot(im.sid); }
    }
    rec.done = true;
  }
  touch(null);
}

function deleteIndividual() {
  const rec = S.rec;
  if (rec.deleted) { rec.deleted = false; rec.deleted_reason = null; rec.duplicate_of = null;
    touch(null); return; }
  const other = prompt("This animal is the SAME as another label.\n" +
    "Which label survives? (its photos stay; this label is merged into it)", "");
  if (!other) return;
  rec.deleted = true; rec.deleted_reason = "duplicate"; rec.duplicate_of = other.trim();
  rec.done = true;
  touch(null);
}

/** `done` = every photo has a decision. It un-sets too, so clearing a decision re-opens the animal. */
function maybeDone() {
  S.rec.done = S.geo.images.every((im) => {
    const r = S.rec.images[im.sid];
    return r && (r.decision !== "unreviewed" || r.deleted);
  });
}

// ---- matches / misses
/** Selection is one spot at a time and a connection is one line.
 *
 *  click a spot        -> it turns PURPLE (selected)
 *  click it again      -> back to normal (deselected)
 *  click a second spot -> a blue line between the two, and BOTH deselect
 *
 *  Nothing stays open afterwards, so a single further click can never start drawing another line
 *  by accident. A spot may hold as many connections as it needs — chaining a spot across four
 *  photos is three separate lines, each removable on its own.
 */
function clickSpot(sid, spot) {
  if (!S.pending) { S.pending = { sid, spot }; S.activeSid = sid; renderCards(); return; }
  if (S.pending.sid === sid && S.pending.spot === spot) { clearPending(); return; }
  if (S.pending.sid === sid) {                  // same photo: nothing to connect, so re-select
    S.pending = { sid, spot }; renderCards(); return;
  }
  connect(S.pending, { sid, spot });
}

const memberOf = (sid, spot) => {
  const sp = spotOf(sid, spot) || {};
  return { ssid: ssidOf(sid, spot), image: sid, spot, xy: sp.xy || null,
    axis: { axis_t: sp.axis_t == null ? null : sp.axis_t, axis_side: sp.side || null,
            bin: sp.bin == null ? null : sp.bin } };
};

/** One gesture, two labels: the connection is match GT *and* an interesting-spot click (#31). */
function connect(a, b) {
  if (a.sid === b.sid) { clearPending(); return; }
  const key = edgeKey(a.sid, a.spot, b.sid, b.spot);
  const existing = allEdges().find((e) => e.key === key);
  if (existing) {
    // Already connected — a human drawing a machine's pair CONFIRMS it rather than duplicating it.
    const m = existing.match;
    if (m.proposed_by === "algorithm") {
      m.confirmed_by_human = true; m.accepted = true; m.interesting = true;
      m.reviewed_at = nowISO();
      showBanner("that pair was already proposed by " + (m.method || "a matcher") +
        " — marked as confirmed by you instead of duplicated", [["ok", hideBanner]]);
    }
    clearPending(); touch(null); return;
  }
  S.rec.matches.push({
    match_id: S.label + "-h" + String(S.rec.matches.length + 1).padStart(2, "0"),
    proposed_by: "human", method: null, also_proposed_by: [], accepted: true,
    rejected_note: null, interesting: true, stale: false,
    members: [memberOf(a.sid, a.spot), memberOf(b.sid, b.spot)],
    edges: [{ a: ssidOf(a.sid, a.spot), b: ssidOf(b.sid, b.spot), score: null }],
    created_at: nowISO(), reviewed_at: nowISO(),
  });
  clearPending();                               // both spots deselect: one click, one line
  touch(null);
}

function clearPending() {
  S.pending = null; S.missArm = false;
  renderCards(); renderArcs();
}

function toggleMatch(match) {
  match.accepted = match.accepted === false ? true : false;
  match.reviewed_at = nowISO();
  if (match.accepted === false && match.proposed_by === "algorithm" && !match.rejected_note) {
    const why = prompt("Why is this match wrong? (optional — it becomes a hard negative)", "");
    if (why) match.rejected_note = why;
  }
  touch(null);
}

function armMiss() {
  if (!S.pending) { showBanner("Select a spot first, then press M and click the card where it " +
    "was never extracted.", [["ok", hideBanner]]); return; }
  S.missArm = true; renderCards();
}

function addMiss(partnerSid, xy) {
  if (!S.pending) return;
  if (partnerSid === S.pending.sid) return;
  S.rec.misses.push({ image: S.pending.sid, spot: S.pending.spot, partner: partnerSid,
    partner_xy: xy ? [Math.round(xy[0] * 10) / 10, Math.round(xy[1] * 10) / 10] : null,
    reviewed_at: nowISO() });
  clearPending();
  touch(null);
}

/** The X column removes exactly ONE connection — never a neighbouring line.
 *
 *  A connection is normally its own record (2 members, 1 edge), so this drops the record. Records
 *  carrying several edges — a multi-member group from an older session — lose only the clicked
 *  edge, and any member left with no edge is pruned; the rest of the group survives.
 *
 *  A human line is simply undone. An algorithm proposal is also DISMISSED: the edge is remembered
 *  so switching matcher back does not re-add it. Rejection (the ✓/✗ pill) is the other action, and
 *  the one that keeps hard negatives.
 */
function removeMatch(match, key) {
  const edges = edgesOfMatch(match);
  if (match.proposed_by === "algorithm") {
    S.rec.dismissed = S.rec.dismissed || [];
    if (!S.rec.dismissed.includes(key)) S.rec.dismissed.push(key);
  }
  if (edges.length <= 1) {
    const i = S.rec.matches.indexOf(match);
    if (i >= 0) S.rec.matches.splice(i, 1);
  } else {
    const gone = edges.find((e) => e.key === key);
    match.edges = (match.edges || []).filter((e) =>
      !(gone && ((e.a === gone.a.ssid && e.b === gone.b.ssid) ||
                 (e.a === gone.b.ssid && e.b === gone.a.ssid))));
    const kept = new Set((match.edges || []).flatMap((e) => [e.a, e.b]));
    match.members = (match.members || []).filter((m) => kept.has(m.ssid));
    if (match.members.length < 2) {
      const i = S.rec.matches.indexOf(match);
      if (i >= 0) S.rec.matches.splice(i, 1);
    }
    match.reviewed_at = nowISO();
  }
  S.selEdge = null;
  touch(null);
}

function removeMiss(ms) {
  const i = S.rec.misses.indexOf(ms);
  if (i >= 0) S.rec.misses.splice(i, 1);
  touch(null);
}

// ---------------------------------------------------------------- hover tooltip
function showTip(ev, sid, spot) {
  const im = imgOf(sid), sp = spotOf(sid, spot);
  if (!im || !sp) return;
  const F = S.review.ui.tooltip_fields;
  const box = $("#tipBox"); box.textContent = "";
  if (F.includes("ssid")) {
    const line = el("div"); line.appendChild(el("span", null, "SSID: "));
    const parts = sp.ssid.split("-");
    line.appendChild(el("span", null, parts.slice(0, -1).join("-") + "-"));
    line.appendChild(el("span", "sp", parts[parts.length - 1]));
    box.appendChild(line);
  }
  if (F.includes("interest")) box.appendChild(el("div", null,
    "Interest: " + (sp.interest == null ? "– (not cached)" : fmt(sp.interest))));
  if (F.includes("head_dist")) {
    const len = im.axis && im.axis.length_px ? im.axis.length_px : im.length_px;
    const px = (sp.axis_t != null && len) ? Math.round(sp.axis_t * len) : null;
    const bad = im.axis && im.axis.judged_ok === false;
    box.appendChild(el("div", null, "Head: " + (bad ? "~" : "") + fmt(sp.axis_t) +
      (px == null ? "" : " (" + px + " px)") + (sp.side ? "   Side: " + sp.side : "")));
  }
  const shape = $("#tipShape"); shape.textContent = "";
  if (F.includes("spot") && sp.contour.length > 2) {
    const xs = sp.contour.map((p) => p[0]), ys = sp.contour.map((p) => p[1]);
    const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
    const k = 52 / Math.max(w, h, 1);
    const path = svgEl("path");
    path.setAttribute("d", poly(sp.contour.map((p) =>
      [32 + (p[0] - (Math.min(...xs) + w / 2)) * k, 32 + (p[1] - (Math.min(...ys) + h / 2)) * k])) + " Z");
    path.setAttribute("fill", "var(--magenta)");
    shape.appendChild(path);
    shape.style.display = "block";
  } else { shape.style.display = "none"; }
  const tip = $("#tip");
  tip.style.display = "flex";
  tip.style.left = Math.min(ev.clientX + 16, innerWidth - 240) + "px";
  tip.style.top = Math.min(ev.clientY + 8, innerHeight - 90) + "px";
}
const hideTip = () => { $("#tip").style.display = "none"; };

function moveTip(ev) {
  const tip = $("#tip");
  if (tip.style.display === "none") return;
  tip.style.left = Math.min(ev.clientX + 16, innerWidth - 250) + "px";
  tip.style.top = Math.min(ev.clientY + 8, innerHeight - 90) + "px";
}

/** Hovering an arc says what it is: both SSIDs, the score, and which matcher proposed it. */
function showArcTip(ev, e) {
  const box = $("#tipBox"); box.textContent = "";
  const st = arcStyle(e.match);
  const line = el("div");
  const a = el("span", null, e.a.ssid); a.style.color = st.color;
  const b = el("span", null, e.b.ssid); b.style.color = st.color;
  line.appendChild(a); line.appendChild(el("span", null, "  →  ")); line.appendChild(b);
  box.appendChild(line);
  box.appendChild(el("div", null, "Score: " + (e.score == null ? "– (human)" : e.score.toFixed(3)) +
    "   ·   " + (e.match.proposed_by === "human" ? "human" : (e.match.method || "algorithm")) +
    (e.match.accepted === false ? "   ·   REJECTED" : "")));
  box.appendChild(el("div", "hint", "click to " +
    (e.match.accepted === false ? "restore" : "reject") + "   ·   hold Ctrl to click through"));
  $("#tipShape").style.display = "none";
  $("#tip").style.display = "flex";
  moveTip(ev);
}

// ---------------------------------------------------------------- events
$("#cards").addEventListener("mouseover", (ev) => {
  const p = ev.target.closest("path.spot");
  const card = ev.target.closest(".card");
  if (card) S.hoverCard = card.dataset.sid;
  if (!p) { hideTip(); return; }
  p.classList.add("hov");
  showTip(ev, p.dataset.sid, Number(p.dataset.spot));
});
$("#cards").addEventListener("mousemove", (ev) => {
  const p = ev.target.closest("path.spot");
  if (p && $("#tip").style.display !== "none") {
    const tip = $("#tip");
    tip.style.left = Math.min(ev.clientX + 16, innerWidth - 240) + "px";
    tip.style.top = Math.min(ev.clientY + 8, innerHeight - 90) + "px";
  }
});
$("#cards").addEventListener("mouseout", (ev) => {
  const p = ev.target.closest("path.spot");
  if (p) p.classList.remove("hov");
  if (!ev.relatedTarget || !ev.relatedTarget.closest || !ev.relatedTarget.closest("path.spot")) hideTip();
});
$("#cards").addEventListener("click", (ev) => {
  const card = ev.target.closest(".card"); if (!card) return;
  const sid = card.dataset.sid;
  S.activeSid = sid;
  const p = ev.target.closest("path.spot");
  if (p) {
    const spot = Number(p.dataset.spot);
    if (S.missArm) { /* a spot click while armed still means "the miss is HERE" */
      const xy = imgXY(ev, card); addMiss(sid, xy); return;
    }
    clickSpot(sid, spot); renderSidebar(); return;
  }
  if (S.missArm && S.pending && sid !== S.pending.sid) { addMiss(sid, imgXY(ev, card)); return; }
  // A click on the photo but not on a spot: snap to the nearest centroid within ~4% of width.
  const xy = imgXY(ev, card);
  if (xy && S.pending == null) {
    const im = imgOf(sid); let best = null, bd = Infinity;
    for (const sp of im.spots) {
      const d = (sp.xy[0] - xy[0]) ** 2 + (sp.xy[1] - xy[1]) ** 2;
      if (d < bd) { bd = d; best = sp; }
    }
    if (best && Math.sqrt(bd) < im.width * 0.04) { clickSpot(sid, best.id); renderSidebar(); return; }
  }
  renderCards();
});

function imgXY(ev, card) {
  const wrap = card.querySelector(".photoWrap"); if (!wrap) return null;
  const r = wrap.getBoundingClientRect(); const im = imgOf(card.dataset.sid);
  if (!r.width || !im) return null;
  return [(ev.clientX - r.left) * im.width / r.width, (ev.clientY - r.top) * im.height / r.height];
}

$("#topN").addEventListener("change", async (e) => {
  S.review.ui.top_n_matches = clamp(Number(e.target.value) || 0, 0, 40);
  markDirty(); await reloadMachine();
});
for (const th of document.querySelectorAll("table.mt th")) {
  th.onclick = () => { const k = th.dataset.sort;
    S.sortDir = (S.sortKey === k) ? -(S.sortDir || -1) : -1; S.sortKey = k; renderMatchTable(); };
}
$("#prev").onclick = () => goto(S.idx - 1);
$("#next").onclick = () => goto(S.idx + 1);
$("#indReject").onclick = setIndividualDecision;
$("#indDelete").onclick = deleteIndividual;
$("#help").onclick = () => showHelp();
$("#modal").onclick = () => $("#modal").classList.remove("open");
$("#addGo").onclick = addMatchByTyping;
for (const id of ["#addA", "#addB"])
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") addMatchByTyping();
    e.stopPropagation(); });
addEventListener("keydown", (e) => { if (e.key === "Control") { document.body.classList.add("ctrl");
  hideTip(); } });
addEventListener("keyup", (e) => { if (e.key === "Control") document.body.classList.remove("ctrl"); });
addEventListener("blur", () => document.body.classList.remove("ctrl"));
addEventListener("resize", renderArcs);
$("#canvas").addEventListener("scroll", renderArcs);
addEventListener("beforeunload", (e) => { if (S.dirty) { flush(); e.preventDefault();
  e.returnValue = "Edits are still saving."; } });
addEventListener("blur", flush);

addEventListener("keydown", (ev) => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName)) return;
  const active = S.activeSid || S.hoverCard || (S.geo && S.geo.images[0] && S.geo.images[0].sid);
  const k = ev.key;
  if (k === "ArrowRight") { goto(S.idx + 1); }
  else if (k === "ArrowLeft") { goto(S.idx - 1); }
  else if (k === "u" || k === "U") { goto(nextUnreviewed()); }
  else if (k === "a" || k === "A") { if (active) setDecision(active, "accept"); }
  else if (k === "r" || k === "R") { if (active) setDecision(active, "reject"); }
  else if (k === "t" || k === "T") { if (active) { const r = ensureImage(active);
    r.train_ok = !r.train_ok; touch(active); } }
  else if (k === "d" || k === "D") { if (active) deleteImage(active); }
  else if (k === "m" || k === "M") { armMiss(); }
  else if (k === "n" || k === "N") { setNormalized(!S.norm); }
  else if (k === "Escape") { clearPending(); hideBanner(); $("#modal").classList.remove("open"); }
  else if (k === "Enter") { clearPending(); }
  else if (k === "?") { showHelp(); }
  else if (/^[1-9]$/.test(k)) {
    const reason = S.review.rejection_reasons.find((r) => r.quick_key === k);
    if (reason && active) toggleReason(active, reason.slug);
  } else return;
  ev.preventDefault();
});

function showHelp() {
  const box = $("#modalBox"); box.textContent = "";
  box.appendChild(el("h3", null, "Keys and gestures"));
  const rows = [
    ["← / →", "previous / next individual"], ["U", "next individual not marked done"],
    ["A / R", "accept / reject the active photo (press again to clear)"],
    ["T", "toggle train_ok — a rejected photo is usually still training data"],
    ["D", "delete the active photo as a duplicate (soft, undoable)"],
    ["1–9", "toggle a rejection reason on the active photo"],
    ["click a spot", "select it — it turns PURPLE; click it again to deselect"],
    ["click a second spot", "one blue connection between the two, then BOTH deselect"],
    ["click spot, M, click card", "record a MISS — the click marks where it should have been"],
    ["click an arc / the ✓✗ pill", "reject or restore that match (it is never deleted)"],
    ["✕ in the table", "remove that ONE connection"],
    ["Esc", "clear the selection"],
    ["footer buttons", "✓ accept · ✗ reject · T train · bin delete · yellow disc re-run anatomy · " +
      "purple disc re-run purple (both queued, never run here)"],
  ];
  const t = el("table");
  for (const [k, v] of rows) { const tr = el("tr");
    const a = el("td"); a.appendChild(el("kbd", null, k)); tr.appendChild(a);
    tr.appendChild(el("td", null, v)); t.appendChild(tr); }
  box.appendChild(t);
  box.appendChild(el("p", null, "Every edit is written to review.json ~300 ms later. " +
    "Nothing here spends money: re-extractions are queued for the operator to run."));
  $("#modal").classList.add("open");
}

boot().catch((e) => showBanner("failed to start: " + e.message, [["reload", () => location.reload()]]));
</script>
</body>
</html>
"""
