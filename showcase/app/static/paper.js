"use strict";
/* Reliability-Guided Fusion — interactive paper. Ink only: experts are drawn as
   solid (image), hatched (text) and dotted (interaction), never by colour. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const EXP = ["img", "txt", "int"];
const NAME = { img: "image", txt: "text", int: "interaction" };
const nice = (s) => String(s).replace(/_/g, " ");
const f2 = (v) => Number(v).toFixed(2);
const f3 = (v) => Number(v).toFixed(3);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fillOf = (k) => (k === "img" ? 'style="fill:var(--ink)"' : `fill="url(#pat-${k})"`);
const SVG = (w, h, body, label = "") => `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${body}</svg>`;
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" class="${o.mono ? "ink-m" : "ink-t"}" font-size="${o.size || 15}" text-anchor="${o.anchor || "start"}"${o.italic ? ' font-style="italic"' : ""}${o.id ? ` id="${o.id}"` : ""}${o.attr || ""}>${s}</text>`;

/* ── theme ──────────────────────────────────────────────── */
const themeBtn = $("#theme");
const syncTheme = () => (themeBtn.textContent = document.documentElement.dataset.theme === "dark" ? "◑ light" : "◐ dark");
themeBtn.onclick = () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  try { localStorage.setItem("rgf-theme", dark ? "dark" : "light"); } catch (e) { /* private mode */ }
  syncTheme();
  window.dispatchEvent(new Event("themechange"));
};
syncTheme();

/* ── ink drawing: lines draw themselves when their layer appears ─────── */
function prepDraw(root) {
  $$(".draw", root).forEach((el) => {
    const L = el.getTotalLength ? el.getTotalLength() : 0;
    el.style.strokeDasharray = L;
    el.style.strokeDashoffset = L;
    el.dataset.len = L;
  });
}
function layer(root, name, state) { // state: "on" | "off" | "dim"
  $$(`[data-layer~="${name}"]`, root).forEach((g) => {
    g.classList.toggle("off", state === "off");
    g.classList.toggle("dim", state === "dim");
    $$(".draw", g).forEach((el) => (el.style.strokeDashoffset = state === "off" ? el.dataset.len : 0));
  });
}

/* ── scrollytelling engine ─────────────────────────────── */
const FIG = {};
function scrolly() {
  $$(".scrolly").forEach((sc) => {
    const fig = FIG[sc.dataset.fig];
    const steps = $$(".step", sc);
    let cur = -1;
    const set = (k) => {
      if (k === cur) return;
      cur = k;
      steps.forEach((s) => s.classList.toggle("on", +s.dataset.step === k));
      fig.show(k);
    };
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && set(+e.target.dataset.step)),
      { rootMargin: "-46% 0px -46% 0px" });
    steps.forEach((s) => io.observe(s));
    set(0);
  });
}

/* ════════════════════════════════════════════════════════════════
   Fig. 1 — the problem (an invented, drawn example)
   ════════════════════════════════════════════════════════════════ */
FIG.fig1 = (() => {
  const el = $("#fig1");
  const rays = Array.from({ length: 8 }, (_, i) => {
    const a = (i * Math.PI) / 4, r1 = 30, r2 = 40;
    return `<line class="ink draw" x1="${330 + r1 * Math.cos(a)}" y1="${72 + r1 * Math.sin(a)}" x2="${330 + r2 * Math.cos(a)}" y2="${72 + r2 * Math.sin(a)}"/>`;
  }).join("");
  const wave = (y, a) => `<path class="ink draw" d="M24 ${y} q 9 ${a} 18 0${" t 18 0".repeat(19)}"/>`;
  const photo = `
    <g data-layer="photo">
      <rect class="ink draw" x="20" y="16" width="370" height="246"/>
      <circle class="ink draw" cx="330" cy="72" r="22"/>${rays}
      <line class="ink draw" x1="20" y1="150" x2="390" y2="150"/>
      ${wave(170, -5)}${wave(190, -4)}${wave(212, -3)}
      <path class="ink draw" d="M70 262 C 150 236, 290 236, 390 252"/>
      <path class="ink draw" d="M120 128 Q 160 92 200 128 Z"/>
      <line class="ink draw" x1="160" y1="110" x2="174" y2="246"/>
      <path class="ink draw" d="M222 246 l 26 -34 l 40 0 M 236 228 l 44 0 l 14 18"/>
      ${T(205, 290, "a photo", { mono: true, size: 12, anchor: "middle" })}
    </g>`;
  const broken = `
    <g data-layer="broken" class="layer off">
      <rect x="20" y="16" width="370" height="246" fill="url(#pat-txt)" opacity=".45"/>
      <rect x="110" y="114" width="190" height="48" class="paper-f" style="stroke:var(--ink)"/>
      ${T(205, 134, "image failed to load", { mono: true, size: 13, anchor: "middle" })}
      ${T(205, 152, "0 bytes", { mono: true, size: 11, anchor: "middle" })}
    </g>`;
  const caption = (id, s) => `<g data-layer="${id}" class="layer off">${T(205, 324, `“${s}”`, { italic: true, size: 18, anchor: "middle" })}</g>`;
  const readings = `
    <g transform="translate(420,40)">
      ${T(0, 0, "the image says", { size: 13, mono: true })}<text id="r-img" x="0" y="26" class="ink-t" font-size="19">a holiday</text>
      ${T(0, 82, "the text says", { size: 13, mono: true })}<text id="r-txt" x="0" y="108" class="ink-t" font-size="19">–</text>
      ${T(0, 164, "together", { size: 13, mono: true })}<text id="r-all" x="0" y="190" class="ink-t" font-size="19">–</text>
      <line class="ink-soft" x1="0" y1="50" x2="170" y2="50"/><line class="ink-soft" x1="0" y1="132" x2="170" y2="132"/>
    </g>`;
  el.innerHTML = SVG(600, 340, photo + broken + caption("capA", "Great weather for my day off.") +
    caption("capB", "Can’t wait to be back at my desk on Monday.") + readings, "Illustration: a beach photo with two captions") +
    `<div style="margin-top:16px"><div class="lbl"><span>a good router would weigh (illustrative)</span><span class="key"><span><i class="sw p-img"></i>image</span><span><i class="sw p-txt"></i>text</span><span><i class="sw p-int"></i>interaction</span></span></div>
     <div class="pibar" id="f1bar"><i class="p-img"></i><i class="p-txt"></i><i class="p-int"></i></div><div class="pilab mono" id="f1lab"></div></div>`;
  prepDraw(el);
  const S = [
    { cap: null, broken: false, r: ["a holiday", "–", "–"], pi: [1, 0, 0], note: "only the image exists" },
    { cap: "capA", broken: false, r: ["a holiday", "a day off", "a holiday post"], pi: [0.34, 0.33, 0.33], note: "any weighting works" },
    { cap: "capB", broken: false, r: ["a holiday", "work on Monday", "sarcasm"], pi: [0.12, 0.13, 0.75], note: "trust the interaction" },
    { cap: "capA", broken: true, r: ["nothing", "a day off", "a holiday post, from the text"], pi: [0.04, 0.8, 0.16], note: "trust the text" },
  ];
  return {
    show(k) {
      const s = S[k];
      layer(el, "photo", "on");
      layer(el, "broken", s.broken ? "on" : "off");
      layer(el, "capA", s.cap === "capA" ? "on" : "off");
      layer(el, "capB", s.cap === "capB" ? "on" : "off");
      ["r-img", "r-txt", "r-all"].forEach((id, i) => ($("#" + id).textContent = s.r[i]));
      $$("#f1bar i").forEach((b, i) => (b.style.width = `${s.pi[i] * 100}%`));
      $("#f1lab").innerHTML = `<span>${EXP.map((e, i) => `${NAME[e]} ${f2(s.pi[i])}`).join(" · ")}</span><span>${s.note}</span>`;
    },
  };
})();

/* ════════════════════════════════════════════════════════════════
   Fig. 2 — V3's gate collapses
   ════════════════════════════════════════════════════════════════ */
FIG.fig2 = (() => {
  const el = $("#fig2");
  const hoped = [0.18, 0.82, 0.45, 0.3, 0.92, 0.62, 0.08, 0.7];
  const X = (a) => 70 + a * 460;
  const inputs = hoped.map((a, i) => {
    const y = 40 + i * 13;
    return `<g data-layer="static" class="layer off"><rect x="70" y="${y}" width="230" height="9" style="fill:var(--ink)"/><rect x="300" y="${y}" width="230" height="9" fill="url(#pat-txt)" style="stroke:var(--ink)" stroke-width=".6"/></g>`;
  }).join("");
  const knobs = hoped.map((a, i) => `<circle class="knob" data-a="${a}" cx="${X(a)}" cy="${232 + (i % 2 ? 6 : -6)}" r="6" style="fill:var(--paper);stroke:var(--ink);transition:cx 1s cubic-bezier(.6,.1,.3,1)" stroke-width="1.3"/>`).join("");
  const body = `
    <g data-layer="static" class="layer off">${T(70, 26, "eight inputs, one fixed rule: image ½ · text ½", { mono: true, size: 12 })}
      ${T(185, 162, "image", { size: 14, anchor: "middle" })}${T(415, 162, "text", { size: 14, anchor: "middle" })}</g>
    ${inputs}
    <g data-layer="gate" class="layer off">
      ${T(300, 196, "a gate α per input: 0 = all image, 1 = all text", { mono: true, size: 12, anchor: "middle" })}
      <line class="ink" x1="70" y1="232" x2="530" y2="232"/>
      <line class="ink" x1="70" y1="222" x2="70" y2="242"/><line class="ink" x1="530" y1="222" x2="530" y2="242"/>
      ${T(70, 266, "image", { size: 14, anchor: "middle" })}${T(530, 266, "text", { size: 14, anchor: "middle" })}
      <line class="ink-soft" x1="300" y1="214" x2="300" y2="250" stroke-dasharray="3 3"/>
      ${knobs}
      <g data-layer="collapse" class="layer off">${T(300, 292, "α = 0.5 ± 0.002 for every input", { mono: true, size: 13, anchor: "middle" })}</g>
    </g>
    <g data-layer="score" class="layer off">
      ${T(200, 306, "FHM AUROC", { mono: true, size: 12 })}
      <rect x="200" y="314" width="${(0.720 - 0.70) / 0.05 * 330}" height="14" fill="url(#pat-txt)" style="stroke:var(--ink)"/>
      <rect x="200" y="336" width="${(0.737 - 0.70) / 0.05 * 330}" height="14" style="fill:var(--ink)"/>
      ${T(195, 325, "learned gate", { size: 14, anchor: "end" })}${T(195, 347, "fixed α = 0.5", { size: 14, anchor: "end" })}
      ${T(200 + (0.720 - 0.70) / 0.05 * 330 + 8, 325, "0.720", { mono: true, size: 13 })}
      ${T(200 + (0.737 - 0.70) / 0.05 * 330 + 8, 347, "0.737", { mono: true, size: 13 })}
      ${T(200, 372, "axis starts at 0.70", { mono: true, size: 10.5 })}
    </g>`;
  el.innerHTML = SVG(600, 380, body, "V3 gate collapse");
  return {
    show(k) {
      layer(el, "static", k === 0 ? "on" : "off");
      layer(el, "gate", k >= 1 ? (k === 3 ? "dim" : "on") : "off");
      layer(el, "collapse", k >= 2 ? "on" : "off");
      layer(el, "score", k === 3 ? "on" : "off");
      $$(".knob", el).forEach((c) => (c.style.cx = k >= 2 ? X(0.5 + (Math.random() - 0.5) * 0.004) : X(+c.dataset.a)));
    },
  };
})();

/* ════════════════════════════════════════════════════════════════
   Fig. 3 — architecture assembles step by step
   ════════════════════════════════════════════════════════════════ */
FIG.fig3 = (() => {
  const el = $("#fig3");
  const box = (x, y, w, h, lines, o = {}) => {
    const sw = o.pat ? `<rect x="${x + 8}" y="${y + 8}" width="13" height="13" ${fillOf(o.pat)} style="stroke:var(--ink)" stroke-width="1"/>` : "";
    const ty = y + h / 2 - (lines.length - 1) * 9 + 5;
    return `<rect class="ink draw" x="${x}" y="${y}" width="${w}" height="${h}"${o.dash ? ' stroke-dasharray="5 4"' : ""}/>${sw}` +
      lines.map((s, i) => T(x + w / 2 + (o.pat ? 7 : 0), ty + i * 18, s, { size: i ? 13 : 15, anchor: "middle", italic: i > 0 && o.it })).join("");
  };
  const arrow = (d, o = {}) => `<path class="ink draw" d="${d}" marker-end="url(#arr)"${o.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
  const sub = (b, s) => `${b}<tspan font-size="11" dy="4">${s}</tspan><tspan dy="-4"></tspan>`;
  const body = `
    <g data-layer="a" class="layer off">
      <rect class="ink-soft" x="132" y="26" width="156" height="370" stroke-dasharray="4 4"/>
      ${T(210, 18, "frozen CLIP ViT-L/14@336", { mono: true, size: 11.5, anchor: "middle" })}
      ${box(16, 64, 92, 54, ["image", "the photo"], { it: true })}${box(16, 296, 92, 54, ["text", "the caption"], { it: true })}
      ${box(148, 56, 124, 70, ["vision", "encoder"])}${box(148, 288, 124, 70, ["text", "encoder"])}
      ${T(210, 205, "tokens, cached", { size: 13, anchor: "middle", italic: true })}${T(210, 222, "once", { size: 13, anchor: "middle", italic: true })}
      ${arrow("M108 91 H146")}${arrow("M108 323 H146")}
    </g>
    <g data-layer="b" class="layer off">
      ${box(330, 40, 160, 62, ["image expert", "image tokens"], { pat: "img", it: true })}
      ${box(330, 168, 160, 84, ["interaction expert", "cross-attention", "+ outer product"], { pat: "int", it: true })}
      ${box(330, 312, 160, 62, ["text expert", "word tokens"], { pat: "txt", it: true })}
      ${arrow("M272 80 H328")}${arrow("M272 104 H300 V196 H328")}${arrow("M272 312 H300 V224 H328")}${arrow("M272 336 H328")}
    </g>
    <g data-layer="c" class="layer off">
      ${box(544, 290, 100, 60, ["router", "π = softmax(·)"], { it: true })}
      <path class="ink draw" d="M210 396 V418 H594 V352" marker-end="url(#arr)"/>
      <path class="ink draw" d="M410 374 V418"/><circle cx="410" cy="418" r="3" class="ink-f"/>
      ${T(300, 436, "summaries v̄, ū · cosine s", { mono: true, size: 11.5, anchor: "middle" })}
      ${T(460, 436, "entropies Ĥ", { mono: true, size: 11.5, anchor: "middle" })}
    </g>
    <g data-layer="d" class="layer off">
      <circle class="ink draw" cx="586" cy="210" r="21"/>${T(586, 217, "Σ", { size: 20, anchor: "middle" })}
      ${arrow("M490 71 H548 L571 191")}${arrow("M490 210 H563")}${arrow("M490 343 H522 V240 L566 223")}
      ${arrow("M594 290 V233")}${T(604, 266, "π", { size: 16, italic: true })}
      ${arrow("M607 210 H646")}${T(628, 198, "p̂", { size: 17, italic: true, anchor: "middle" })}
      ${T(512, 62, sub("p", "img"), { size: 14, italic: true })}${T(510, 202, sub("p", "int"), { size: 14, italic: true })}${T(494, 334, sub("p", "txt"), { size: 14, italic: true })}
    </g>
    <g data-layer="e" class="layer off">
      ${box(330, 452, 302, 44, ["reliability target  t = softmax(−ℓ / τ)"], { dash: true })}
      ${arrow("M626 452 V352", { dash: true })}${T(618, 442, "KL(t ‖ π)", { mono: true, size: 11.5, anchor: "end" })}
      ${arrow("M62 350 V474 H328", { dash: true })}${T(70, 466, "true label y", { mono: true, size: 11.5 })}
      ${T(330, 516, "training only", { mono: true, size: 11.5 })}
    </g>`;
  el.innerHTML = SVG(650, 520, body, "Architecture") + `<div id="eqs" style="margin-top:14px"></div>`;
  prepDraw(el);
  const E = [
    "π = softmax( MLP<sub>r</sub> [ v̄ ; ū ; Ĥ<sub>img</sub>, Ĥ<sub>txt</sub>, Ĥ<sub>int</sub> ; <i>s</i> ] )",
    "<i>p̂</i> = Σ<sub><i>k</i></sub> π<sub><i>k</i></sub> <i>p</i><sub><i>k</i></sub>",
    "<i>t</i><sub><i>k</i></sub> = exp(−ℓ<sub><i>k</i></sub> / τ) ⁄ Σ<sub><i>j</i></sub> exp(−ℓ<sub><i>j</i></sub> / τ),&nbsp;&nbsp; ℓ<sub><i>k</i></sub> = −log <i>p</i><sub><i>k</i></sub>(<i>y</i>)",
    "ℒ = −log <i>p̂</i>(<i>y</i>) + λ<sub>aux</sub> · ⅓ Σ<sub><i>k</i></sub> ℓ<sub><i>k</i></sub> + λ<sub>rel</sub> · KL(<i>t</i> ‖ π)",
  ];
  const shown = [[], [], [0], [0, 1], [0, 1, 2, 3]];
  return {
    show(k) {
      ["a", "b", "c", "d", "e"].forEach((n, i) => layer(el, n, i <= k ? "on" : "off"));
      $("#eqs").innerHTML = shown[k].map((i) =>
        `<div style="display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;border-top:1px solid var(--rule-3);font-size:16.5px"><span>${E[i]}</span><span class="mono faint">(${i + 1})</span></div>`).join("");
    },
  };
})();

/* ════════════════════════════════════════════════════════════════
   Data for §5–§7
   ════════════════════════════════════════════════════════════════ */
let D = null;
const meta = (key) => D.datasets.find((d) => d.key === key);

/* ── ternary plot on canvas (ink) ── */
function triGeom(w, h) {
  const pad = 40, side = Math.min(w - 2 * pad, (h - 2 * pad - 10) / 0.866);
  const cx = w / 2, top = (h - side * 0.866) / 2 - 2;
  return { A: [cx, top], B: [cx - side / 2, top + side * 0.866], C: [cx + side / 2, top + side * 0.866] };
}
const triPos = (g, p) => [p[0] * g.A[0] + p[1] * g.B[0] + p[2] * g.C[0], p[0] * g.A[1] + p[1] * g.B[1] + p[2] * g.C[1]];
function triFrame(x, g, w) {
  const ink = cssv("--ink"), soft = cssv("--rule-3");
  x.lineWidth = 1; x.strokeStyle = soft;
  for (const t of [1 / 3, 2 / 3]) for (const [p, q, r] of [[g.A, g.B, g.C], [g.B, g.C, g.A], [g.C, g.A, g.B]]) {
    x.beginPath(); x.moveTo(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t); x.lineTo(p[0] + (r[0] - p[0]) * t, p[1] + (r[1] - p[1]) * t); x.stroke();
  }
  x.strokeStyle = ink; x.lineWidth = 1.2;
  x.beginPath(); x.moveTo(...g.A); x.lineTo(...g.B); x.lineTo(...g.C); x.closePath(); x.stroke();
  x.fillStyle = ink; x.font = "italic 15px Charter, Georgia, serif";
  x.textAlign = "center"; x.fillText("image", g.A[0], g.A[1] - 12);
  x.textAlign = "left"; x.fillText("text", Math.max(2, g.B[0] - 14), g.B[1] + 24);
  x.textAlign = "right"; x.fillText("interaction", Math.min(w - 2, g.C[0] + 14), g.C[1] + 24);
}
function triCentre(x, g) {
  const c = triPos(g, [1 / 3, 1 / 3, 1 / 3]), ink = cssv("--ink"), paper = cssv("--paper");
  x.fillStyle = paper; x.strokeStyle = ink; x.lineWidth = 1.4;
  x.beginPath(); x.arc(c[0], c[1], 6.5, 0, 7); x.fill(); x.stroke();
  x.beginPath(); x.arc(c[0], c[1], 1.6, 0, 7); x.fillStyle = ink; x.fill();
  x.font = "13px ui-monospace, Menlo, monospace"; x.textAlign = "right";
  const tw = x.measureText("static ⅓").width;
  x.fillStyle = paper; x.fillRect(c[0] - 16 - tw, c[1] - 9, tw + 8, 18);
  x.strokeStyle = cssv("--rule-2"); x.strokeRect(c[0] - 16 - tw, c[1] - 9, tw + 8, 18);
  x.fillStyle = cssv("--ink-2"); x.fillText("static ⅓", c[0] - 12, c[1] + 4);
}
function mark(x, p, kind) { // kinds: ok, bad, intact, noimg, notxt
  const ink = cssv("--ink");
  x.strokeStyle = ink; x.fillStyle = ink; x.lineWidth = 1;
  if (kind === "ok" || kind === "intact") { x.globalAlpha = kind === "ok" ? 0.55 : 0.32; x.beginPath(); x.arc(p[0], p[1], 2.1, 0, 7); x.fill(); }
  else if (kind === "bad" || kind === "noimg") { x.globalAlpha = 0.9; x.beginPath(); x.arc(p[0], p[1], kind === "bad" ? 2.6 : 3.1, 0, 7); x.stroke(); }
  else { x.globalAlpha = 0.9; const r = 3; x.beginPath(); x.moveTo(p[0] - r, p[1] - r); x.lineTo(p[0] + r, p[1] + r); x.moveTo(p[0] + r, p[1] - r); x.lineTo(p[0] - r, p[1] + r); x.stroke(); }
  x.globalAlpha = 1;
}
const kindOf = (r, mode) => (mode === "mixed" ? { intact: "intact", image_missing: "noimg", text_missing: "notxt" }[r[10]] : r[1] === r[2] ? "ok" : "bad");

/* ════════════════════════════════════════════════════════════════
   Fig. 4 — findings (ternary → morph → slope → forest)
   ════════════════════════════════════════════════════════════════ */
FIG.fig4 = (() => {
  const el = $("#fig4");
  el.innerHTML = `<div data-layer="tri" class="layer" style="position:relative;width:100%;aspect-ratio:1.08"><canvas id="f4c" style="position:absolute;inset:0;width:100%;height:100%"></canvas></div>
    <div data-layer="slope" class="layer off" style="position:absolute;inset:18px 0 14px"></div>
    <div data-layer="forest" class="layer off" style="position:absolute;inset:18px 0 14px"></div>
    <div class="key" id="f4key" style="margin-top:8px"></div>`;
  let step = -1, anim = null, pts = [];
  const CAP = [
    "<b>Fig. 4.</b> Router weights for each clean FHM test meme (seed 0 of 10). Filled dot: correct prediction; ring: wrong.",
    "<b>Fig. 4.</b> Clean Harm-C (seed 0). Weight piles up toward the image corner.",
    "<b>Fig. 4.</b> The same Harm-C memes under the pre-registered mixed-quality test (seed 0). Ring: image missing; cross: text missing; dot: intact.",
    "<b>Fig. 4.</b> Mean router weight on a modality when it is present and when it is missing, pooled over 10 seeds.",
    "<b>Fig. 4.</b> Dynamic − static, paired over 10 seeds, 95% CI. Ring: clean data; square: mixed quality. * Holm p &lt; 0.05. † Outside the Holm family.",
  ];
  const KEY = {
    clean: '<span>● correct</span><span>○ wrong</span><span class="mono">◎ static fusion (⅓ each)</span>',
    mixed: '<span>· intact</span><span>○ image missing</span><span>× text missing</span>',
  };
  function draw(points, mode) {
    const cv = $("#f4c"), w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const x = cv.getContext("2d"); x.scale(dpr, dpr);
    const g = triGeom(w, h);
    triFrame(x, g, w);
    for (const p of points) mark(x, triPos(g, p.pi), p.kind);
    triCentre(x, g);
  }
  function setPoints(ds, mode) {
    return D[mode][ds].rows.map((r) => ({ id: r[0], pi: [r[4], r[5], r[6]], kind: kindOf(r, mode) }));
  }
  function morph(from, to, ms = 1300) {
    cancelAnimationFrame(anim);
    const map = new Map(from.map((p) => [p.id, p.pi]));
    const t0 = performance.now();
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / ms), e = ease(t);
      const cur = to.map((p) => {
        const a = map.get(p.id) || p.pi;
        return { pi: a.map((v, i) => v + (p.pi[i] - v) * e), kind: t < 0.15 ? "intact" : p.kind };
      });
      draw(cur, "mixed");
      if (t < 1) anim = requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }
  function slope() {
    const G = D.router_groups, names = Object.keys(G);
    const W = 620, H = 440, top = 44, bot = 390, VM = 0.7, Y = (v) => bot - (v / VM) * (bot - top);
    const panel = (x0, key, miss, title) => {
      const xa = x0 + 20, xb = x0 + 150;
      let s = T(x0 + 85, 22, title, { size: 15, anchor: "middle" });
      s += `<line class="ink-soft" x1="${xa}" y1="${top}" x2="${xa}" y2="${bot}"/><line class="ink-soft" x1="${xb}" y1="${top}" x2="${xb}" y2="${bot}"/>`;
      s += T(xa, bot + 22, "present", { mono: true, size: 11.5, anchor: "middle" }) + T(xb, bot + 22, "missing", { mono: true, size: 11.5, anchor: "middle" });
      const ends = names.map((n) => ({ n, a: G[n].intact.pi[key], b: G[n][miss].pi[key] })).sort((p, q) => q.b - p.b);
      let lastY = -99;
      ends.forEach((e) => {
        s += `<line class="ink draw" x1="${xa}" y1="${Y(e.a)}" x2="${xb}" y2="${Y(e.b)}" stroke-width="1.4"/>`;
        s += `<circle cx="${xa}" cy="${Y(e.a)}" r="3" class="ink-f"/><circle cx="${xb}" cy="${Y(e.b)}" r="3" style="fill:var(--paper);stroke:var(--ink)"/>`;
        let ly = Y(e.b) + 4; if (ly - lastY < 13) ly = lastY + 13; lastY = ly;
        s += T(xb + 9, ly, `${e.n} <tspan class="ink-m" font-size="11">${f2(e.b)}</tspan>`, { size: 13 });
      });
      return s;
    };
    let s = "";
    for (const v of [0, 0.2, 0.4, 0.6]) s += `<line class="ink-soft" x1="34" y1="${Y(v)}" x2="${W}" y2="${Y(v)}" opacity=".5"/>` + T(28, Y(v) + 4, v, { mono: true, size: 10.5, anchor: "end" });
    s += panel(40, 0, "image missing", "weight on the image") + panel(340, 1, "text missing", "weight on the text");
    return SVG(W, H, s, "Slope chart of router weights");
  }
  function forest() {
    const S = D.significance, rows = Object.keys(S.clean);
    const W = 600, rh = 50, top = 40, x0 = 110, x1 = 470, lo = -0.03, hi = 0.03;
    const H = top + rows.length * rh + 40, X = (v) => x0 + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (x1 - x0);
    let s = "";
    for (const v of [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]) {
      s += `<line class="${v === 0 ? "ink" : "ink-soft"}" x1="${X(v)}" y1="${top - 12}" x2="${X(v)}" y2="${top + rows.length * rh}"/>`;
      s += T(X(v), top + rows.length * rh + 18, v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(2), { mono: true, size: 10.5, anchor: "middle" });
    }
    s += T((x0 + x1) / 2, H - 2, "← static better · dynamic better →", { mono: true, size: 11, anchor: "middle" });
    s += T(W, top - 18, "Holm p  clean · mixed", { mono: true, size: 10.5, anchor: "end" });
    const pv = (r) => { if (!r) return "–"; const v = r.holm === "–" ? r.p : r.holm; return v + (parseFloat(v.replace("<", "")) < 0.05 ? "*" : "") + (r.holm === "–" ? "†" : ""); };
    rows.forEach((n, i) => {
      const y = top + i * rh + rh / 2;
      s += T(x0 - 14, y + 5, n, { size: 15, anchor: "end" });
      const c = S.clean[n], m = S.mixed[n];
      s += `<line class="ink-soft" style="stroke:var(--ink-3)" x1="${X(c.ci[0])}" y1="${y - 7}" x2="${X(c.ci[1])}" y2="${y - 7}" stroke-width="1.6"/><circle cx="${X(c.delta)}" cy="${y - 7}" r="4.5" style="fill:var(--paper);stroke:var(--ink)" stroke-width="1.3"/>`;
      if (m) s += `<line class="ink" x1="${X(m.ci[0])}" y1="${y + 7}" x2="${X(m.ci[1])}" y2="${y + 7}" stroke-width="1.6"/><rect x="${X(m.delta) - 4.5}" y="${y + 2.5}" width="9" height="9" class="ink-f"/>`;
      s += T(W, y + 5, `${pv(c)} · ${pv(m)}`, { mono: true, size: 12, anchor: "end" });
    });
    return SVG(W, H, s, "Dynamic minus static, with confidence intervals");
  }
  return {
    show(k) {
      const prev = step; step = k;
      $("#cap4").innerHTML = CAP[k];
      const triOn = k <= 2;
      layer(el, "tri", triOn ? "on" : "off");
      $('[data-layer="slope"]', el).classList.toggle("off", k !== 3);
      $('[data-layer="forest"]', el).classList.toggle("off", k !== 4);
      $("#f4key").innerHTML = k <= 1 ? KEY.clean : k === 2 ? KEY.mixed : "";
      if (!D) return;
      if (k === 3) { const L = $('[data-layer="slope"]', el); if (!L.innerHTML) { L.innerHTML = slope(); prepDraw(L); } requestAnimationFrame(() => $$(".draw", L).forEach((e) => (e.style.strokeDashoffset = 0))); }
      if (k === 4) { const L = $('[data-layer="forest"]', el); if (!L.innerHTML) L.innerHTML = forest(); }
      if (k === 0) { cancelAnimationFrame(anim); pts = setPoints("fhm", "clean"); draw(pts, "clean"); }
      if (k === 1) { cancelAnimationFrame(anim); pts = setPoints("harm_c", "clean"); draw(pts, "clean"); }
      if (k === 2) { const to = setPoints("harm_c", "mixed"); prev === 1 ? morph(setPoints("harm_c", "clean"), to) : draw(to, "mixed"); pts = to; }
    },
    redraw() { const k = step; step = -1; this.show(k); },
  };
})();

/* ════════════════════════════════════════════════════════════════
   Explore every dataset (interactive ternary)
   ════════════════════════════════════════════════════════════════ */
const Explore = {
  ds: "harm_c", mode: "mixed", pts: [],
  init() {
    $("#ds").innerHTML = D.datasets.map((d) => `<button data-k="${d.key}" aria-pressed="${d.key === this.ds}">${d.label}</button>`).join("");
    $$("#ds button").forEach((b) => (b.onclick = () => { this.ds = b.dataset.k; this.sync(); }));
    $$("#mode button").forEach((b) => (b.onclick = () => { this.mode = b.dataset.m; this.sync(); }));
    this.sync();
    const cv = $("#tri"), tip = $("#tip");
    cv.onmousemove = (e) => {
      const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
      let best = null, bd = 90;
      for (const p of this.pts) { const d = (p.xy[0] - mx) ** 2 + (p.xy[1] - my) ** 2; if (d < bd) { bd = d; best = p; } }
      if (!best) return tip.classList.remove("on");
      const r = best.r, cls = meta(this.ds).classes;
      tip.innerHTML = `<div class="id">${esc(r[0])}</div>
        <div class="row"><span>label</span><span>${esc(cls[r[1]])}</span></div>
        <div class="row"><span>predicted</span><span>${esc(cls[r[2]])} · ${f2(r[3])}</span></div>
        ${r[10] !== "intact" ? `<div class="row"><span>condition</span><span>${nice(r[10])}</span></div>` : ""}
        <div class="pibar" style="height:10px;margin:8px 0 6px">${EXP.map((k, i) => `<i class="p-${k}" style="width:${r[4 + i] * 100}%"></i>`).join("")}</div>
        ${EXP.map((k, i) => `<div class="row"><span>${NAME[k]}</span><span>π ${f2(r[4 + i])} · P(true) ${f2(r[7 + i])}</span></div>`).join("")}`;
      tip.style.left = Math.min(e.clientX + 16, innerWidth - 270) + "px"; tip.style.top = e.clientY + 14 + "px";
      tip.classList.add("on");
    };
    cv.onmouseleave = () => tip.classList.remove("on");
  },
  sync() {
    if (this.mode === "mixed" && !D.mixed[this.ds]) this.ds = "fhm";
    $$("#ds button").forEach((b) => { b.setAttribute("aria-pressed", b.dataset.k === this.ds); b.disabled = this.mode === "mixed" && !D.mixed[b.dataset.k]; });
    $$("#mode button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === this.mode));
    this.draw();
  },
  draw() {
    const cv = $("#tri"), w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    if (!w) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const x = cv.getContext("2d"); x.scale(dpr, dpr);
    const g = triGeom(w, h), set = D[this.mode][this.ds];
    triFrame(x, g, w);
    this.pts = set.rows.map((r) => ({ r, xy: triPos(g, [r[4], r[5], r[6]]) }));
    for (const p of this.pts) mark(x, p.xy, kindOf(p.r, this.mode));
    triCentre(x, g);
    this.side(set);
  },
  side(set) {
    const m = meta(this.ds);
    const bar = (pi) => `<div class="pibar" style="height:16px">${EXP.map((k, i) => `<i class="p-${k}" style="width:${pi[i] * 100}%"></i>`).join("")}</div>`;
    const grp = (name, n, pi) => `<div class="gmean"><div class="gl"><span>${name}</span><span class="mono faint">n ${n.toLocaleString()} · ${pi.map(f2).join(" / ")}</span></div>${bar(pi)}</div>`;
    let h = `<div class="lbl"><span>${m.label} · ${m.task}</span></div>
      <div class="key" style="margin-bottom:6px"><span><i class="sw p-img"></i>image</span><span><i class="sw p-txt"></i>text</span><span><i class="sw p-int"></i>interaction</span></div>`;
    if (this.mode === "mixed") {
      const G = D.router_groups[m.label], s = D.significance.mixed[m.label];
      h += grp("both present · dot", G.intact.n, G.intact.pi) + grp("image missing · ring", G["image missing"].n, G["image missing"].pi) + grp("text missing · cross", G["text missing"].n, G["text missing"].pi);
      h += `<p class="cap">Bars are pooled over all 10 seeds; marks show seed 0. Mixed-quality ${s.metric}: dynamic <span class="num">${s.a.toFixed(4)}</span> vs static <span class="num">${s.b.toFixed(4)}</span>, Holm p <span class="num">${s.holm}</span>; dynamic ahead on ${s.wins} seeds.</p>`;
    } else {
      const mean = (rows) => [4, 5, 6].map((i) => rows.reduce((a, r) => a + r[i], 0) / rows.length);
      const ok = set.rows.filter((r) => r[1] === r[2]), bad = set.rows.filter((r) => r[1] !== r[2]);
      const s = D.significance.clean[m.label];
      h += grp("all test samples", set.rows.length, mean(set.rows)) + grp("correct · dot", ok.length, mean(ok)) + (bad.length ? grp("wrong · ring", bad.length, mean(bad)) : "");
      h += `<p class="cap">Seed 0 of 10${this.ds === "food101" ? `, a fixed random ${set.rows.length.toLocaleString()} of the 22,715 test pages` : ""}. Clean ${s.metric}: dynamic <span class="num">${s.a.toFixed(4)}</span> vs static <span class="num">${s.b.toFixed(4)}</span>, ${s.holm === "–" ? `p <span class="num">${s.p}</span> (outside the Holm family)` : `Holm p <span class="num">${s.holm}</span>`}.</p>`;
    }
    $("#exside").innerHTML = h;
  },
};

/* ════════════════════════════════════════════════════════════════
   §6 small multiples
   ════════════════════════════════════════════════════════════════ */
function fig5() {
  const sets = ["FHM", "MAMI", "Harm-C", "Harm-P"], c = 74, x0 = 66, y0 = 26;
  const get = (a, b) => D.transfer.find((r) => r.pair === `${a} → ${b}`)?.scores.Dynamic;
  let s = "";
  sets.forEach((b, j) => (s += T(x0 + j * c + c / 2, 16, b, { size: 12.5, anchor: "middle" })));
  sets.forEach((a, i) => {
    s += T(x0 - 8, y0 + i * c + c / 2 + 4, a, { size: 12.5, anchor: "end" });
    sets.forEach((b, j) => {
      const v = get(a, b), t = Math.max(0, Math.min(1, (v - 0.35) / 0.6));
      s += `<rect x="${x0 + j * c + 2}" y="${y0 + i * c + 2}" width="${c - 4}" height="${c - 4}" style="fill:var(--ink);stroke:var(--ink)" fill-opacity="${(0.06 + 0.84 * t).toFixed(2)}" stroke-width="${i === j ? 1.6 : 0}"/>`;
      s += `<text x="${x0 + j * c + c / 2}" y="${y0 + i * c + c / 2 + 5}" font-size="14" text-anchor="middle" style="font-family:var(--mono);fill:${t > 0.55 ? "var(--paper)" : "var(--ink)"}">${f2(v)}</text>`;
    });
  });
  s += T(x0 - 8, 14, "train ↓ test →", { mono: true, size: 10, anchor: "end" });
  $("#fig5").innerHTML = SVG(x0 + 4 * c + 4, y0 + 4 * c + 4, s, "Transfer heatmap");
}
function fig6() {
  const rows = D.ablations.filter((r) => r.delta != null && !/^τ = (0.5|1|2)$/.test(r.variant));
  const W = 360, rh = 30, x0 = 150, zero = 300, sc = 140 / 0.01;
  let s = `<line class="ink" x1="${zero}" y1="0" x2="${zero}" y2="${rows.length * rh}"/>`;
  rows.forEach((r, i) => {
    const y = i * rh + rh / 2, sig = parseFloat(r.p) < 0.05, w = Math.max(1, Math.abs(r.delta) * sc);
    const name = r.variant.replace("λ_rel = 0 (no reliability target)", "no reliability target").replace("Without ", "no ").replace("Router without ", "router: no ").replace(" expert", "");
    s += T(0, y + 4, esc(name), { size: 12.5 });
    s += `<rect x="${r.delta < 0 ? zero - w : zero}" y="${y - 6}" width="${w}" height="12" style="stroke:var(--ink);fill:${sig ? "var(--ink)" : "none"}"/>`;
    s += T(zero + 6, y + 4, `${r.delta > 0 ? "+" : ""}${r.delta.toFixed(4)}`, { mono: true, size: 10.5 });
  });
  s += T(zero - 140, rows.length * rh + 14, "−0.01", { mono: true, size: 10, anchor: "middle" }) + T(zero, rows.length * rh + 14, "0", { mono: true, size: 10, anchor: "middle" });
  $("#fig6").innerHTML = SVG(W, rows.length * rh + 18, s, "Ablations");
}
function fig7() {
  const want = [["Dynamic (ours)", "ours (head)"], ["Static fusion", "static"], ["QMF", "QMF"], ["PDF", "PDF"],
    ["CLIP ViT-L/14@336 (frozen)", "CLIP backbone"], ["Qwen2.5-VL 7B (0-shot)", "Qwen2.5-VL 7B"], ["LLaVA-1.6 7B (0-shot)", "LLaVA-1.6 7B"]];
  const c = Object.fromEntries(D.cost.map((r) => [r.method, r]));
  const W = 360, rh = 28, x0 = 104, x1 = 300, lmin = -1, lmax = 4, X = (v) => x0 + ((Math.log10(v) - lmin) / (lmax - lmin)) * (x1 - x0);
  let s = "";
  for (const t of [0.1, 1, 10, 100, 1000, 10000]) s += `<line class="ink-soft" x1="${X(t)}" y1="0" x2="${X(t)}" y2="${want.length * rh}"/>` + T(X(t), want.length * rh + 14, t >= 1000 ? `${t / 1000}s` : `${t}ms`, { mono: true, size: 9.5, anchor: "middle" });
  want.forEach(([k, n], i) => {
    const r = c[k], y = i * rh + rh / 2, ours = /ours/.test(k);
    s += T(0, y + 4, n, { size: 12.5 }) + `<rect x="${x0}" y="${y - 6}" width="${X(r.ms_b1) - x0}" height="12" style="stroke:var(--ink);fill:${ours ? "var(--ink)" : "none"}"/>`;
    s += T(X(r.ms_b1) + 5, y + 4, r.ms_b1 >= 100 ? Math.round(r.ms_b1).toLocaleString() : r.ms_b1.toFixed(1), { mono: true, size: 10 });
  });
  $("#fig7").innerHTML = SVG(W, want.length * rh + 18, s, "Latency");
}

/* ════════════════════════════════════════════════════════════════
   §7 table
   ════════════════════════════════════════════════════════════════ */
function table() {
  const L = D.leaderboard, cols = L.datasets;
  const best = cols.map((_, j) => Math.max(...L.rows.map((r) => r.scores[j]?.mean ?? -1)));
  const breaks = new Set(["Hate-CLIPper Align", "QMF", "Dynamic (ours)"]);
  const head = `<thead><tr><th>Method</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
  const body = L.rows.map((r) => {
    const cls = [breaks.has(r.method) ? "grp" : "", /ours/.test(r.method) ? "ours" : ""].join(" ");
    return `<tr class="${cls}"><td>${esc(r.method)}</td>${r.scores.map((s, j) => !s ? `<td class="faint">–</td>` :
      `<td class="${s.mean === best[j] ? "best" : ""}"><span class="num">${s.mean.toFixed(3)}</span>${s.std != null ? `<small>±${s.std.toFixed(3)}</small>` : "<small>¹</small>"}</td>`).join("")}</tr>`;
  }).join("");
  $("#res").innerHTML = `<caption><b>Table 1.</b> Test AUROC (accuracy for MVSA and Food-101), mean ± std over seeds. Bold: best in column. ¹ Single run (Hate-CLIPper with its published settings; zero-shot LMMs are deterministic).</caption>${head}<tbody>${body}</tbody>`;
}

/* ════════════════════════════════════════════════════════════════
   §4 the instrument (live model)
   ════════════════════════════════════════════════════════════════ */
const Demo = (() => {
  const TRIES = [["pizza", "Homemade margherita pizza with fresh basil and mozzarella"], ["ramen", "Spicy tonkotsu ramen with a soft-boiled egg"],
    ["chocolate cake", "Warm chocolate lava cake with vanilla ice cream"], ["Caesar salad", "Classic Caesar salad with parmesan and croutons"], ["french fries", "Crispy golden french fries with ketchup"]];
  let ready = false, image = null, res = null, active = 0;
  const text = $("#text");
  $("#tries").innerHTML = "Try: " + TRIES.map(([k], i) => `<button data-i="${i}">${k}</button>`).join(" · ");
  $$("#tries button").forEach((b) => (b.onclick = () => { text.value = TRIES[+b.dataset.i][1]; }));
  const btn = $("#go");
  const update = () => (btn.disabled = !(ready && image));

  async function poll() {
    try {
      const s = await (await fetch("api/status")).json();
      if (s.status === "ready") {
        ready = true; update();
        const dev = { mps: "Apple GPU (mps)", cuda: "CUDA GPU", cpu: "CPU" }[s.device] || s.device;
        $("#status").innerHTML = `model <b>ready</b> · ${dev} · frozen CLIP ViT-L/14@336 + fusion head · food101/dynamic seed 0 · loaded in ${s.load_seconds}s`;
        return;
      }
      if (s.status === "disabled") { $("#status").textContent = "The live model is switched off (explorer-only mode). Start the app without --explorer-only to use it."; return; }
      if (s.status === "error") { $("#status").textContent = "The model failed to load."; $("#err").hidden = false; $("#err").textContent = s.error; return; }
      $("#status").textContent = "Loading CLIP ViT-L/14@336 and the fusion head…";
    } catch { $("#status").textContent = "Server not reachable. Start it with: python -m showcase.app"; }
    setTimeout(poll, 1200);
  }

  function load(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const fr = new FileReader();
    fr.onload = () => {
      const img = new Image();
      img.onload = () => {
        const k = Math.min(1, 1280 / Math.max(img.width, img.height));
        const cv = document.createElement("canvas");
        cv.width = Math.round(img.width * k); cv.height = Math.round(img.height * k);
        cv.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height);
        image = cv.toDataURL("image/jpeg", 0.92);
        $("#preview").src = image; $("#preview").hidden = false; $("#swap").hidden = false; $("#ph").hidden = true;
        $("#imgname").textContent = `${file.name.slice(0, 28)} · ${img.width}×${img.height}`;
        update();
      };
      img.src = fr.result;
    };
    fr.readAsDataURL(file);
  }
  const ph = $("#photo");
  ph.onclick = () => $("#file").click();
  ph.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && $("#file").click();
  $("#file").onchange = (e) => load(e.target.files[0]);
  ["dragenter", "dragover"].forEach((t) => ph.addEventListener(t, (e) => { e.preventDefault(); ph.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => ph.addEventListener(t, (e) => { e.preventDefault(); ph.classList.remove("over"); }));
  ph.addEventListener("drop", (e) => load(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) load(it.getAsFile()); });

  btn.onclick = async () => {
    btn.disabled = true; $("#golbl").textContent = "Running…"; $("#err").hidden = true;
    try {
      const r = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image, text: text.value }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      res = j; active = 0; render();
    } catch (e) { $("#err").hidden = false; $("#err").textContent = `The model returned an error: ${e.message}`; }
    finally { $("#golbl").textContent = "Run the model"; update(); }
  };

  const COND = { original: "original input", blank_image: "image removed", blank_text: "text removed", mismatch: "mismatched text" };
  function shift(c, base) {
    const k = { blank_image: 0, blank_text: 1, mismatch: 1 }[c.key];
    if (k === undefined) return "";
    const a = base.pi[k], b = c.pi[k], away = b < a;
    return `π<sub>${EXP[k]}</sub> ${f2(a)} → ${f2(b)} ${away ? "✓ moved away" : "✗ leaned in"}`;
  }
  function render() {
    $("#blank").hidden = true; $("#readout").hidden = false;
    $("#ms").textContent = `${res.ms} ms for 4 inputs`;
    const base = res.conditions[0];
    $("#stress").innerHTML = `<tr><th>input</th><th>router weights</th><th>prediction</th><th style="text-align:right">shift</th></tr>` +
      res.conditions.map((c, i) => `<tr class="r ${i === active ? "on" : ""}" data-i="${i}"><td class="c">${COND[c.key]}<small>${esc(c.detail)}</small></td>
        <td><div class="mini">${EXP.map((k, j) => `<i class="p-${k}" style="width:${c.pi[j] * 100}%"></i>`).join("")}</div></td>
        <td class="pr">${esc(nice(c.dynamic[0].label))} <span class="mono">${f2(c.dynamic[0].p)}</span></td><td class="d">${shift(c, base)}</td></tr>`).join("");
    $$("#stress tr.r").forEach((tr) => (tr.onclick = () => { active = +tr.dataset.i; render(); }));
    const c = res.conditions[active];
    $("#condname").textContent = COND[c.key];
    $("#what").textContent = nice(c.dynamic[0].label);
    $("#pv").textContent = f3(c.dynamic[0].p);
    const st = c.static[0], agree = st.label === c.dynamic[0].label;
    $("#vs").innerHTML = `Static fusion (⅓ each) says <b>${esc(nice(st.label))}</b> <span class="mono">${f3(st.p)}</span> · ${agree ? "agrees" : "disagrees"}`;
    $("#top5").innerHTML = c.dynamic.map((d) => `<tr><td class="n">${esc(nice(d.label))}</td><td><div class="track"><i style="width:0"></i></div></td><td class="v">${f3(d.p)}</td></tr>`).join("");
    void $("#top5").offsetWidth;
    $$("#top5 .track i").forEach((t, i) => (t.style.width = `${c.dynamic[i].p * 100}%`));
    $$("#pibar i").forEach((b, i) => (b.style.width = `${c.pi[i] * 100}%`));
    $("#pilab").innerHTML = EXP.map((k, i) => `<span>${NAME[k]} ${f3(c.pi[i])}</span>`).join("");
    const order = EXP.map((k, i) => [k, c.pi[i]]).sort((a, b) => b[1] - a[1]);
    const unc = c.experts.map((e) => e.entropy), worst = EXP[unc.indexOf(Math.max(...unc))];
    $("#say").innerHTML = order[0][1] < 0.4
      ? `No expert dominates; the weights stay close to static fusion.`
      : `The router leans on the ${NAME[order[0][0]]} expert (${f2(order[0][1])} against the static ⅓).` +
        (worst !== order[0][0] ? ` The ${NAME[worst]} expert is the least sure (entropy ${f2(Math.max(...unc))}).` : "");
    $("#experts").innerHTML = `<tr><th>expert</th><th>top guess</th><th>p</th><th>uncertainty</th><th style="text-align:right">π</th></tr>` +
      c.experts.map((e, i) => `<tr><td><i class="sw p-${e.name}"></i>${NAME[e.name]}</td><td class="g">${esc(nice(e.top[0].label))}</td><td class="mono">${f3(e.top[0].p)}</td>
        <td><span class="unc"><i style="width:${e.entropy * 100}%"></i></span><span class="mono">${f2(e.entropy)}</span></td><td class="mono" style="text-align:right">${f3(c.pi[i])}</td></tr>`).join("");
  }
  poll();
  return {};
})();

/* ── boot ─────────────────────────────────────────────── */
scrolly();
fetch("data/explorer.json").then((r) => r.json()).then((data) => {
  D = data;
  FIG.fig4.redraw();
  Explore.init(); fig5(); fig6(); fig7(); table();
});
let rz;
window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { FIG.fig4.redraw(); D && Explore.draw(); }, 120); });
window.addEventListener("themechange", () => { FIG.fig4.redraw(); D && Explore.draw(); });

/* Dev-only: ?shot=<selector>,<offset px> scrolls there after load (used for screenshots). */
(() => {
  const q = new URLSearchParams(location.search).get("shot");
  if (!q) return;
  const [sel, dy = "0"] = q.split(",");
  setTimeout(() => {
    document.documentElement.style.scrollBehavior = "auto";
    const el = sel && sel !== "top" ? document.querySelector(sel) : null;
    scrollTo(0, (el ? el.getBoundingClientRect().top + scrollY : 0) + +dy);
  }, 700);
})();
