const SUIT = { S: "spades", H: "hearts", D: "diamonds", C: "clubs" };
document.body.insertAdjacentHTML("afterbegin", '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><symbol id="suit-S" viewBox="0 0 24 24"><path d="M12 2s-8 6.3-8 11.2C4 16 6 18 8.4 18c1.3 0 2.4-.5 3-1.2-.2 2-1 3.6-2.6 5.2h6.4c-1.6-1.6-2.4-3.2-2.6-5.2.6.7 1.7 1.2 3 1.2C18 18 20 16 20 13.2 20 8.3 12 2 12 2z"/></symbol><symbol id="suit-H" viewBox="0 0 24 24"><path d="M12 21s-7.6-4.6-9.6-9.3C.9 8.3 3 4.5 6.7 4.5c2.2 0 3.6 1.2 4.3 2.5.7-1.3 2.1-2.5 4.3-2.5 3.7 0 5.8 3.8 4.3 7.2C19.6 16.4 12 21 12 21z"/></symbol><symbol id="suit-D" viewBox="0 0 24 24"><path d="M12 2l8 10-8 10-8-10z"/></symbol><symbol id="suit-C" viewBox="0 0 24 24"><path d="M12 2.5a4 4 0 0 1 3.5 6 4 4 0 1 1-2.6 7.1c.2 2.4 1 4.2 2.7 5.9H8.4c1.7-1.7 2.5-3.5 2.7-5.9a4 4 0 1 1-2.6-7.1 4 4 0 0 1 3.5-6z"/></symbol></svg>');
const token = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = ["--signal", "--brass", "--win", "--loss"].map(token);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function cardHTML(c) {
  if (c === null) return '<div class="card back"></div>';
  if (c === "") return '<div class="card slot"></div>';
  const r = c[0] === "T" ? "10" : c[0], s = c[1];
  return `<div class="card ${s === "H" || s === "D" ? "red" : ""}"><span class="r">${r}</span><svg class="s" role="img" aria-label="${SUIT[s]}"><use href="#suit-${s}"/></svg></div>`;
}

function renderTable(el, t) {
  const seat = (i, pos) => {
    const cards = t.hole[i] ? t.hole[i].map(cardHTML).join("") : t.hole[1 - i].map(() => cardHTML(null)).join("");
    const bet = t.bets && t.bets[i] ? ` &middot; <span class="bet">bet ${t.bets[i]}</span>` : "";
    const stack = t.stacks ? `<div class="stack">${t.stacks[i]} chips${bet}</div>` : "";
    const bubble = t.bubbles && t.bubbles[i] ? `<div class="bubble">${esc(t.bubbles[i])}</div>` : "";
    const btn = t.button === i ? '<div class="btn-chip">D</div>' : "";
    const inner = pos === "top"
      ? `<div class="name">${esc(t.names[i])}</div>${stack}<div class="cards">${cards}</div>`
      : `<div class="cards">${cards}</div><div class="name">${esc(t.names[i])}</div>${stack}`;
    return `<div class="seat ${pos} ${t.active === i ? "active" : ""}">${inner}${bubble}${btn}</div>`;
  };
  const board = [0, 1, 2, 3, 4].map(k => cardHTML(t.board[k] || "")).join("");
  const top = t.top ?? 1;
  el.innerHTML = `<div class="table">${seat(top, "top")}
    <div class="center"><div class="cards">${board}</div><div class="pot">Pot ${t.pot}</div></div>
    ${seat(1 - top, "bottom")}</div>`;
}

function barsHTML(probs, chosen) {
  return '<div class="bars">' + Object.entries(probs).map(([k, p]) =>
    `<span class="${k === chosen ? "chosen" : ""}">${esc(k)}</span><div class="bar"><div style="width:${(100 * p).toFixed(1)}%"></div></div><span>${(100 * p).toFixed(0)}%</span>`
  ).join("") + "</div>";
}

function fitCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 400;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + "px";
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function drawChart(cv, chart) {
  const [ctx, W, H] = fitCanvas(cv, 190);
  const L = 52, R = 10, T = 10, B = 24;
  const names = Object.keys(chart.series);
  const pts = names.flatMap(n => chart.series[n]);
  if (!pts.length) return;
  const f = chart.log ? (v => Math.log10(Math.max(v, 1e-9))) : (v => v);
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = Math.min(...pts.map(p => f(p[1]))), y1 = Math.max(...pts.map(p => f(p[1])));
  if (!chart.log && y0 > 0) y0 = 0;
  if (!chart.log && y1 < 0) y1 = 0;
  if (x1 === x0) x1 = x0 + 1;
  if (y1 - y0 < 1e-9) { y1 += 0.5; y0 -= 0.5; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  ctx.font = `12px ${token("--font-num")}`; ctx.fillStyle = token("--smoke"); ctx.strokeStyle = token("--seam"); ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = y0 + (y1 - y0) * i / 4, py = Y(y);
    ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(W - R, py); ctx.stroke();
    const v = chart.log ? Math.pow(10, y) : y;
    const step = Math.abs(chart.log ? v - Math.pow(10, y - (y1 - y0) / 4) : (y1 - y0) / 4) || 1;
    const txt = Math.abs(v) >= 10000 ? v.toExponential(1) : v.toFixed(Math.min(4, Math.max(0, Math.ceil(-Math.log10(step)))));
    ctx.textAlign = "right"; ctx.fillText(txt, L - 6, py + 4);
  }
  if (!chart.log && y0 < 0 && y1 > 0) {
    ctx.strokeStyle = token("--seam-strong"); ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.textAlign = "center";
  ctx.fillText(Math.round(x0), L, H - 6); ctx.fillText(Math.round(x1), W - R - 10, H - 6);
  names.forEach((n, i) => {
    ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 2; ctx.beginPath();
    chart.series[n].forEach((p, k) => k ? ctx.lineTo(X(p[0]), Y(f(p[1]))) : ctx.moveTo(X(p[0]), Y(f(p[1]))));
    ctx.stroke();
    const last = chart.series[n][chart.series[n].length - 1];
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.beginPath(); ctx.arc(X(last[0]), Y(f(last[1])), 3, 0, 7); ctx.fill();
  });
}

function legendHTML(names) {
  return '<div class="legend">' + names.map((n, i) => `<span><i style="background:${COLORS[i % COLORS.length]}"></i>${esc(n)}</span>`).join("") + "</div>";
}

function renderReadouts(el, stats, heroKeys, empty) {
  const entries = Object.entries(stats || {});
  if (!entries.length) { el.innerHTML = `<div class="empty">${esc(empty)}</div>`; return; }
  const tone = v => /^\+/.test(v) ? "win" : /^-\d/.test(v) ? "loss" : "";
  const hero = entries.filter(([k]) => heroKeys.includes(k));
  const meta = entries.filter(([k]) => !heroKeys.includes(k));
  el.innerHTML = `<div class="readouts">${hero.map(([k, v]) =>
    `<div class="readout"><div class="v ${tone(String(v))}">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("")}</div>` +
    (meta.length ? `<div class="meta-line">${meta.map(([k, v]) => `<span>${esc(k)} <b>${esc(v)}</b></span>`).join("")}</div>` : "");
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
