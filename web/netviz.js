const rgbToken = name => token(name).split(",").map(Number);
const POS = rgbToken("--signal-rgb"), NEG = rgbToken("--heat-rgb"), LIT = rgbToken("--brass-rgb");
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const STREET_NAMES = ["preflop", "flop", "turn", "river"];

const SLIDERS = [
  ["eq", "hand equity", 0, 1, 0.01],
  ["pot", "pot size", 0, 1, 0.01],
  ["odds", "pot odds", 0, 0.6, 0.01],
  ["my", "my stack", 0, 0.5, 0.01],
  ["opp", "opp stack", 0, 0.5, 0.01],
  ["raises", "raises this street", 0, 3, 1],
  ["oppr", "opp raises (hand)", 0, 4, 1],
  ["myr", "my raises (hand)", 0, 4, 1],
];

function featuresToX(f) {
  const x = new Array(21).fill(0);
  x[f.street] = 1;
  x[4] = f.eq;
  x[5 + Math.min(Math.floor(f.eq * 8), 7)] = 1;
  [f.pot, f.odds, f.my, f.opp, f.inpos ? 1 : 0, f.raises / 3, f.oppr / 4, f.myr / 4].forEach((v, i) => { x[13 + i] = v; });
  return x;
}

function xToFeatures(x) {
  return { street: x.slice(0, 4).indexOf(1), eq: x[4], pot: x[13], odds: x[14], my: x[15], opp: x[16],
           inpos: x[17] > 0.5, raises: Math.round(x[18] * 3), oppr: Math.round(x[19] * 4), myr: Math.round(x[20] * 4) };
}

class NetViz {
  constructor(root, height = 640) {
    this.root = root;
    this.height = height;
    root.classList.add("nv");
    root.innerHTML = `
      <div class="row nv-top">
        <select class="nv-mode">
          <option value="signal">show: signal flowing for this situation</option>
          <option value="weights">show: raw weights</option>
          <option value="change">show: what the last training steps changed</option>
        </select>
        <label class="muted"><input type="checkbox" class="nv-flow" checked> animate</label>
        <span class="muted nv-hint">hover a neuron to trace it · click to pin</span>
      </div>
      <div class="nv-presets"></div>
      <div class="nv-wrap"><canvas class="nv-canvas"></canvas><div class="nv-tip hidden"></div></div>
      <div class="nv-bottom">
        <div><h3>Build a situation</h3><div class="nv-builder"></div></div>
        <div><h3 class="nv-outtitle">What the network says</h3><div class="nv-out"></div></div>
      </div>`;
    this.cv = root.querySelector(".nv-canvas");
    this.tip = root.querySelector(".nv-tip");
    this.base = document.createElement("canvas");
    this.mode = "signal";
    this.f = null;
    this.particles = [];
    this.focus = null;
    this.pinned = null;
    this.dirty = true;
    root.querySelector(".nv-mode").onchange = e => { this.mode = e.target.value; this.dirty = true; };
    root.querySelector(".nv-flow").onchange = e => { this.flow = e.target.checked; this.particles = []; };
    this.flow = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    root.querySelector(".nv-flow").checked = this.flow;
    this.visible = true;
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }).observe(this.cv);
    this.cv.addEventListener("mousemove", e => this.onMove(e));
    this.cv.addEventListener("mouseleave", () => { this.focus = this.pinned; this.showTip(null); });
    this.cv.addEventListener("click", () => {
      this.pinned = this.focus && (!this.pinned || this.pinned.l !== this.focus.l || this.pinned.i !== this.focus.i) ? this.focus : null;
    });
    new ResizeObserver(() => { this.layout(); }).observe(this.cv);
    const loop = () => { if (this.visible) this.frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  setNet(net) {
    const same = this.net && this.net.name.split(" - ")[0] === net.name.split(" - ")[0];
    this.prev = same ? this.net.weights : null;
    if (!same) this.particles = [];
    const fresh = !this.net || this.net.probes.length !== net.probes.length;
    this.net = net;
    if (!this.f) this.f = xToFeatures(net.probes[0].x);
    if (fresh) this.buildControls();
    this.layout();
    this.recompute();
  }

  buildControls() {
    const p = this.root.querySelector(".nv-presets");
    p.innerHTML = this.net.probes.map((pr, i) => `<button data-i="${i}">${esc(pr.label)}</button>`).join("");
    p.querySelectorAll("button").forEach(b => b.onclick = () => {
      this.f = xToFeatures(this.net.probes[+b.dataset.i].x);
      this.syncControls();
      this.recompute();
    });
    const b = this.root.querySelector(".nv-builder");
    b.innerHTML = `<label>street</label><select data-k="street">${STREET_NAMES.map((s, i) => `<option value="${i}">${s}</option>`).join("")}</select><span></span>
      <label>in position</label><input type="checkbox" data-k="inpos"><span></span>` +
      SLIDERS.map(([k, label, lo, hi, step]) => `<label>${label}</label><input type="range" data-k="${k}" min="${lo}" max="${hi}" step="${step}"><span data-v="${k}"></span>`).join("");
    b.querySelectorAll("[data-k]").forEach(el => el.oninput = () => {
      const k = el.dataset.k;
      this.f[k] = el.type === "checkbox" ? el.checked : +el.value;
      this.syncControls();
      this.recompute();
    });
    this.syncControls();
  }

  syncControls() {
    const b = this.root.querySelector(".nv-builder");
    b.querySelectorAll("[data-k]").forEach(el => {
      const v = this.f[el.dataset.k];
      if (el.type === "checkbox") el.checked = v; else el.value = v;
    });
    b.querySelectorAll("[data-v]").forEach(el => {
      const v = this.f[el.dataset.v];
      el.textContent = Number.isInteger(v) ? v : v.toFixed(2);
    });
  }

  forward(x) {
    const acts = [x];
    this.net.weights.forEach((W, l) => {
      const a = acts[l], last = l === this.net.weights.length - 1;
      const z = this.net.biases[l].map((b, j) => a.reduce((s, ai, i) => s + ai * W[i][j], b));
      acts.push(last ? z : z.map(v => Math.max(0, v)));
    });
    return acts;
  }

  strategy(out) {
    const legal = out.map((_, i) => !(i === 0 && this.f.odds === 0));
    const pos = out.map((v, i) => legal[i] ? Math.max(0, v) : 0);
    const s = pos.reduce((a, b) => a + b, 0);
    if (s > 0) return pos.map(v => v / s);
    if (this.net.kind === "advantage") {
      let best = -1;
      out.forEach((v, i) => { if (legal[i] && (best < 0 || v > out[best])) best = i; });
      return out.map((_, i) => (i === best ? 1 : 0));
    }
    const n = legal.filter(Boolean).length;
    return legal.map(l => (l ? 1 / n : 0));
  }

  recompute() {
    this.acts = this.forward(featuresToX(this.f));
    this.probs = this.strategy(this.acts[this.acts.length - 1]);
    const edges = [];
    this.net.weights.forEach((W, l) => W.forEach((row, i) => {
      const a = this.acts[l][i];
      if (a) row.forEach((w, j) => edges.push([Math.abs(a * w), l, i, j, a * w]));
    }));
    edges.sort((a, b) => b[0] - a[0]);
    this.hot = edges.slice(0, 600);
    this.hotTotal = this.hot.reduce((s, e) => s + e[0], 0);
    this.dirty = true;
    const out = this.root.querySelector(".nv-out");
    const labels = this.net.output_labels;
    const raw = this.acts[this.acts.length - 1];
    this.root.querySelector(".nv-outtitle").textContent =
      this.net.kind === "advantage" ? "Predicted regret per action → strategy" : "Average strategy (what the bot plays)";
    out.innerHTML = barsHTML(Object.fromEntries(labels.map((l, i) => [l, this.probs[i]])), labels[this.probs.indexOf(Math.max(...this.probs))]) +
      `<p class="info">raw outputs: ${raw.map((v, i) => `${esc(labels[i])} ${v.toFixed(3)}`).join(" · ")}</p>`;
  }

  layout() {
    if (!this.net) return;
    const dpr = window.devicePixelRatio || 1;
    const W = this.cv.clientWidth || 800, H = this.height;
    for (const c of [this.cv, this.base]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    this.cv.style.height = H + "px";
    this.W = W; this.H = H; this.dpr = dpr;
    const sizes = this.net.sizes, L = 150, R = 170, T = 30, B = 14;
    this.xs = sizes.map((_, l) => L + l * (W - L - R) / (sizes.length - 1));
    this.ys = sizes.map(s => Array.from({ length: s }, (_, i) => T + (i + 0.5) * (H - T - B) / s));
    this.rad = sizes.map(s => Math.max(2.2, Math.min(7, (H - T - B) / s * 0.36)));
    this.dirty = true;
  }

  edgeValue(l, i, j) {
    const w = this.net.weights[l][i][j];
    if (this.mode === "signal") return this.acts[l][i] * w;
    if (this.mode === "change" && this.prev && this.prev[l]) return w - this.prev[l][i][j];
    return w;
  }

  drawBase() {
    const ctx = this.base.getContext("2d");
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.W, this.H);
    this.net.weights.forEach((W, l) => {
      let mx = 1e-9;
      W.forEach((row, i) => row.forEach((_, j) => { mx = Math.max(mx, Math.abs(this.edgeValue(l, i, j))); }));
      for (const strong of [false, true]) {
        W.forEach((row, i) => row.forEach((_, j) => {
          const v = this.edgeValue(l, i, j), a = Math.abs(v) / mx;
          if (a < 0.03 || (a >= 0.4) !== strong) return;
          ctx.strokeStyle = rgba(v > 0 ? POS : NEG, 0.08 + a * 0.75);
          ctx.lineWidth = 0.35 + a * 1.6;
          ctx.beginPath();
          ctx.moveTo(this.xs[l], this.ys[l][i]);
          ctx.lineTo(this.xs[l + 1], this.ys[l + 1][j]);
          ctx.stroke();
        }));
      }
    });
    this.dirty = false;
  }

  spawn() {
    if (!this.hot.length || this.particles.length > 260) return;
    for (let k = 0; k < 5; k++) {
      let r = Math.random() * this.hotTotal, e = this.hot[0];
      for (const h of this.hot) { r -= h[0]; if (r <= 0) { e = h; break; } }
      this.particles.push({ l: e[1], i: e[2], j: e[3], pos: e[4] > 0, t: 0, v: 0.012 + Math.random() * 0.01 });
    }
  }

  frame() {
    if (!this.net || !this.W) return;
    if (this.dirty) this.drawBase();
    const ctx = this.cv.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    ctx.drawImage(this.base, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    if (this.flow && this.mode !== "change") {
      this.spawn();
      this.particles = this.particles.filter(p => (p.t += p.v) < 1);
      for (const p of this.particles) {
        const x = this.xs[p.l] + (this.xs[p.l + 1] - this.xs[p.l]) * p.t;
        const y = this.ys[p.l][p.i] + (this.ys[p.l + 1][p.j] - this.ys[p.l][p.i]) * p.t;
        ctx.fillStyle = rgba(p.pos ? [170, 230, 255] : [255, 190, 160], 0.9 * Math.sin(Math.PI * p.t));
        ctx.beginPath(); ctx.arc(x, y, 1.8, 0, 7); ctx.fill();
      }
    }

    const f = this.focus || this.pinned;
    if (f) {
      ctx.fillStyle = "rgba(10,15,13,0.74)";
      ctx.fillRect(0, 0, this.W, this.H);
      const links = [];
      if (f.l > 0) this.net.weights[f.l - 1].forEach((row, i) => links.push([f.l - 1, i, f.i, this.acts[f.l - 1][i] * row[f.i]]));
      if (f.l < this.net.sizes.length - 1) this.net.weights[f.l][f.i].forEach((w, j) => links.push([f.l, f.i, j, this.acts[f.l][f.i] * w || w * 0.15]));
      const mx = Math.max(1e-9, ...links.map(k => Math.abs(k[3])));
      for (const [l, i, j, v] of links) {
        const a = Math.abs(v) / mx;
        ctx.strokeStyle = rgba(v > 0 ? POS : NEG, 0.15 + 0.85 * a);
        ctx.lineWidth = 0.6 + 2.6 * a;
        ctx.beginPath(); ctx.moveTo(this.xs[l], this.ys[l][i]); ctx.lineTo(this.xs[l + 1], this.ys[l + 1][j]); ctx.stroke();
      }
    }

    ctx.font = `12px ${token("--font-ui")}`;
    const last = this.net.sizes.length - 1;
    this.net.sizes.forEach((s, l) => {
      const a = this.acts[l], mx = Math.max(1e-9, ...a.map(Math.abs)), r = this.rad[l];
      for (let i = 0; i < s; i++) {
        const v = l === last ? this.probs[i] : a[i] / mx;
        const on = l === last ? v : Math.max(0, v);
        if (on > 0.05) {
          ctx.fillStyle = rgba(LIT, 0.18 * on);
          ctx.beginPath(); ctx.arc(this.xs[l], this.ys[l][i], r * 2.4, 0, 7); ctx.fill();
        }
        ctx.beginPath(); ctx.arc(this.xs[l], this.ys[l][i], r, 0, 7);
        ctx.fillStyle = on > 0 ? rgba(LIT, 0.15 + 0.85 * on) : "#1a221e";
        ctx.fill();
        const isF = f && f.l === l && f.i === i;
        ctx.strokeStyle = isF ? token("--ivory") : "#3e4842";
        ctx.lineWidth = isF ? 2 : 0.8;
        ctx.stroke();
        if (l === 0) {
          ctx.fillStyle = a[i] ? token("--chalk") : token("--ash"); ctx.textAlign = "right";
          ctx.fillText(`${this.net.input_labels[i]}  ${a[i].toFixed(2)}`, this.xs[0] - r - 6, this.ys[0][i] + 4);
        } else if (l === last) {
          ctx.fillStyle = token("--ivory"); ctx.textAlign = "left";
          ctx.fillText(`${this.net.output_labels[i]}  ${(100 * this.probs[i]).toFixed(0)}%`, this.xs[l] + r + 8, this.ys[l][i] + 4);
        }
      }
    });
    ctx.fillStyle = token("--smoke"); ctx.textAlign = "center";
    this.net.sizes.forEach((s, l) => ctx.fillText(l === 0 ? `inputs (${s})` : l === last ? `outputs (${s})` : `hidden ${l} (${s})`, this.xs[l], 14));
  }

  onMove(e) {
    if (!this.net) return;
    const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 1e9;
    this.xs.forEach((x, l) => {
      if (Math.abs(mx - x) > 14) return;
      this.ys[l].forEach((y, i) => { const d = Math.hypot(mx - x, my - y); if (d < bd) { bd = d; best = { l, i }; } });
    });
    this.focus = best && bd < this.rad[best.l] + 6 ? best : this.pinned;
    this.showTip(this.focus === this.pinned && !best ? null : this.focus, mx, my);
  }

  showTip(n, mx, my) {
    if (!n) { this.tip.classList.add("hidden"); return; }
    const { l, i } = n, last = this.net.sizes.length - 1, a = this.acts[l][i];
    const name = (ll, k) => ll === 0 ? this.net.input_labels[k] : `hidden ${ll} · #${k}`;
    let html;
    if (l === 0) {
      html = `<b>input: ${esc(this.net.input_labels[i])}</b><br>value ${a.toFixed(3)}`;
    } else {
      const W = this.net.weights[l - 1];
      const contrib = this.acts[l - 1].map((ai, k) => [k, ai * W[k][i]]).filter(c => c[1]).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 4);
      html = (l === last ? `<b>output: ${esc(this.net.output_labels[i])}</b><br>raw ${a.toFixed(3)} → plays it ${(100 * this.probs[i]).toFixed(1)}%`
                         : `<b>hidden layer ${l} · neuron ${i}</b><br>activation ${a.toFixed(3)}${a === 0 ? " <span class='muted'>(ReLU off)</span>" : ""}`) +
        `<br>bias ${this.net.biases[l - 1][i].toFixed(3)}<br><span class="muted">biggest pushes:</span><br>` +
        contrib.map(([k, v]) => `<span style="color:rgb(${v > 0 ? POS : NEG})">${v > 0 ? "+" : ""}${v.toFixed(3)}</span> from ${esc(name(l - 1, k))}`).join("<br>");
    }
    this.tip.innerHTML = html;
    this.tip.classList.remove("hidden");
    const left = mx + 16 + 260 > this.W ? mx - 276 : mx + 16;
    this.tip.style.left = left + "px";
    this.tip.style.top = Math.min(my + 12, this.H - 140) + "px";
  }
}
