/* Minimal dependency-free canvas charts (dark theme, DPR aware).
   LineChart: multi-series line chart with autoscale, optional logY,
   optional fixed y-range, legend, and decimation for long runs. */
"use strict";

const CHART_COLORS = ["#4da3ff", "#37c98b", "#e8b45a", "#e5604c", "#b78aff",
                      "#5ad7d0", "#ff8ac2", "#a3be5a"];

class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = Object.assign({ logY: false, yMin: null, yMax: null,
                                xLabel: "", maxPoints: 700, refLine: null }, opts);
    this.series = new Map();   // name -> {x:[], y:[], color}
    this._colorIdx = 0;
  }
  clear() { this.series.clear(); this._colorIdx = 0; this.render(); }
  push(name, x, y) {
    if (y === undefined || y === null || Number.isNaN(y)) return;
    let s = this.series.get(name);
    if (!s) {
      s = { x: [], y: [], color: CHART_COLORS[this._colorIdx++ % CHART_COLORS.length] };
      this.series.set(name, s);
    }
    s.x.push(x); s.y.push(y);
  }
  _prep() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth || 340, h = c.clientHeight || 190;
    if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }
  render() {
    const [w, h] = this._prep();
    const ctx = this.ctx, o = this.opts;
    ctx.clearRect(0, 0, w, h);
    const padL = 46, padR = 8, padT = 8, padB = 20;
    const pw = w - padL - padR, ph = h - padT - padB;
    if (pw <= 0 || ph <= 0) return;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of this.series.values()) {
      for (let i = 0; i < s.x.length; i++) {
        const yv = o.logY ? (s.y[i] > 0 ? s.y[i] : null) : s.y[i];
        if (yv === null) continue;
        if (s.x[i] < xMin) xMin = s.x[i];
        if (s.x[i] > xMax) xMax = s.x[i];
        if (yv < yMin) yMin = yv;
        if (yv > yMax) yMax = yv;
      }
    }
    if (o.refLine !== null) { yMin = Math.min(yMin, o.refLine); yMax = Math.max(yMax, o.refLine); }
    if (o.yMin !== null) yMin = o.yMin;
    if (o.yMax !== null) yMax = o.yMax;
    if (!isFinite(xMin) || !isFinite(yMin)) { this._empty(ctx, w, h); return; }
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) { yMax += Math.abs(yMax) * 0.1 + 1e-6; yMin -= Math.abs(yMin) * 0.1 + 1e-6; }
    if (!o.logY) { const m = (yMax - yMin) * 0.08; yMin -= m; yMax += m; }

    const ly = v => o.logY ? Math.log10(v) : v;
    const Y0 = ly(yMin), Y1 = ly(yMax);
    const sx = x => padL + (x - xMin) / (xMax - xMin) * pw;
    const sy = v => padT + (1 - (ly(v) - Y0) / (Y1 - Y0)) * ph;

    // grid + y ticks
    ctx.strokeStyle = "#1f2836"; ctx.fillStyle = "#7f8b9c";
    ctx.font = "10px Segoe UI"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const nTicks = 4;
    for (let i = 0; i <= nTicks; i++) {
      const vy = Y0 + (Y1 - Y0) * i / nTicks;
      const val = o.logY ? Math.pow(10, vy) : vy;
      const py = sy(val);
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
      ctx.fillText(fmtNum(val), padL - 4, py);
    }
    // x ticks
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let i = 0; i <= 4; i++) {
      const vx = xMin + (xMax - xMin) * i / 4;
      ctx.fillText(fmtNum(vx), sx(vx), h - padB + 4);
    }
    if (o.refLine !== null && o.refLine >= yMin && o.refLine <= yMax) {
      ctx.strokeStyle = "#e8b45a"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(padL, sy(o.refLine)); ctx.lineTo(w - padR, sy(o.refLine)); ctx.stroke();
      ctx.setLineDash([]);
    }
    // series
    for (const s of this.series.values()) {
      const step = Math.max(1, Math.floor(s.x.length / this.opts.maxPoints));
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i += step) {
        const yv = s.y[i];
        if (o.logY && yv <= 0) continue;
        const px = sx(s.x[i]), py = sy(yv);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // legend
    let lx = padL + 6;
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "11px Segoe UI";
    for (const [name, s] of this.series) {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, padT + 2, 8, 8);
      ctx.fillStyle = "#d7dee8";
      ctx.fillText(name, lx + 12, padT);
      lx += 20 + ctx.measureText(name).width;
      if (lx > w - 80) break;
    }
  }
  _empty(ctx, w, h) {
    ctx.fillStyle = "#3a4557"; ctx.font = "12px Segoe UI";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("no data yet", w / 2, h / 2);
  }
}

function fmtNum(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toExponential(0);
}
