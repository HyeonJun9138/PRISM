/* PRISM dataGen - Canvas 2D world renderer.
 *
 * Frame convention (identical to the Python side):
 *   local tangent plane, +X = East [m], +Y = North [m],
 *   psi = heading [rad] CCW from +X, wrapped to (-pi, pi].
 *
 * Screen mapping (canvas y grows downward), with `rot` = view rotation [rad]:
 *   rx =  (x-cx)*cos(rot) + (y-cy)*sin(rot)
 *   ry = -(x-cx)*sin(rot) + (y-cy)*cos(rot)
 *   sx = W/2 + rx*scale        sy = H/2 - ry*scale
 * A world direction theta therefore appears at canvas angle (rot - theta).
 * Everything is drawn in screen space so that text never ends up mirrored or
 * upside down; only glyph orientations use the angle mapping above.
 */
'use strict';

/* ----------------------------------------------------------------- utils */

/** Read the dark-theme palette out of the stylesheet so canvas and chrome
 *  can never drift apart. @returns {Object<string,string>} */
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n, fb) => (cs.getPropertyValue(n) || fb).trim() || fb;
  return {
    bg: g('--bg', '#0a0d12'),
    grid: g('--grid', '#17202c'),
    gridMajor: g('--grid-major', '#1f2b3a'),
    fg: g('--fg', '#dbe4ef'),
    dim: g('--fg-dim', '#8fa1b8'),
    faint: g('--fg-faint', '#5d6d82'),
    line: g('--line', '#263243'),
    accent: g('--accent', '#4fc3f7'),
    own: g('--own', '#ffffff'),
    nmac: g('--nmac', '#ef5350'),
    dwc: g('--dwc', '#e8a33d'),
    ok: g('--ok', '#66bb6a'),
    warn: g('--warn', '#e8c547'),
    bad: g('--bad', '#ef5350'),
    fan: g('--fan', '#7fd4ff'),
    panel: g('--panel', '#111720')
  };
}

/** Wrap an angle to (-pi, pi]. @param {number} a rad @returns {number} rad */
function wrapAngle(a) {
  let x = a % (2 * Math.PI);
  if (x <= -Math.PI) x += 2 * Math.PI;
  if (x > Math.PI) x -= 2 * Math.PI;
  return x;
}

/** Format a metre value compactly. @param {number} m @returns {string} */
function fmtM(m) {
  if (m === null || m === undefined || !isFinite(m)) return '--';
  const a = Math.abs(m);
  if (a >= 10000) return (m / 1000).toFixed(1) + ' km';
  if (a >= 1000) return (m / 1000).toFixed(2) + ' km';
  if (a >= 100) return m.toFixed(0) + ' m';
  return m.toFixed(1) + ' m';
}

/** Radians to degrees, 1 decimal. @param {number} r @returns {string} */
function fmtDeg(r) {
  if (r === null || r === undefined || !isFinite(r)) return '--';
  return (r * 180 / Math.PI).toFixed(1) + '°';
}

/** Nice 1-2-5 step at or below `x`. @param {number} x @returns {number} */
function niceStep(x) {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-9))));
  const f = x / e;
  const m = f >= 5 ? 5 : (f >= 2 ? 2 : 1);
  return m * e;
}

/** Liang-Barsky clip of a segment to a rect.
 * @param {number} x1 @param {number} y1 @param {number} x2 @param {number} y2
 * @param {number} w @param {number} h
 * @returns {?number[]} [x1,y1,x2,y2] clipped, or null if fully outside. */
function clipSeg(x1, y1, x2, y2, w, h) {
  let t0 = 0, t1 = 1;
  const dx = x2 - x1, dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1 - 0, w - x1, y1 - 0, h - y1];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i] < 0) return null; continue; }
    const r = q[i] / p[i];
    if (p[i] < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
    else { if (r < t0) return null; if (r < t1) t1 = r; }
  }
  return [x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy];
}

/** 2x2 symmetric eigen-decomposition.
 * @param {number} a R[0][0] @param {number} b R[0][1] @param {number} d R[1][1]
 * @returns {number[]} [semiMajor, semiMinor, angle_rad] for the 1-sigma ellipse. */
function eig2(a, b, d) {
  const tr = a + d, det = a * d - b * b;
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l1 = tr / 2 + disc;
  const l2 = Math.max(tr / 2 - disc, 1e-12);
  let ang;
  if (Math.abs(b) > 1e-14) ang = Math.atan2(l1 - a, b);
  else ang = (a >= d) ? 0 : Math.PI / 2;
  return [Math.sqrt(Math.max(l1, 1e-12)), Math.sqrt(l2), ang];
}

/** Constant-velocity CPA of an intruder relative to the ownship.
 * @param {Object} own frame ownship record
 * @param {Object} ag  frame agent record
 * @returns {{range:number, bearing:number, relBearing:number,
 *            cpa:?number, tcpa:?number}} distances [m], angles [rad], time [s]. */
function cpaOf(own, ag) {
  if (!own || !ag || ag.x === null || ag.x === undefined) {
    return { range: NaN, bearing: NaN, relBearing: NaN, cpa: null, tcpa: null };
  }
  const px = ag.x - own.x, py = ag.y - own.y;
  const range = Math.hypot(px, py);
  const bearing = Math.atan2(py, px);
  const relBearing = wrapAngle(bearing - (own.psi || 0));
  const vx = (ag.vx || 0) - (own.vx || 0), vy = (ag.vy || 0) - (own.vy || 0);
  const vv = vx * vx + vy * vy;
  if (vv <= 1e-9) return { range, bearing, relBearing, cpa: range, tcpa: null };
  const tcpa = -(px * vx + py * vy) / vv;
  const tc = Math.max(tcpa, 0);
  return {
    range, bearing, relBearing,
    cpa: Math.hypot(px + vx * tc, py + vy * tc), tcpa
  };
}

/* --------------------------------------------------------------- renderer */

class Renderer {
  /** @param {HTMLCanvasElement} canvas target canvas. */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.pal = readPalette();
    this.W = 800; this.H = 600; this.dpr = 1;
    this.cx = 0; this.cy = 0; this.scale = 0.15; this.rot = 0;
    this.hoverId = null;
    this.lastDrawMs = 0;
    this.resize();
  }

  /** Match the backing store to the CSS box and the device pixel ratio. */
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    this.W = Math.max(2, Math.round(r.width));
    this.H = Math.max(2, Math.round(r.height));
    this.canvas.width = Math.round(this.W * this.dpr);
    this.canvas.height = Math.round(this.H * this.dpr);
  }

  /* -- transforms -------------------------------------------------------- */

  /** World -> screen. @param {number} x m @param {number} y m
   *  @returns {number[]} [sx, sy] in CSS pixels. */
  w2s(x, y) {
    const c = Math.cos(this.rot), s = Math.sin(this.rot);
    const dx = x - this.cx, dy = y - this.cy;
    const rx = dx * c + dy * s, ry = -dx * s + dy * c;
    return [this.W / 2 + rx * this.scale, this.H / 2 - ry * this.scale];
  }

  /** Screen -> world. @param {number} sx @param {number} sy
   *  @returns {number[]} [x, y] in metres. */
  s2w(sx, sy) {
    const c = Math.cos(this.rot), s = Math.sin(this.rot);
    const rx = (sx - this.W / 2) / this.scale;
    const ry = (this.H / 2 - sy) / this.scale;
    return [rx * c - ry * s + this.cx, rx * s + ry * c + this.cy];
  }

  /** Canvas rotation of a world direction. @param {number} th rad
   *  @returns {number} rad, usable directly with ctx.rotate/ctx.ellipse. */
  ang(th) { return this.rot - th; }

  /** Adopt the camera described by the app state. @param {Object} S */
  camera(S) {
    const own = S.frame && S.frame.own;
    this.rot = (S.view.headingUp && own && isFinite(own.psi))
      ? (own.psi - Math.PI / 2) : 0;
    if (S.view.follow && own && isFinite(own.x)) {
      this.cx = own.x; this.cy = own.y;
      S.view.cx = own.x; S.view.cy = own.y;
    } else {
      this.cx = S.view.cx; this.cy = S.view.cy;
    }
    this.scale = S.view.scale;
  }

  /** Fit a world AABB into the viewport.
   * @param {number[]} b [xmin,ymin,xmax,ymax] in metres
   * @param {Object} view the mutable view object of the app state. */
  fit(b, view) {
    if (!b || b.length < 4) return;
    const w = Math.max(b[2] - b[0], 1), h = Math.max(b[3] - b[1], 1);
    view.cx = 0.5 * (b[0] + b[2]);
    view.cy = 0.5 * (b[1] + b[3]);
    view.scale = 0.88 * Math.min(this.W / w, this.H / h);
    view.follow = false;
  }

  /** Zoom about a screen point, keeping the world point under it fixed.
   * @param {number} sx @param {number} sy @param {number} factor
   * @param {Object} view mutable view state. */
  zoomAt(sx, sy, factor, view) {
    const before = this.s2w(sx, sy);
    const ns = Math.max(1e-4, Math.min(view.scale * factor, 20));
    view.scale = ns; this.scale = ns;
    const after = this.s2w(sx, sy);
    view.cx += before[0] - after[0];
    view.cy += before[1] - after[1];
    this.cx = view.cx; this.cy = view.cy;
  }

  /** Pan by a screen-space delta. @param {number} dsx @param {number} dsy
   *  @param {Object} view mutable view state. */
  panBy(dsx, dsy, view) {
    const a = this.s2w(0, 0), b = this.s2w(dsx, dsy);
    view.cx -= (b[0] - a[0]); view.cy -= (b[1] - a[1]);
    this.cx = view.cx; this.cy = view.cy;
  }

  /** Nearest agent within a pixel radius.
   * @param {number} sx @param {number} sy @param {Object} S app state
   * @param {number} [tol=14] pixels
   * @returns {?number} agent id. */
  hitTest(sx, sy, S, tol) {
    tol = tol || 14;
    const fr = S.frame; if (!fr) return null;
    let best = null, bd = tol * tol;
    (fr.agents || []).forEach(a => {
      const gx = (a.x !== null && a.x !== undefined) ? a.x : a.ox;
      const gy = (a.y !== null && a.y !== undefined) ? a.y : a.oy;
      if (gx === null || gx === undefined) return;
      const p = this.w2s(gx, gy);
      const d = (p[0] - sx) * (p[0] - sx) + (p[1] - sy) * (p[1] - sy);
      if (d < bd) { bd = d; best = a.id; }
    });
    return best;
  }

  /* -- main draw --------------------------------------------------------- */

  /** Render one frame of the whole scene. @param {Object} S app state. */
  draw(S) {
    const t0 = performance.now();
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = this.pal.bg;
    ctx.fillRect(0, 0, this.W, this.H);
    this.camera(S);

    const fr = S.frame;
    if (S.layers.grid) this._grid();
    if (fr && S.layers.context) this._context(fr.static || S.staticCtx);
    if (fr) {
      if (S.layers.rings) this._rings(fr.own);
      if (S.layers.trails) this._trails(S);
      if (S.layers.fan && S.fan) this._fan(S);
      this._agents(S);
      this._ownship(fr.own);
      this._selection(S);
    } else {
      this._placeholder();
    }
    this._hud(S);
    this.lastDrawMs = performance.now() - t0;
  }

  _placeholder() {
    const ctx = this.ctx;
    ctx.fillStyle = this.pal.faint;
    ctx.font = '13px ' + 'Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no frame - press "New simulation" or load a replay episode',
      this.W / 2, this.H / 2);
    ctx.textAlign = 'left';
  }

  /* -- layers ------------------------------------------------------------ */

  /** World-aligned metric grid with edge labels in metres. */
  _grid() {
    const ctx = this.ctx;
    const corners = [this.s2w(0, 0), this.s2w(this.W, 0),
                     this.s2w(0, this.H), this.s2w(this.W, this.H)];
    const xs = corners.map(c => c[0]), ys = corners.map(c => c[1]);
    const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    const y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    const step = niceStep(110 / this.scale);
    if (!isFinite(step) || step <= 0) return;
    const maxLines = 400;
    if ((x1 - x0) / step > maxLines || (y1 - y0) / step > maxLines) return;

    ctx.lineWidth = 1;
    ctx.font = '10px Consolas, monospace';
    ctx.textBaseline = 'alphabetic';

    const drawLine = (ax, ay, bx, by, major, label, vertical) => {
      const p = this.w2s(ax, ay), q = this.w2s(bx, by);
      const c = clipSeg(p[0], p[1], q[0], q[1], this.W, this.H);
      if (!c) return;
      ctx.strokeStyle = major ? this.pal.gridMajor : this.pal.grid;
      ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(c[2], c[3]); ctx.stroke();
      if (!major || !label) return;
      ctx.fillStyle = this.pal.faint;
      let lx, ly;
      if (vertical) {
        const bot = c[1] > c[3] ? [c[0], c[1]] : [c[2], c[3]];
        lx = Math.min(Math.max(bot[0] + 3, 2), this.W - 44);
        ly = Math.min(bot[1] - 4, this.H - 4);
      } else {
        const lft = c[0] < c[2] ? [c[0], c[1]] : [c[2], c[3]];
        lx = Math.max(lft[0] + 3, 3);
        ly = Math.min(Math.max(lft[1] - 3, 11), this.H - 3);
      }
      ctx.fillText(label, lx, ly);
    };

    const start = Math.ceil(x0 / step) * step;
    for (let x = start; x <= x1; x += step) {
      const major = Math.abs(x / step) % 5 < 1e-6;
      drawLine(x, y0, x, y1, major, major ? fmtM(x) + ' E' : null, true);
    }
    const starty = Math.ceil(y0 / step) * step;
    for (let y = starty; y <= y1; y += step) {
      const major = Math.abs(y / step) % 5 < 1e-6;
      drawLine(x0, y, x1, y, major, major ? fmtM(y) + ' N' : null, false);
    }
  }

  /** World bounds, corridors and no-fly zones. @param {Object} st static ctx */
  _context(st) {
    if (!st) return;
    const ctx = this.ctx;
    if (st.bounds && st.bounds.length === 4) {
      const p = [this.w2s(st.bounds[0], st.bounds[1]),
                 this.w2s(st.bounds[2], st.bounds[1]),
                 this.w2s(st.bounds[2], st.bounds[3]),
                 this.w2s(st.bounds[0], st.bounds[3])];
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = this.pal.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]);
      ctx.closePath(); ctx.stroke(); ctx.restore();
    }
    (st.corridors || []).forEach(c => {
      const dx = c.x2 - c.x1, dy = c.y2 - c.y1;
      const L = Math.hypot(dx, dy) || 1;
      const nx = -dy / L * c.half_width, ny = dx / L * c.half_width;
      const q = [this.w2s(c.x1 + nx, c.y1 + ny), this.w2s(c.x2 + nx, c.y2 + ny),
                 this.w2s(c.x2 - nx, c.y2 - ny), this.w2s(c.x1 - nx, c.y1 - ny)];
      ctx.fillStyle = 'rgba(79,195,247,0.055)';
      ctx.strokeStyle = 'rgba(79,195,247,0.30)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(q[i][0], q[i][1]);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    });
    (st.nfz || []).forEach(z => {
      const p = this.w2s(z.x, z.y), r = z.r * this.scale;
      ctx.save();
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(239,83,80,0.07)'; ctx.fill();
      ctx.strokeStyle = 'rgba(239,83,80,0.55)'; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.stroke();
      ctx.clip();
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(239,83,80,0.16)';
      const gap = 9;
      for (let k = -r * 2; k < r * 2; k += gap) {
        ctx.beginPath();
        ctx.moveTo(p[0] - r + k, p[1] - r);
        ctx.lineTo(p[0] + r + k, p[1] + r);
        ctx.stroke();
      }
      ctx.restore();
    });
  }

  /** NMAC and DWC range rings around the ownship. @param {Object} own */
  _rings(own) {
    if (!own || !isFinite(own.x)) return;
    const ctx = this.ctx;
    const C = this.w2s(own.x, own.y);
    const rings = [
      { r: (window.PRISM_CONST && window.PRISM_CONST.NMAC_RADIUS_M) || 152.4,
        c: this.pal.nmac, label: 'NMAC' },
      { r: (window.PRISM_CONST && window.PRISM_CONST.DWC_HMD_M) || 1219.2,
        c: this.pal.dwc, label: 'DWC HMD' }
    ];
    ctx.save();
    ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
    ctx.font = '10px Consolas, monospace';
    rings.forEach(g => {
      const rp = g.r * this.scale;
      if (rp < 4 || rp > 20000) return;
      ctx.strokeStyle = g.c; ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.arc(C[0], C[1], rp, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 0.85; ctx.fillStyle = g.c;
      ctx.fillText(g.label + ' ' + fmtM(g.r), C[0] + 4, C[1] - rp - 3);
    });
    ctx.restore();
  }

  /** Fading ground-truth trails. @param {Object} S */
  _trails(S) {
    const ctx = this.ctx;
    ctx.lineWidth = 1.4;
    S.trails.forEach((pts, id) => {
      if (!pts || pts.length < 2) return;
      const col = S.colorOf(S.typeOf(id));
      const n = pts.length;
      for (let i = 1; i < n; i++) {
        const a = this.w2s(pts[i - 1][0], pts[i - 1][1]);
        const b = this.w2s(pts[i][0], pts[i][1]);
        ctx.globalAlpha = 0.06 + 0.5 * (i / n);
        ctx.strokeStyle = col;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
    });
    if (S.ownTrail && S.ownTrail.length > 1) {
      const n = S.ownTrail.length;
      for (let i = 1; i < n; i++) {
        const a = this.w2s(S.ownTrail[i - 1][0], S.ownTrail[i - 1][1]);
        const b = this.w2s(S.ownTrail[i][0], S.ownTrail[i][1]);
        ctx.globalAlpha = 0.08 + 0.6 * (i / n);
        ctx.strokeStyle = this.pal.own;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  /** Translucent branch-rollout fan - the multimodality check. @param {Object} S */
  _fan(S) {
    const ctx = this.ctx;
    const fan = S.fan;
    if (!fan || !fan.agents) return;
    ctx.lineWidth = 1.2;
    ctx.lineJoin = 'round';
    fan.agents.forEach(a => {
      if (S.selected !== null && S.selected !== undefined &&
          S.fanSelectedOnly && a.id !== S.selected) return;
      const col = S.colorOf(S.typeOf(a.id));
      const nb = (a.branches || []).length || 1;
      const alpha = Math.max(0.10, Math.min(0.55, 3.2 / nb));
      (a.branches || []).forEach(br => {
        if (!br || br.length < 2) return;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = col;
        ctx.beginPath();
        const p0 = this.w2s(br[0][0], br[0][1]);
        ctx.moveTo(p0[0], p0[1]);
        for (let i = 1; i < br.length; i++) {
          const p = this.w2s(br[i][0], br[i][1]);
          ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
        const e = this.w2s(br[br.length - 1][0], br[br.length - 1][1]);
        ctx.globalAlpha = Math.min(1, alpha * 2.2);
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(e[0], e[1], 1.7, 0, 2 * Math.PI); ctx.fill();
      });
    });
    ctx.globalAlpha = 1;
  }

  /** Traffic glyphs, observation crosses and covariance ellipses. @param {Object} S */
  _agents(S) {
    const ctx = this.ctx;
    const fr = S.frame;
    const now = fr.t || 0;
    ctx.font = '10px Consolas, monospace';
    (fr.agents || []).forEach(a => {
      const col = S.colorOf(a.type);
      const hasGt = (a.x !== null && a.x !== undefined && isFinite(a.x));

      /* --- ground truth glyph --- */
      if (S.layers.gt && hasGt) {
        const p = this.w2s(a.x, a.y);
        const th = this.ang(a.psi || 0);
        const det = !!a.det;
        ctx.save();
        ctx.translate(p[0], p[1]); ctx.rotate(th);
        ctx.beginPath();
        ctx.moveTo(9, 0); ctx.lineTo(-6, 5.2);
        ctx.lineTo(-3.4, 0); ctx.lineTo(-6, -5.2);
        ctx.closePath();
        if (det) {
          ctx.fillStyle = col; ctx.globalAlpha = 0.95; ctx.fill();
          ctx.globalAlpha = 1; ctx.strokeStyle = '#0a0d12'; ctx.lineWidth = 0.8;
          ctx.stroke();
        } else {
          ctx.globalAlpha = 0.55; ctx.strokeStyle = col; ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.restore();

        /* velocity vector, 4 s of travel */
        if (isFinite(a.vx)) {
          const q = this.w2s(a.x + a.vx * 4, a.y + a.vy * 4);
          ctx.globalAlpha = a.det ? 0.6 : 0.28;
          ctx.strokeStyle = col; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
          ctx.globalAlpha = 1;
        }

        /* fresh-track highlight: a re-acquisition gets a new track_id */
        const fresh = S.freshTracks.get(a.id);
        if (fresh !== undefined && (now - fresh) < 2.5) {
          const k = 1 - (now - fresh) / 2.5;
          ctx.globalAlpha = 0.25 + 0.6 * k;
          ctx.strokeStyle = this.pal.warn; ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(p[0], p[1], 10 + 16 * (1 - k), 0, 2 * Math.PI);
          ctx.stroke(); ctx.globalAlpha = 1;
        }

        if (S.layers.labels) {
          ctx.fillStyle = this.pal.dim;
          ctx.fillText('#' + a.id, p[0] + 10, p[1] - 8);
        }
      }

      /* --- observation --- */
      if (S.layers.obs && a.det && a.ox !== null && a.ox !== undefined) {
        const o = this.w2s(a.ox, a.oy);
        ctx.strokeStyle = col; ctx.globalAlpha = 0.95; ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(o[0] - 4.5, o[1]); ctx.lineTo(o[0] + 4.5, o[1]);
        ctx.moveTo(o[0], o[1] - 4.5); ctx.lineTo(o[0], o[1] + 4.5);
        ctx.stroke();
        if (hasGt) {
          const p = this.w2s(a.x, a.y);
          ctx.globalAlpha = 0.35; ctx.setLineDash([2, 3]);
          ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(o[0], o[1]); ctx.stroke();
          ctx.setLineDash([]);
        }
        if (S.layers.ell && a.R) {
          const e = eig2(a.R[0][0], a.R[0][1], a.R[1][1]);
          const rx = e[0] * this.scale, ry = e[1] * this.scale;
          if (isFinite(rx) && isFinite(ry) && rx < 1e5) {
            ctx.globalAlpha = 0.85; ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(o[0], o[1], Math.max(rx, 0.7), Math.max(ry, 0.7),
              this.ang(e[2]), 0, 2 * Math.PI);
            ctx.stroke();
            ctx.globalAlpha = 0.10; ctx.fillStyle = col; ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
      }

      /* hover ring */
      if (this.hoverId === a.id && hasGt) {
        const p = this.w2s(a.x, a.y);
        ctx.strokeStyle = this.pal.fg; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p[0], p[1], 12, 0, 2 * Math.PI); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  }

  /** Ownship glyph, velocity vector and heading tick. @param {Object} own */
  _ownship(own) {
    if (!own || !isFinite(own.x)) return;
    const ctx = this.ctx;
    const p = this.w2s(own.x, own.y);
    if (isFinite(own.vx)) {
      const q = this.w2s(own.x + own.vx * 6, own.y + own.vy * 6);
      ctx.strokeStyle = this.pal.own; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      ctx.globalAlpha = 1;
      const a = Math.atan2(q[1] - p[1], q[0] - p[0]);
      ctx.beginPath();
      ctx.moveTo(q[0], q[1]);
      ctx.lineTo(q[0] - 7 * Math.cos(a - 0.4), q[1] - 7 * Math.sin(a - 0.4));
      ctx.lineTo(q[0] - 7 * Math.cos(a + 0.4), q[1] - 7 * Math.sin(a + 0.4));
      ctx.closePath(); ctx.fillStyle = this.pal.own; ctx.fill();
    }
    ctx.save();
    ctx.translate(p[0], p[1]); ctx.rotate(this.ang(own.psi || 0));
    ctx.beginPath();
    ctx.moveTo(12, 0); ctx.lineTo(-8, 7); ctx.lineTo(-4.5, 0); ctx.lineTo(-8, -7);
    ctx.closePath();
    ctx.fillStyle = this.pal.own; ctx.fill();
    ctx.strokeStyle = '#0a0d12'; ctx.lineWidth = 1; ctx.stroke();
    ctx.restore();
  }

  /** Selection highlight plus the predicted CPA geometry. @param {Object} S */
  _selection(S) {
    const fr = S.frame;
    if (S.selected === null || S.selected === undefined || !fr) return;
    const a = (fr.agents || []).find(x => x.id === S.selected);
    if (!a || a.x === null || a.x === undefined) return;
    const ctx = this.ctx;
    const p = this.w2s(a.x, a.y);
    ctx.strokeStyle = this.pal.fg; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p[0], p[1], 14, 0, 2 * Math.PI); ctx.stroke();

    const own = fr.own;
    const g = cpaOf(own, a);
    const o = this.w2s(own.x, own.y);
    ctx.save();
    ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.6;
    ctx.strokeStyle = this.pal.dim; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
    ctx.restore();

    if (g.tcpa !== null && g.tcpa > 0 && isFinite(g.tcpa) && g.tcpa < 300) {
      const ax = a.x + (a.vx || 0) * g.tcpa, ay = a.y + (a.vy || 0) * g.tcpa;
      const ox = own.x + (own.vx || 0) * g.tcpa, oy = own.y + (own.vy || 0) * g.tcpa;
      const A = this.w2s(ax, ay), O = this.w2s(ox, oy);
      const nmac = (window.PRISM_CONST && window.PRISM_CONST.NMAC_RADIUS_M) || 152.4;
      const col = g.cpa < nmac ? this.pal.bad
        : (g.cpa < ((window.PRISM_CONST && window.PRISM_CONST.DWC_HMD_M) || 1219.2)
          ? this.pal.warn : this.pal.ok);
      ctx.save();
      ctx.globalAlpha = 0.9; ctx.strokeStyle = col; ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(A[0], A[1]); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(O[0], O[1]); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(O[0], O[1]); ctx.stroke();
      [A, O].forEach(q => {
        ctx.beginPath(); ctx.arc(q[0], q[1], 3, 0, 2 * Math.PI);
        ctx.fillStyle = col; ctx.fill();
      });
      ctx.font = '11px Consolas, monospace';
      ctx.fillStyle = col;
      ctx.fillText('CPA ' + fmtM(g.cpa) + ' @ ' + g.tcpa.toFixed(1) + ' s',
        (A[0] + O[0]) / 2 + 6, (A[1] + O[1]) / 2 - 5);
      ctx.restore();
    }
  }

  /* -- HUD --------------------------------------------------------------- */

  /** Scale bar, north arrow, wind arrow and the frame clock. @param {Object} S */
  _hud(S) {
    const ctx = this.ctx;
    const fr = S.frame;
    ctx.font = '11px Consolas, monospace';
    ctx.textBaseline = 'alphabetic';

    /* scale bar (bottom-left) */
    const target = 120 / this.scale;
    const len = niceStep(target);
    const px = len * this.scale;
    const bx = 16, by = this.H - 22;
    ctx.strokeStyle = this.pal.dim; ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(bx, by - 5); ctx.lineTo(bx, by); ctx.lineTo(bx + px, by);
    ctx.lineTo(bx + px, by - 5); ctx.stroke();
    ctx.fillStyle = this.pal.dim;
    ctx.fillText(fmtM(len), bx + px / 2 - 16, by - 8);

    /* north arrow (top-right) */
    const nx = this.W - 34, ny = 34;
    const na = this.ang(Math.PI / 2);
    ctx.save();
    ctx.translate(nx, ny); ctx.rotate(na);
    ctx.strokeStyle = this.pal.dim; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(-14, 0); ctx.lineTo(14, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(14, 0); ctx.lineTo(7, 4); ctx.lineTo(7, -4);
    ctx.closePath(); ctx.fillStyle = this.pal.dim; ctx.fill();
    ctx.restore();
    const lp = [nx + 20 * Math.cos(na), ny + 20 * Math.sin(na)];
    ctx.fillStyle = this.pal.fg;
    ctx.fillText('N', lp[0] - 3, lp[1] + 4);

    /* wind arrow (top-left) */
    if (fr && fr.wind && isFinite(fr.wind.speed)) {
      const w = fr.wind;
      const wx = 44, wy = 46;
      const wa = this.ang(w.dir || 0);
      const L = 14 + Math.min(w.speed || 0, 20) * 1.6;
      ctx.save();
      ctx.translate(wx, wy); ctx.rotate(wa);
      ctx.strokeStyle = this.pal.accent; ctx.globalAlpha = 0.85; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(-L / 2, 0); ctx.lineTo(L / 2, 0); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(L / 2, 0); ctx.lineTo(L / 2 - 6, 4.2);
      ctx.lineTo(L / 2 - 6, -4.2); ctx.closePath();
      ctx.fillStyle = this.pal.accent; ctx.fill();
      ctx.restore();
      ctx.globalAlpha = 1;
      ctx.fillStyle = this.pal.accent;
      ctx.fillText('wind ' + (w.speed || 0).toFixed(1) + ' m/s toward ' +
        fmtDeg(w.dir || 0), 10, 22);
    }

    /* clock */
    if (fr) {
      ctx.fillStyle = this.pal.fg;
      const nAg = (fr.agents || []).length;
      const nDet = (fr.agents || []).filter(a => a.det).length;
      const txt = 't = ' + (fr.t || 0).toFixed(2) + ' s   step ' + (fr.step || 0) +
        (fr.n_steps ? '/' + fr.n_steps : '') +
        '   agents ' + nAg + '  detected ' + nDet;
      ctx.fillText(txt, 10, this.H - 6);
    }
  }
}

/* ------------------------------------------------------------- mini charts */

const Charts = {
  /** Prepare a canvas for crisp drawing. @param {HTMLCanvasElement} cv
   *  @returns {Object} {ctx, w, h} in CSS pixels. */
  _prep(cv) {
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const w = Math.max(2, cv.clientWidth), h = Math.max(2, cv.clientHeight);
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  },

  /** Draw one histogram panel per prediction horizon.
   * @param {HTMLCanvasElement} cv target canvas
   * @param {Object[]} horizons entries with {horizon_s, p50, p90, hist:{edges,counts}}
   * @param {number} nmac NMAC radius [m], drawn as a reference line. */
  histograms(cv, horizons, nmac) {
    const pal = readPalette();
    const P = this._prep(cv);
    const ctx = P.ctx;
    ctx.fillStyle = pal.bg; ctx.fillRect(0, 0, P.w, P.h);
    if (!horizons || !horizons.length) {
      ctx.fillStyle = pal.faint; ctx.font = '12px Consolas, monospace';
      ctx.fillText('no diagnosis yet', 10, 20); return;
    }
    const n = horizons.length;
    const pad = 6, gap = 8;
    const pw = (P.w - pad * 2 - gap * (n - 1)) / n;
    horizons.forEach((h, i) => {
      const x0 = pad + i * (pw + gap), y0 = 16, ph = P.h - y0 - 20;
      const hist = h.hist || { edges: [], counts: [] };
      const counts = hist.counts || [];
      const edges = hist.edges || [];
      const maxC = Math.max.apply(null, counts.concat([1]));
      const emax = edges.length ? edges[edges.length - 1] : 1;
      ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph);
      ctx.fillStyle = pal.accent;
      const bw = pw / Math.max(counts.length, 1);
      counts.forEach((c, k) => {
        const bh = (c / maxC) * (ph - 2);
        ctx.fillRect(x0 + k * bw, y0 + ph - bh, Math.max(bw - 0.6, 0.7), bh);
      });
      /* NMAC reference */
      if (emax > 0 && nmac < emax) {
        const xr = x0 + (nmac / emax) * pw;
        ctx.strokeStyle = pal.bad; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(xr, y0); ctx.lineTo(xr, y0 + ph); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = pal.bad; ctx.font = '9px Consolas, monospace';
        ctx.fillText('NMAC', xr + 2, y0 + 9);
      }
      /* median */
      if (h.p50 !== null && h.p50 !== undefined && emax > 0) {
        const xm = x0 + Math.min(h.p50 / emax, 1) * pw;
        ctx.strokeStyle = pal.warn;
        ctx.beginPath(); ctx.moveTo(xm, y0); ctx.lineTo(xm, y0 + ph); ctx.stroke();
      }
      ctx.fillStyle = pal.fg; ctx.font = '11px Consolas, monospace';
      ctx.fillText('CV @ ' + h.horizon_s + ' s', x0 + 2, y0 - 4);
      ctx.fillStyle = pal.dim; ctx.font = '10px Consolas, monospace';
      const p50 = (h.p50 === null || h.p50 === undefined) ? '--' : h.p50.toFixed(0);
      const p90 = (h.p90 === null || h.p90 === undefined) ? '--' : h.p90.toFixed(0);
      ctx.fillText('p50 ' + p50 + '  p90 ' + p90 + ' m  n=' + (h.n || 0),
        x0 + 2, y0 + ph + 12);
    });
  },

  /** Horizontal bar chart of median error by category.
   * @param {HTMLCanvasElement} cv target canvas
   * @param {Object} groups {platform:{key:{label,p50,n}}, intent:{...}}
   * @param {number} nmac NMAC radius [m] used as the reference scale. */
  breakdown(cv, groups, nmac) {
    const pal = readPalette();
    const P = this._prep(cv);
    const ctx = P.ctx;
    ctx.fillStyle = pal.bg; ctx.fillRect(0, 0, P.w, P.h);
    const rows = [];
    ['platform', 'intent'].forEach(kind => {
      const g = groups[kind] || {};
      Object.keys(g).forEach(k => {
        const e = g[k];
        if (!e || !e.n) return;
        rows.push({ label: (kind === 'platform' ? 'P' : 'I') + ' ' +
          (e.label || k), p50: e.p50, n: e.n });
      });
    });
    if (!rows.length) {
      ctx.fillStyle = pal.faint; ctx.font = '12px Consolas, monospace';
      ctx.fillText('no per-category samples', 10, 20); return;
    }
    const maxV = Math.max.apply(null, rows.map(r => r.p50 || 0).concat([nmac * 1.2]));
    const lw = 132, pad = 6;
    const rh = Math.max(9, Math.min(18, (P.h - pad * 2) / rows.length));
    ctx.font = '10px Consolas, monospace';
    rows.forEach((r, i) => {
      const y = pad + i * rh;
      ctx.fillStyle = pal.dim;
      ctx.fillText(r.label.slice(0, 20), 4, y + rh - 3);
      const bw = ((r.p50 || 0) / maxV) * (P.w - lw - 54);
      ctx.fillStyle = (r.p50 || 0) < nmac ? pal.ok : pal.warn;
      ctx.fillRect(lw, y + 2, Math.max(bw, 1), rh - 5);
      ctx.fillStyle = pal.fg;
      ctx.fillText((r.p50 === null ? '--' : (r.p50 || 0).toFixed(0)) + ' m',
        lw + Math.max(bw, 1) + 4, y + rh - 4);
    });
    const xr = lw + (nmac / maxV) * (P.w - lw - 54);
    ctx.strokeStyle = pal.bad; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xr, 0); ctx.lineTo(xr, P.h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.bad;
    ctx.fillText('NMAC', xr + 2, 10);
  }
};

window.PRISM_CONST = window.PRISM_CONST || {};
window.Renderer = Renderer;
window.Charts = Charts;
window.prismUtil = { wrapAngle, fmtM, fmtDeg, eig2, cpaOf, niceStep, readPalette };
