/* Risk map (doc [6]): ownship-centred, heading-up bird's-eye view.
   Each prediction branch step splats a Gaussian sprite; colour = horizon
   bucket (0-5 s red / 5-10 s amber / 10-15 s blue), size = model sigma,
   alpha = branch probability. Certain object -> small & dark; uncertain ->
   wide & faint. Exactly the doc [6] painting rule, live. */
"use strict";

function RiskMap(canvas) {
  const ctx = canvas.getContext("2d");
  let range = 1600;                       // metres from centre to top edge

  // pre-rendered radial sprite (unit gaussian blob)
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = 64;
  const sctx = sprite.getContext("2d");
  const grad = sctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.45)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  sctx.fillStyle = grad;
  sctx.fillRect(0, 0, 64, 64);

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    range = Math.max(400, Math.min(4000, range * (e.deltaY < 0 ? 0.87 : 1.15)));
  }, { passive: false });

  const BUCKET_COLOR = [[229, 96, 76], [232, 180, 90], [77, 163, 255]];
  const SWEEP_PERIOD_MS = 2400;          // one radar revolution
  const TYPE_KO = { 0: "새", 1: "드론", 2: "고정익", 3: "eVTOL",
                    4: "GA", 5: "잔해", 6: "미확인" };

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr; canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }

  function draw(frame, statics) {
    const [w, h] = resize();
    ctx.clearRect(0, 0, w, h);
    if (!frame) return;
    const own = frame.own;
    const cx = w / 2, cy = h * 0.62;        // ownship below centre: look ahead
    const scale = (h * 0.58) / range;
    const cs = Math.cos(own.psi), sn = Math.sin(own.psi);

    // world -> screen (rotate so ownship heading points UP)
    function S(xw, yw) {
      const dx = xw - own.x, dy = yw - own.y;
      const fwd = dx * cs + dy * sn;         // along heading
      const rgt = dx * sn - dy * cs;         // to the right of heading
      return [cx + rgt * scale, cy - fwd * scale];
    }

    // range rings
    ctx.strokeStyle = "#1c2534"; ctx.fillStyle = "#55617a";
    ctx.font = "10px Consolas"; ctx.lineWidth = 1;
    const maxRing = Math.min(range, 1500);
    [500, 1000, 1500].forEach(r => {
      if (r * scale < Math.max(w, h)) {
        ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, 7); ctx.stroke();
        ctx.fillText(`${r}m`, cx + 4, cy - r * scale + 11);
      }
    });

    // radar sweep: rotating beam with an afterglow tail, clipped to the
    // outer ring. Pure cosmetics -- but it sells the "live sensor" feel.
    const sweepR = maxRing * scale;
    const ang = ((performance.now() % SWEEP_PERIOD_MS) / SWEEP_PERIOD_MS) * 2 * Math.PI;
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, sweepR, 0, 7); ctx.clip();
    if (ctx.createConicGradient) {
      const g = ctx.createConicGradient(ang, cx, cy);
      g.addColorStop(0.0, "rgba(77,163,255,0.30)");
      g.addColorStop(0.10, "rgba(77,163,255,0.10)");
      g.addColorStop(0.28, "rgba(77,163,255,0.02)");
      g.addColorStop(0.30, "rgba(77,163,255,0)");
      g.addColorStop(1.0, "rgba(77,163,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(cx - sweepR, cy - sweepR, 2 * sweepR, 2 * sweepR);
    }
    // leading edge line
    ctx.strokeStyle = "rgba(140,200,255,0.55)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + sweepR * Math.cos(ang), cy + sweepR * Math.sin(ang));
    ctx.stroke();
    ctx.restore();
    // NMAC / DWC rings
    [[statics ? statics.nmac_m : 152.4, "rgba(229,96,76,0.8)"],
     [statics ? statics.dwc_m : 1219.2, "rgba(232,180,90,0.45)"]].forEach(([r, c]) => {
      ctx.strokeStyle = c; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
    });

    // risk splats -- additive blending makes overlap glow
    ctx.globalCompositeOperation = "lighter";
    if (frame.preds) {
      Object.values(frame.preds).forEach(p => {
        p.mu.forEach((branch, k) => {
          const pi = p.pi[k];
          if (pi < 0.03) return;
          branch.forEach((pt, i) => {
            const stepIdx = p.steps[i];               // 0..29 original index
            const bucket = stepIdx < 10 ? 0 : stepIdx < 20 ? 1 : 2;
            const [R, G, B] = BUCKET_COLOR[bucket];
            const sg = p.sig[k][i];
            const rx = Math.max(18, 2.0 * sg[0] * scale);   // 2-sigma extent
            const ry = Math.max(18, 2.0 * sg[1] * scale);
            const [X, Y] = S(pt[0], pt[1]);
            if (X < -rx || X > w + rx || Y < -ry || Y > h + ry) return;
            const alpha = 0.16 * pi * (1.0 - 0.4 * (stepIdx / 30));
            ctx.save();
            ctx.translate(X, Y);
            ctx.rotate(-(p.thetaScreen || 0));
            ctx.globalAlpha = alpha;
            // tint the white sprite via a temp fill trick: draw sprite then
            // multiply is expensive -- draw pre-tinted with fillStyle shadow:
            ctx.drawImage(tinted(R, G, B), -rx, -ry, 2 * rx, 2 * ry);
            ctx.restore();
          });
        });
      });
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;

    // traffic current positions (observed = solid, undetected = hollow),
    // labelled with the MODEL'S inferred type -- green when it matches the
    // ground truth, amber when the aux head is wrong. Live doc 5.2 check.
    ctx.font = "11px Consolas"; ctx.textAlign = "left";
    frame.agents.forEach(a => {
      const [X, Y] = S(a.x, a.y);
      if (X < -20 || X > w + 20 || Y < -20 || Y > h + 20) return;
      const col = (statics && statics.type_colors && statics.type_colors[a.type]) || "#8b97a8";
      const LEN = { 0: 0.9, 1: 0.8, 2: 2.8, 3: 13, 4: 11, 5: 0.6, 6: 2.5 };
      const rr = Math.max(2, (LEN[a.type] || 2) * scale * 0.5);
      ctx.beginPath(); ctx.arc(X, Y, rr, 0, 7);
      if (a.det) { ctx.fillStyle = col; ctx.fill(); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke(); }
      const p = frame.preds && frame.preds[String(a.slot)];
      if (p && p.type) {
        const [tIdx, prob] = p.type;
        const ok = tIdx === a.type;
        ctx.fillStyle = ok ? "rgba(55,201,139,0.95)" : "rgba(232,180,90,0.95)";
        ctx.fillText(`${TYPE_KO[tIdx] || "?"} ${(prob * 100) | 0}%`, X + 7, Y - 6);
        if (!ok) {
          ctx.fillStyle = "rgba(127,139,156,0.8)";
          ctx.fillText(`(실제 ${TYPE_KO[a.type] || "?"})`, X + 7, Y + 6);
        }
      }
    });

    // ownship marker (fixed, heading up)
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(cx, cy - 11); ctx.lineTo(cx - 7, cy + 7);
    ctx.lineTo(cx, cy + 3); ctx.lineTo(cx + 7, cy + 7);
    ctx.closePath(); ctx.fill();
  }

  // tinted sprite cache
  const tintCache = new Map();
  function tinted(r, g, b) {
    const key = `${r},${g},${b}`;
    let c = tintCache.get(key);
    if (c) return c;
    c = document.createElement("canvas");
    c.width = c.height = 64;
    const t = c.getContext("2d");
    t.drawImage(sprite, 0, 0);
    t.globalCompositeOperation = "source-in";
    t.fillStyle = `rgb(${r},${g},${b})`;
    t.fillRect(0, 0, 64, 64);
    tintCache.set(key, c);
    return c;
  }

  return { draw };
}
