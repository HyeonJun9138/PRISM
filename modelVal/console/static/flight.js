/* Flight view: follow-camera world renderer with trails, prediction fans,
   GT/OBS layers. Pure canvas, no deps. */
"use strict";

/* real airframe lengths [m] by PlatformType -- glyphs render at TRUE scale
   (length x zoom), floored at 3 px so distant traffic stays visible */
const AIRFRAME_LEN_M = { 0: 0.9, 1: 0.8, 2: 2.8, 3: 13, 4: 11, 5: 0.6, 6: 2.5 };
const OWNSHIP_LEN_M = 13;   // the ownship is an eVTOL

function FlightView(canvas) {
  const ctx = canvas.getContext("2d");
  let zoom = 0.5;                        // px per metre
  const trails = new Map();              // slot -> [[x,y],...]
  const ownTrail = [];                   // flown ownship track
  let lastTrailSeq = -1;
  let cam = { x: 0, y: 0, snapped: false };
  const show = { gt: true, obs: true, preds: true, cands: true };

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    zoom = Math.max(0.05, Math.min(1.6, zoom * (e.deltaY < 0 ? 1.15 : 0.87)));
  }, { passive: false });

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
    if (!frame) { return; }
    const own = frame.own;
    // snap on first frame / scenario jump; ease during normal flight
    if (!cam.snapped || Math.hypot(own.x - cam.x, own.y - cam.y) > 1500) {
      cam.x = own.x; cam.y = own.y; cam.snapped = true;
    } else {
      cam.x += (own.x - cam.x) * 0.12;
      cam.y += (own.y - cam.y) * 0.12;
    }
    const sx = xw => (xw - cam.x) * zoom + w / 2;
    const sy = yw => h / 2 - (yw - cam.y) * zoom;

    // grid (250 m)
    ctx.strokeStyle = "#151c2a"; ctx.lineWidth = 1;
    const step = 250 * zoom;
    const ox = ((-cam.x * zoom + w / 2) % step + step) % step;
    const oy = ((cam.y * zoom + h / 2) % step + step) % step;
    for (let x = ox; x < w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = oy; y < h; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    // statics: corridors + NFZ + bounds
    if (statics) {
      (statics.corridors || []).forEach(c => {
        const [p0, p1] = [c.p0 || c[0], c.p1 || c[1]];
        const hw = (c.half_width || c[2] || 60) * zoom;
        const a = { x: sx(p0[0]), y: sy(p0[1]) }, b = { x: sx(p1[0]), y: sy(p1[1]) };
        ctx.strokeStyle = "rgba(77,163,255,0.16)"; ctx.lineWidth = hw * 2;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      });
      (statics.nfz || statics.no_fly_zones || []).forEach(z => {
        const c = z.centre || z.center || z[0], r = (z.radius || z[1]) * zoom;
        ctx.strokeStyle = "rgba(229,96,76,0.55)"; ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        ctx.beginPath(); ctx.arc(sx(c[0]), sy(c[1]), r, 0, 7); ctx.stroke();
        ctx.setLineDash([]);
      });
      if (statics.side_m || statics.bounds) {
        const s = statics.side_m ? statics.side_m / 2 : null;
        if (s) {
          ctx.strokeStyle = "rgba(127,139,156,0.35)"; ctx.setLineDash([3, 6]);
          ctx.strokeRect(sx(-s), sy(s), 2 * s * zoom, 2 * s * zoom);
          ctx.setLineDash([]);
        }
      }
    }

    // in planner mode the fans are shown only when they ARE the planner's
    // input (risk arm 'model'); other arms draw their own world-view below
    const armName = frame.pilot && frame.pilot.control === "planner"
      ? ((frame.pilot.risk && frame.pilot.risk.name) || frame.pilot.risk)
      : null;

    // prediction fans (under the aircraft glyphs)
    if (show.preds && frame.preds && (armName === null || armName === "model")) {
      const cols = ["#5ad7d0", "#e8b45a", "#b78aff"];
      Object.values(frame.preds).forEach(p => {
        p.mu.forEach((branch, k) => {
          const alpha = Math.max(0.10, Math.min(0.85, p.pi[k]));
          ctx.strokeStyle = cols[k % 3];
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          branch.forEach((pt, i) => {
            const X = sx(pt[0]), Y = sy(pt[1]);
            if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          });
          ctx.stroke();
          // sigma ellipse at the horizon tip
          const tip = branch[branch.length - 1];
          const sg = p.sig[k][p.sig[k].length - 1];
          ctx.globalAlpha = alpha * 0.30;
          ctx.fillStyle = cols[k % 3];
          ctx.beginPath();
          ctx.ellipse(sx(tip[0]), sy(tip[1]), Math.max(2, sg[0] * zoom),
                      Math.max(2, sg[1] * zoom), 0, 0, 7);
          ctx.fill();
        });
        ctx.globalAlpha = 1;
      });
    }

    // what the ACTIVE risk arm feeds the planner (its world-view):
    //   orange dashed ring = static risk reach (standoff + sigma)
    //   cyan arrow + ring  = CV extrapolation with grown uncertainty
    if (armName && frame.pilot.viz) {
      const vz = frame.pilot.viz;
      const standoff = vz.standoff || 0;
      ctx.setLineDash([4, 6]);
      (vz.pts || []).forEach(([x, y, sg]) => {
        ctx.strokeStyle = "rgba(232,180,90,0.65)"; ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), Math.max(3, (standoff + sg) * zoom), 0, 7);
        ctx.stroke();
      });
      if (vz.tracks) {
        const T = vz.T || 7;
        const sgT = Math.sqrt((vz.sigma0 || 0) ** 2 + ((vz.growth || 0) * T) ** 2);
        vz.tracks.forEach(([x, y, vx, vy]) => {
          const ex = x + vx * T, ey = y + vy * T;
          ctx.strokeStyle = "rgba(90,215,208,0.75)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(sx(x), sy(y)); ctx.lineTo(sx(ex), sy(ey));
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(sx(ex), sy(ey), Math.max(3, (standoff + sgT) * zoom), 0, 7);
          ctx.stroke();
        });
      }
      ctx.setLineDash([]);
    }

    // candidate fan: every trajectory the planner scored this cycle
    // (teal = feasible, red ghost = violates dynamics/corridor/collision)
    if (show.cands && frame.pilot && frame.pilot.cands) {
      frame.pilot.cands.forEach(cd => {
        ctx.strokeStyle = cd.v ? "rgba(90,215,208,0.16)"
                               : "rgba(229,96,76,0.06)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        cd.p.forEach((pt, i) => {
          const X = sx(pt[0]), Y = sy(pt[1]);
          if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        });
        ctx.stroke();
      });
    }

    // active Frenet plan (planner mode): the path the ownship is tracking,
    // drawn with a white under-glow so it pops above the candidate fan
    if (frame.pilot && frame.pilot.poly && frame.pilot.poly.length > 1) {
      const em = frame.pilot.emergency;
      const trace = () => {
        ctx.beginPath();
        frame.pilot.poly.forEach((pt, i) => {
          const X = sx(pt[0]), Y = sy(pt[1]);
          if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        });
        ctx.stroke();
      };
      ctx.globalAlpha = 0.35;
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 4.5;
      trace();
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = em ? "#ff5470" : "#7dff9a";
      ctx.lineWidth = 2.6;
      ctx.setLineDash([8, 5]);
      trace();
      ctx.setLineDash([]);
      const tip = frame.pilot.poly[frame.pilot.poly.length - 1];
      ctx.fillStyle = em ? "#ff5470" : "#7dff9a";
      ctx.beginPath(); ctx.arc(sx(tip[0]), sy(tip[1]), 4.5, 0, 7); ctx.fill();
      ctx.globalAlpha = 1;
    }

    // traffic
    frame.agents.forEach(a => {
      const tr = trails.get(a.slot) || [];
      tr.push([a.x, a.y]);
      if (tr.length > 60) tr.shift();
      trails.set(a.slot, tr);
      const col = (statics && statics.type_colors && statics.type_colors[a.type]) || "#8b97a8";
      if (show.gt) {
        ctx.strokeStyle = col; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.4;
        ctx.beginPath();
        tr.forEach((pt, i) => { const X = sx(pt[0]), Y = sy(pt[1]);
          if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
        ctx.stroke(); ctx.globalAlpha = 1;
        const px = Math.max(3, (AIRFRAME_LEN_M[a.type] || 2) * zoom * 0.6);
        glyph(a.x, a.y, a.psi, col, a.det ? 1 : 0.5, px);
      }
      if (show.obs && a.det && a.ox !== undefined) {
        ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.8; ctx.lineWidth = 1;
        const X = sx(a.ox), Y = sy(a.oy);
        ctx.beginPath(); ctx.moveTo(X - 4, Y); ctx.lineTo(X + 4, Y);
        ctx.moveTo(X, Y - 4); ctx.lineTo(X, Y + 4); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
    for (const slot of [...trails.keys()]) {
      if (!frame.agents.some(a => a.slot === slot)) trails.delete(slot);
    }

    // flown ownship track (the trail the paper's trajectory figures show)
    if (frame.seq !== lastTrailSeq) {
      lastTrailSeq = frame.seq;
      ownTrail.push([own.x, own.y]);
      if (ownTrail.length > 240) ownTrail.shift();   // ~2 min at 2 Hz
    }
    if (ownTrail.length > 1) {
      ctx.strokeStyle = "rgba(255,255,255,0.38)"; ctx.lineWidth = 1.7;
      ctx.beginPath();
      ownTrail.forEach((pt, i) => {
        const X = sx(pt[0]), Y = sy(pt[1]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      });
      ctx.stroke();
    }

    // safety rings around ownship
    ring(own.x, own.y, statics ? statics.nmac_m : 152.4, "rgba(229,96,76,0.7)");
    ring(own.x, own.y, statics ? statics.dwc_m : 1219.2, "rgba(232,180,90,0.4)");

    // ownship (same true-scale rule as traffic)
    glyph(own.x, own.y, own.psi, "#ffffff", 1,
          Math.max(4, OWNSHIP_LEN_M * zoom * 0.6));

    function ring(x, y, r, colr) {
      ctx.strokeStyle = colr; ctx.lineWidth = 1.2; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.arc(sx(x), sy(y), r * zoom, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
    }
    function glyph(x, y, psi, colr, alpha, size) {
      const X = sx(x), Y = sy(y);
      if (X < -30 || X > w + 30 || Y < -30 || Y > h + 30) return;
      ctx.save(); ctx.translate(X, Y); ctx.rotate(-psi);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = colr;
      ctx.beginPath();
      ctx.moveTo(size, 0); ctx.lineTo(-size * 0.6, size * 0.5);
      ctx.lineTo(-size * 0.3, 0); ctx.lineTo(-size * 0.6, -size * 0.5);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(11,14,20,0.9)";   // dark outline pops on any bg
      ctx.lineWidth = 1; ctx.stroke();
      ctx.restore(); ctx.globalAlpha = 1;
    }
  }

  function clearTrails() { trails.clear(); ownTrail.length = 0; }
  return { draw, show, clearTrails, get zoom() { return zoom; } };
}
