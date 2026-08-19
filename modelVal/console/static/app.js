/* Validation console wiring: SSE frames -> both canvases + metric tiles,
   keyboard -> /api/keys, model loader, session controls. */
"use strict";

const $ = id => document.getElementById(id);
async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) } : undefined;
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: "bad json" }));
  if (!r.ok || j.error) throw new Error(j.error || r.status);
  return j;
}
function toast(msg, err) {
  const el = document.createElement("div");
  el.className = "toast-item" + (err ? " err" : "");
  el.textContent = msg;
  $("toast").appendChild(el);
  setTimeout(() => el.remove(), err ? 8000 : 3500);
}

const S = { frame: null, statics: null, playing: false, es: null };
const flight = FlightView($("cv-flight"));
const risk = RiskMap($("cv-risk"));

// ------------------------------------------------------------------ models
async function loadModelList() {
  const j = await api("/api/models");
  const sel = $("model-select");
  sel.innerHTML = j.models.map(m =>
    `<option value="${m.path}">${m.job} · ${m.kind} (${m.mb}MB)</option>`).join("")
    || `<option value="">모델 없음 — 먼저 학습하세요</option>`;
}
$("btn-load").onclick = async () => {
  const path = $("model-select").value;
  if (!path) return;
  $("model-chip").textContent = "로딩 중...";
  try {
    const info = await api("/api/model/load", { path });
    $("model-chip").textContent =
      `${info.path.split("/").slice(-3, -1).join("/")} · ep${info.epoch} · ` +
      `${(info.params / 1e6).toFixed(1)}M · ${info.device}`;
    $("model-chip").className = "chip ok";
    toast("모델 로드 완료 — 같은 시나리오 재시작");
  } catch (e) {
    $("model-chip").textContent = "로드 실패";
    $("model-chip").className = "chip";
    toast(e.message, true);
  }
};

// ---------------------------------------------------------------- controls
$("btn-play").onclick = async () => {
  S.playing = !S.playing;
  await api("/api/control", { action: S.playing ? "play" : "pause" });
  $("btn-play").textContent = S.playing ? "❚❚ Pause" : "▶ Play";
};
$("btn-reset").onclick = () => api("/api/control", { action: "reset" })
  .then(() => toast("새 시나리오"));
$("preset-select").onchange = e =>
  api("/api/control", { action: "preset", preset: e.target.value })
    .then(() => toast("프리셋 변경: " + e.target.value));
$("speed-select").onchange = e =>
  api("/api/control", { action: "speed", speed: parseFloat(e.target.value) });

// ------------------------------------------------------------------- pilot
$("pilot-select").onchange = async e => {
  try {
    const j = await api("/api/pilot", { control: e.target.value });
    $("risk-select").disabled = j.control !== "planner";
    toast(j.control === "planner"
      ? "Frenet 플래너 조종 — 키보드 비활성" : "수동 조종 복귀");
  } catch (err) { toast(err.message, true); e.target.value = "manual"; }
};
$("risk-select").onchange = async e => {
  try {
    const j = await api("/api/pilot", { risk: e.target.value });
    toast(`위험 소스 → ${j.risk} · 같은 시나리오(시드 ${j.seed}) 재시작`);
  } catch (err) { toast(err.message, true); }
};

// ---------------------------------------------------------------- keyboard
const keys = { left: false, right: false, up: false, down: false, space: false };
function pushKeys() {
  const turn = (keys.left ? 1 : 0) + (keys.right ? -1 : 0);   // left = CCW = +
  const accel = (keys.up ? 1 : 0) + (keys.down ? -1 : 0);
  api("/api/keys", { turn, accel, brake: keys.space }).catch(() => {});
}
const KEYMAP = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up",
                 ArrowDown: "down", " ": "space" };
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === "g" || e.key === "G") { flight.show.gt = !flight.show.gt; return; }
  if (e.key === "o" || e.key === "O") { flight.show.obs = !flight.show.obs; return; }
  if (e.key === "p" || e.key === "P") { flight.show.preds = !flight.show.preds; return; }
  if (e.key === "c" || e.key === "C") { flight.show.cands = !flight.show.cands; return; }
  const k = KEYMAP[e.key];
  if (k) { e.preventDefault(); if (!keys[k]) { keys[k] = true; pushKeys(); } }
});
window.addEventListener("keyup", e => {
  const k = KEYMAP[e.key];
  if (k) { e.preventDefault(); keys[k] = false; pushKeys(); }
});
window.addEventListener("blur", () => {
  Object.keys(keys).forEach(k => keys[k] = false);
  pushKeys();
});

// ------------------------------------------------- SSE (+ polling fallback)
// Some external networks (AV web shields, institutional middleboxes) RESET
// long-lived SSE responses. Plain GETs demonstrably pass, so after repeated
// SSE failures the client silently switches to 400 ms polling.
let sseFails = 0;

function ingestFrame(f) {
  if (f.statics) S.statics = f.statics;
  if (S.frame && f.scenario !== S.frame.scenario) {
    flight.clearTrails();            // rotation: old trails reference old world
    toast(`시나리오 #${f.scenario} 시작`);
  }
  S.frame = f;
  updateMetrics(f);
}

function openStream() {
  if (S.es) S.es.close();
  const es = new EventSource("/api/stream");
  S.es = es;
  es.onmessage = ev => {
    sseFails = 0;
    ingestFrame(JSON.parse(ev.data));
  };
  es.onerror = () => {
    es.close();
    if (S.es !== es) return;
    S.es = null;
    sseFails += 1;
    if (sseFails >= 3) {
      toast("스트림이 차단된 네트워크 — 폴링 모드로 전환");
      startPollingFrames();
    } else {
      setTimeout(openStream, 1200);
    }
  };
}

function startPollingFrames() {
  let cursor = 0;
  let busy = false;
  setInterval(async () => {
    if (busy) return;
    busy = true;
    try {
      const j = await api(`/api/frames?after=${cursor}`);
      cursor = j.cursor;
      j.frames.forEach(ingestFrame);
    } catch (e) { /* transient; next tick retries */ }
    busy = false;
  }, 400);
}

// ----------------------------------------------------------------- metrics
function updateMetrics(f) {
  const m = f.metrics || {};
  $("m-lat").textContent = m.latency_ms ? m.latency_ms + " ms" : "-";
  $("m-latmax").textContent = m.latency_max_ms ? "max " + m.latency_max_ms : "";
  $("m-tracks").textContent = m.tracks_predicted ?? "-";
  $("m-err5").textContent = m.err_5s != null ? m.err_5s + " m" : "-";
  $("m-n5").textContent = m.n_5s ? "n=" + m.n_5s : "";
  $("m-err15").textContent = m.err_15s != null ? m.err_15s + " m" : "-";
  $("m-n15").textContent = m.n_15s ? "n=" + m.n_15s : "";
  const cov = m.cov90_15s;
  $("m-cov").textContent = cov != null ? cov.toFixed(2) : "-";
  $("m-cov").parentElement.className =
    "tile" + (cov != null ? (Math.abs(cov - 0.9) < 0.04 ? " good" : " alert") : "");
  $("m-sep").textContent = m.sep_m != null ? m.sep_m + " m" : "-";
  $("m-minsep").textContent = m.min_sep_session_m != null
    ? "세션 최소 " + m.min_sep_session_m : "";
  $("m-sep").parentElement.className = "tile" + (m.nmac_now ? " alert" : "");
  $("m-nmac").textContent = m.nmac_events ?? 0;
  const p = f.pilot || {};
  // keep the selects honest even if this page missed a state change
  // (another tab / API client); never fight the user's open dropdown
  const pName = p.control || "manual";
  const rName = p.risk && p.risk.name ? p.risk.name : (p.risk || "none");
  const pilotSel = $("pilot-select"), riskSel = $("risk-select");
  if (document.activeElement !== pilotSel && pilotSel.value !== pName)
    pilotSel.value = pName;
  if (document.activeElement !== riskSel && riskSel.value !== rName)
    riskSel.value = rName;
  riskSel.disabled = pName !== "planner";
  if (p.control === "planner") {
    $("m-plan").textContent = `${p.valid ?? "-"}/${p.total ?? "-"}`;
    const riskName = p.risk && p.risk.name ? p.risk.name : (p.risk || "");
    const fb = p.risk && p.risk.fallback ? "→none" : "";
    $("m-plansub").textContent =
      `${riskName}${fb}` + (p.plan_ms != null ? ` · ${p.plan_ms}ms` : "") +
      (p.emergency ? " · 비상" : "");
    $("m-plan").parentElement.className = "tile" + (p.emergency ? " alert" : "");
  } else {
    $("m-plan").textContent = "수동";
    $("m-plansub").textContent = "";
    $("m-plan").parentElement.className = "tile";
  }
  $("m-time").textContent = f.t != null ? f.t.toFixed(0) + " s" : "-";
  $("m-scenario").textContent = `시나리오 #${f.scenario} · ${f.preset}`;
  const banner = $("alert-banner");
  if (m.nmac_now) { banner.textContent = "⚠ NMAC — 152 m 침범"; banner.className = "nmac"; }
  else if (m.dwc_now) { banner.textContent = "DWC 분리 상실"; banner.className = "dwc"; }
  else { banner.className = ""; }
}

// -------------------------------------------------------------- render loop
function loop() {
  flight.draw(S.frame, S.statics);
  risk.draw(S.frame, S.statics);
  if (S.frame) {
    const o = S.frame.own;
    $("hud").textContent =
      `V   ${o.v.toFixed(1)} m/s\n` +
      `HDG ${((90 - o.psi * 57.2958) % 360 + 360) % 360 | 0}°\n` +
      `ω   ${o.omega_deg.toFixed(1)} °/s\n` +
      `줌  ${flight.zoom.toFixed(2)} px/m`;
  }
  requestAnimationFrame(loop);
}

window.addEventListener("load", async () => {
  try { await loadModelList(); } catch (e) { toast(e.message, true); }
  try {
    const env = await api("/api/env");
    if (env.model && env.model.path) {
      $("model-chip").textContent = env.model.path.split("/").slice(-3, -1).join("/");
      $("model-chip").className = "chip ok";
    }
    if (env.pilot) {
      $("pilot-select").value = env.pilot.control;
      $("risk-select").value = env.pilot.risk;
      $("risk-select").disabled = env.pilot.control !== "planner";
    }
  } catch (e) { /* server still warming */ }
  openStream();
  loop();
});
