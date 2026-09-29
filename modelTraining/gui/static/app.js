/* PRISM model-training GUI: single explicit state object, SSE live updates,
   charts rebuilt from persisted metrics when switching jobs. No frameworks. */
"use strict";

const S = {
  runs: [],            // discovered dataset runs with role assignment
  roles: {},           // path -> "train" | "val" | "off"
  fields: [],
  config: {},
  jobId: null,
  sse: null,
  poll: null,          // interval id when following an out-of-process job
  pollCursor: 0,
  charts: {},
  lastStatus: null,
};
const LIVE_STATES = ["queued", "indexing", "running", "validating",
                     "finalizing", "evaluating", "stopping"];

// ------------------------------------------------------------------ helpers
const $ = id => document.getElementById(id);
async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) } : undefined;
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: "bad JSON from " + path }));
  if (!r.ok || j.error) throw new Error(j.error || (r.status + " " + path));
  return j;
}
function toast(msg, isErr) {
  const el = document.createElement("div");
  el.className = "toast-item" + (isErr ? " err" : "");
  el.textContent = msg;
  $("toast").appendChild(el);
  setTimeout(() => el.remove(), isErr ? 9000 : 4000);
}
function getPath(obj, dotted) {
  return dotted.split(".").reduce((o, k) => (o ? o[k] : undefined), obj);
}
function setPath(obj, dotted, v) {
  const ks = dotted.split(".");
  let o = obj;
  for (const k of ks.slice(0, -1)) { if (!(k in o)) o[k] = {}; o = o[k]; }
  o[ks[ks.length - 1]] = v;
}
function fmtEta(s) {
  if (s === null || s === undefined || !isFinite(s)) return "-";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

// -------------------------------------------------------------------- init
window.addEventListener("load", async () => {
  initCharts();
  try {
    const env = await api("/api/env");
    $("env-chip").textContent = env.cuda_available
      ? `${env.gpu} · CUDA ${env.cuda} · ${env.vram_gb} GB`
      : "CPU only (CUDA 미탐지)";
  } catch (e) { $("env-chip").textContent = "env error"; }
  await loadDefaults();
  await rescanRuns();
  await refreshJobs();
  bindButtons();
  window.addEventListener("resize", renderAllCharts);
});

function initCharts() {
  S.charts.loss = new LineChart($("ch-loss"), { logY: true });
  S.charts.val = new LineChart($("ch-val"));
  S.charts.nll = new LineChart($("ch-nll"));
  S.charts.cov = new LineChart($("ch-cov"), { yMin: 0, yMax: 1, refLine: 0.9 });
  S.charts.mode = new LineChart($("ch-mode"), { yMin: 0, yMax: 1 });
  S.charts.lr = new LineChart($("ch-lr"), { logY: true });
}
function clearCharts() { Object.values(S.charts).forEach(c => c.clear()); }
function renderAllCharts() { Object.values(S.charts).forEach(c => c.render()); }

// ---------------------------------------------------------------- data tab
async function rescanRuns() {
  try {
    const j = await api("/api/data/runs");
    S.runs = j.runs;
    for (const r of S.runs) {
      if (!(r.path in S.roles)) {
        const p = (r.preset || "").toLowerCase();
        S.roles[r.path] = p.includes("valid") ? "val"
          : (p.includes("ood") || p.includes("test")) ? "off" : "train";
      }
    }
    renderRunsTable();
  } catch (e) { toast("런 스캔 실패: " + e.message, true); }
}
function renderRunsTable() {
  const rows = S.runs.map(r => `
    <tr>
      <td title="${r.path}">${r.run_id}${r.schema_ok ? "" : " ⚠"}</td>
      <td class="num">${r.n_episodes}</td>
      <td title="${r.preset || ""}">${r.preset || "-"}</td>
      <td><select class="role-sel" data-path="${r.path}">
        <option value="train" ${S.roles[r.path] === "train" ? "selected" : ""}>train</option>
        <option value="val" ${S.roles[r.path] === "val" ? "selected" : ""}>val</option>
        <option value="off" ${S.roles[r.path] === "off" ? "selected" : ""}>사용 안 함</option>
      </select></td>
    </tr>`).join("");
  $("runs-table").innerHTML = S.runs.length
    ? `<table class="mini"><tr><th>run</th><th class="c-eps">eps</th><th class="c-preset">preset</th><th class="c-role">역할</th></tr>${rows}</table>`
    : `<div class="hint">%PRISM_ROOT%\\Logs 에 런이 없습니다. dataGen으로 먼저 생성하세요.</div>`;
  document.querySelectorAll(".role-sel").forEach(sel =>
    sel.addEventListener("change", () => { S.roles[sel.dataset.path] = sel.value; }));
}

// -------------------------------------------------------------- config form
async function loadDefaults() {
  const j = await api("/api/config/defaults");
  S.config = j.config;
  S.fields = j.fields;
  const groups = {};
  for (const f of S.fields) (groups[f.group] = groups[f.group] || []).push(f);
  let html = "";
  for (const [g, fs] of Object.entries(groups)) {
    html += `<div class="cfg-group"><div class="gname">${g}</div>`;
    for (const f of fs) {
      const v = getPath(S.config, f.key);
      const title = f.help ? ` title="${f.help}"` : "";
      if (f.type === "select") {
        const opts = f.options.map(o =>
          `<option value="${o}" ${o == v ? "selected" : ""}>${o}</option>`).join("");
        html += `<label class="row"${title}><span>${f.label}</span><select data-key="${f.key}">${opts}</select></label>`;
      } else if (f.type === "text") {
        html += `<label class="row"${title}><span>${f.label}</span><input type="text" data-key="${f.key}" value="${v ?? ""}"></label>`;
      } else {
        const step = f.type === "float" ? "any" : "1";
        html += `<label class="row"${title}><span>${f.label}</span><input type="number" step="${step}" data-key="${f.key}" value="${v}"></label>`;
      }
    }
    html += "</div>";
  }
  $("config-form").innerHTML = html;
  $("config-form").querySelectorAll("[data-key]").forEach(inp =>
    inp.addEventListener("change", () => {
      const spec = S.fields.find(f => f.key === inp.dataset.key);
      let val = inp.value;
      if (spec && spec.type === "int") val = parseInt(val, 10);
      if (spec && spec.type === "float") val = parseFloat(val);
      setPath(S.config, inp.dataset.key, val);
    }));
}

// ------------------------------------------------------------------ actions
function roster() {
  return {
    train: S.runs.filter(r => S.roles[r.path] === "train").map(r => r.path),
    val: S.runs.filter(r => S.roles[r.path] === "val").map(r => r.path),
  };
}
function bindButtons() {
  $("btn-rescan").onclick = rescanRuns;
  $("btn-refresh-jobs").onclick = refreshJobs;
  $("btn-start").onclick = startTraining;
  $("btn-stop").onclick = () => stopJob(false);
  $("btn-force").onclick = () => stopJob(true);
  $("btn-eval").onclick = startEval;
  $("job-select").onchange = () => attachJob($("job-select").value);
}
async function startTraining() {
  const ro = roster();
  if (!ro.train.length) { toast("train 역할 런을 최소 1개 지정하세요", true); return; }
  const sweep = $("seed-sweep").value.trim();
  const body = { config: S.config, roster: ro, tag: S.config.tag };
  if (sweep) {
    const seeds = sweep.split(",").map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
    if (seeds.length) body.seeds = seeds;
  }
  try {
    const j = await api("/api/train/start", body);
    toast(`학습 시작: ${j.job_ids.length}개 잡 큐 등록`);
    await refreshJobs();
    attachJob(j.job_ids[0]);
  } catch (e) { toast("시작 실패: " + e.message, true); }
}
async function stopJob(force) {
  if (!S.jobId) return;
  try {
    await api("/api/train/stop", { job_id: S.jobId, force });
    toast(force ? "강제 중단 요청" : "정리 후 종료 요청 (최종 모델 패키징)");
  } catch (e) { toast(e.message, true); }
}
async function startEval() {
  if (!S.jobId) return;
  const ro = roster();
  const runs = ro.val.length ? ro.val : ro.train;
  try {
    const j = await api("/api/eval/start", { job_id: S.jobId, runs });
    toast("평가 잡 시작: " + j.job_ids[0]);
    await refreshJobs();
  } catch (e) { toast("평가 실패: " + e.message, true); }
}

// -------------------------------------------------------------------- jobs
async function refreshJobs() {
  try {
    const j = await api("/api/jobs");
    const sel = $("job-select");
    const cur = S.jobId;
    sel.innerHTML = j.jobs.map(job =>
      `<option value="${job.id}">${job.id} · ${job.status}</option>`).join("")
      || `<option value="">(잡 없음)</option>`;
    if (cur && j.jobs.some(x => x.id === cur)) sel.value = cur;
    else if (j.jobs.length) { attachJob(j.jobs[0].id); sel.value = j.jobs[0].id; }
  } catch (e) { toast(e.message, true); }
}

async function attachJob(jobId) {
  if (!jobId || jobId === S.jobId) return;
  S.jobId = jobId;
  if (S.sse) { S.sse.close(); S.sse = null; }
  if (S.poll) { clearInterval(S.poll); S.poll = null; }
  clearCharts();
  $("log-view").textContent = "";
  // rebuild charts from persisted metrics (works for finished/offline jobs too)
  try {
    const j = await api(`/api/jobs/metrics?job_id=${encodeURIComponent(jobId)}&after=0`);
    S.pollCursor = j.cursor;
    ingestMetrics(j.metrics);
    renderAllCharts();
  } catch (e) { toast("메트릭 로드 실패: " + e.message, true); }
  try {
    const d = await api(`/api/jobs/detail?job_id=${encodeURIComponent(jobId)}`);
    applyStatus(d);
    renderCkpts(d);
    if (!d.offline && !["finished", "stopped", "failed", "aborted"].includes(d.status)) {
      openStream(jobId);
    } else if (d.offline && LIVE_STATES.includes(d.status)) {
      // job runs in ANOTHER process (headless CLI): follow it by polling the
      // persisted metrics/job.json instead of SSE. Stop control is disabled -
      // this server cannot signal a foreign process.
      startPolling(jobId, d);
    }
  } catch (e) { toast(e.message, true); }
}

function startPolling(jobId, detail) {
  const epochs = detail?.config?.config?.optim?.epochs || 0;
  toast("외부 프로세스 잡 — 5초 간격 폴링으로 따라갑니다");
  S.poll = setInterval(async () => {
    if (S.jobId !== jobId) { clearInterval(S.poll); S.poll = null; return; }
    try {
      const j = await api(`/api/jobs/metrics?job_id=${encodeURIComponent(jobId)}&after=${S.pollCursor}`);
      if (j.metrics.length) {
        S.pollCursor = j.cursor;
        ingestMetrics(j.metrics);
        renderAllCharts();
        const last = j.metrics[j.metrics.length - 1];
        if (epochs && last.epoch !== undefined) {
          $("st-epoch").textContent = `${last.epoch}/${epochs}`;
          $("progress-bar").style.width = (100 * last.epoch / epochs).toFixed(1) + "%";
          $("progress-text").textContent =
            `${jobId} · 외부 프로세스 · epoch ${last.epoch}/${epochs}`;
        }
        if (last.samples_per_s) $("st-sps").textContent = last.samples_per_s;
      }
      const d = await api(`/api/jobs/detail?job_id=${encodeURIComponent(jobId)}`);
      const badge = $("status-badge");
      badge.textContent = d.status + " (외부)";
      badge.className = "badge " + d.status;
      if (!LIVE_STATES.includes(d.status)) {
        clearInterval(S.poll); S.poll = null;
        applyStatus(d);
        renderCkpts(d);
        refreshJobs();
      }
    } catch (e) { /* transient read races are fine; next tick retries */ }
  }, 5000);
}

function openStream(jobId) {
  const es = new EventSource(`/api/train/stream?job_id=${encodeURIComponent(jobId)}`);
  S.sse = es;
  let needRender = false;
  es.addEventListener("metric", ev => {
    ingestMetrics(JSON.parse(ev.data).items);
    needRender = true;
  });
  es.addEventListener("log", ev => {
    const lines = JSON.parse(ev.data).lines;
    const lv = $("log-view");
    lv.textContent += lines.join("\n") + "\n";
    lv.scrollTop = lv.scrollHeight;
  });
  es.addEventListener("status", ev => {
    const st = JSON.parse(ev.data);
    applyStatus(st);
    if (["finished", "stopped", "failed", "aborted"].includes(st.status)) {
      es.close(); S.sse = null;
      refreshJobs();
      api(`/api/jobs/detail?job_id=${encodeURIComponent(jobId)}`)
        .then(renderCkpts).catch(() => {});
    }
  });
  es.onerror = () => {   // auto-reconnect after a hiccup, unless job ended
    if (S.sse === es && S.lastStatus && !["finished", "stopped", "failed"].includes(S.lastStatus.status)) {
      setTimeout(() => { if (S.sse === es) { es.close(); S.sse = null; openStream(jobId); } }, 2000);
    }
  };
  setInterval(() => { if (needRender) { renderAllCharts(); needRender = false; } }, 800);
}

// ------------------------------------------------------------- metric flow
function ingestMetrics(items) {
  for (const m of items || []) {
    if (m.split === "train") {
      S.charts.loss.push("total", m.step, m.loss);
      S.charts.loss.push("traj", m.step, m.loss_traj);
      S.charts.loss.push("mode", m.step, m.loss_mode);
      S.charts.loss.push("type", m.step, m.loss_type);
      S.charts.lr.push("lr", m.step, m.lr);
      S.charts.lr.push("samples/s", m.step, m.samples_per_s);
    } else if (m.split === "val") {
      S.charts.val.push("minADE", m.epoch, m.minADE);
      S.charts.val.push("minFDE", m.epoch, m.minFDE);
      S.charts.nll.push("NLL", m.epoch, m.NLL);
      S.charts.cov.push("cov90@5s", m.epoch, m.cov90_5s);
      S.charts.cov.push("cov90@10s", m.epoch, m.cov90_10s);
      S.charts.cov.push("cov90@15s", m.epoch, m.cov90_15s);
      for (const k of Object.keys(m)) {
        if (k.startsWith("use_")) S.charts.mode.push(k.replace("use_", "mode "), m.epoch, m[k]);
      }
      S.charts.mode.push("entropy", m.epoch, m.mode_entropy);
    }
  }
}

// ------------------------------------------------------------------- status
function applyStatus(st) {
  S.lastStatus = st;
  const badge = $("status-badge");
  badge.textContent = st.status;
  badge.className = "badge " + st.status;
  const p = st.progress || {};
  $("progress-bar").style.width = (p.pct || 0).toFixed(1) + "%";
  const detail = p.phase_detail ? ` · ${p.phase_detail}` : "";
  $("progress-text").textContent =
    `${st.id || ""} · ${(p.pct || 0).toFixed(1)}%${detail}`;
  $("st-epoch").textContent = p.epochs ? `${p.epoch}/${p.epochs}` : "-";
  $("st-eta").textContent = fmtEta(p.eta_s);
  $("st-sps").textContent = p.samples_per_s || "-";
  $("st-gpu").textContent = p.gpu_mem_gb ? p.gpu_mem_gb + " GB" : "-";
  if (st.summary && st.summary.best) {
    $("st-best").textContent = st.summary.best.value !== undefined && isFinite(st.summary.best.value)
      ? `${st.summary.best.value.toFixed(2)} @ep${st.summary.best.epoch}` : "-";
  }
  // offline = the job lives in another process; this server cannot stop it
  const live = ["queued", "indexing", "running", "validating", "finalizing", "evaluating"].includes(st.status)
    && !st.offline;
  $("btn-stop").disabled = !live;
  $("btn-force").disabled = !live;
  $("btn-eval").disabled = (live || !S.jobId)
    || (st.offline && LIVE_STATES.includes(st.status));
  if (st.error) toast(st.error.split("\n").pop(), true);
}

function renderCkpts(d) {
  const cks = d.checkpoints || [];
  const rows = cks.map(c => `<tr><td>${c.name}</td><td class="num">${c.mb} MB</td>
      <td class="num">${c.mtime}</td>
      <td><button class="ghost" data-ck="${c.name}">여기서 재개</button></td></tr>`).join("");
  $("ckpt-table").innerHTML = cks.length
    ? `<table class="mini"><tr><th>파일</th><th>크기</th><th>저장</th><th></th></tr>${rows}</table>`
    : `<div class="hint">아직 체크포인트 없음</div>`;
  let extra = "";
  if (d.model_card) {
    const mc = d.model_card;
    extra += `<div class="kv">최종 모델: <b>${mc.final_model_path}</b> · params <b>${(mc.parameters/1e6).toFixed(2)}M</b>`;
    if (mc.latency_ms && mc.latency_ms.batch1_ms !== undefined) {
      extra += ` · 추론 <b>${mc.latency_ms.batch1_ms} ms</b>(b1)`;
    }
    extra += `</div>`;
  }
  if (d.eval_report) {
    const o = d.eval_report.overall || {};
    extra += `<div class="kv">평가: minADE <b>${(o.minADE ?? NaN).toFixed?.(1) ?? "-"} m</b> · NLL <b>${(o.NLL ?? NaN).toFixed?.(3) ?? "-"}</b> · cov90@15s <b>${o.cov90_15s?.toFixed?.(2) ?? "-"}</b></div>`;
  }
  $("ckpt-table").insertAdjacentHTML("beforeend", extra);
  $("ckpt-table").querySelectorAll("[data-ck]").forEach(btn =>
    btn.addEventListener("click", async () => {
      try {
        const j = await api("/api/train/resume", { job_id: S.jobId, checkpoint: btn.dataset.ck });
        toast("재개 잡 시작: " + j.job_ids[0]);
        await refreshJobs();
        attachJob(j.job_ids[0]);
      } catch (e) { toast(e.message, true); }
    }));
}
