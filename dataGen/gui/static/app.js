/* PRISM dataGen GUI - application state, API glue and UI wiring.
 *
 * Deliberately plain: ONE mutable state object `S`, ONE requestAnimationFrame
 * render loop, explicit DOM updates throttled to human rates. No framework, no
 * external asset, no build step.
 */
'use strict';

/* ============================== state ================================== */

const S = {
  mode: 'live',                 // 'live' | 'replay'
  tab: 'live',

  meta: null,
  colors: {},                   // PlatformType -> hex
  typeNames: {}, intentNames: {}, modeNames: {}, riskNames: {}, geomNames: {},
  presets: [],
  config: null,                 // the config dict currently in the left panel

  sim_id: null,
  frame: null,
  staticCtx: null,
  nSteps: 0,
  playing: false,
  speed: 1,

  trails: new Map(),            // agent id -> [[x,y,t], ...]
  ownTrail: [],
  types: new Map(),             // agent id -> PlatformType
  lastTrackId: new Map(),
  freshTracks: new Map(),       // agent id -> t of last track_id change
  trailSeconds: 20,

  layers: { gt: 1, obs: 1, ell: 1, fan: 0, trails: 1, rings: 1, grid: 1, context: 1, labels: 0 },
  view: { cx: 0, cy: 0, scale: 0.15, follow: false, headingUp: false },
  fitted: false,

  selected: null,
  fan: null,
  fanSelectedOnly: false,

  job: null, jobTimer: 0,
  runs: [], runIndex: null, sortKey: null, sortDir: 1,
  replay: null, replayIdx: 0, replayPlaying: false, replayTimer: 0,
  diag: null,

  es: null, esErrors: 0,
  fps: 0, _fpsT: 0, _fpsN: 0
};

/** Colour for a platform type. @param {number} t @returns {string} hex */
S.colorOf = function (t) { return S.colors[t] || '#90a4ae'; };
/** Last-known platform type of an agent id. @param {number} id @returns {number} */
S.typeOf = function (id) { const v = S.types.get(id); return (v === undefined) ? 6 : v; };

/* ============================== helpers ================================ */

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.prototype.slice.call(document.querySelectorAll(sel));

/** Set the status bar text. @param {string} msg @param {boolean} [err] */
function setStatus(msg, err) {
  const el = $('#statusMsg');
  el.textContent = msg;
  $('#status').classList.toggle('err', !!err);
  if (err) console.warn('[prism]', msg);
}

/** JSON fetch with a uniform error envelope.
 * @param {string} path @param {Object} [opts] fetch options
 * @returns {Promise<Object>} parsed body */
async function api(path, opts) {
  let res, body;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error('network error on ' + path + ': ' + e.message);
  }
  try { body = await res.json(); } catch (e) { body = null; }
  if (!res.ok || (body && body.ok === false)) {
    const er = body && body.error;
    throw new Error(er ? (er.code + ': ' + er.message) : ('HTTP ' + res.status + ' ' + path));
  }
  return body;
}

/** POST JSON. @param {string} path @param {Object} obj @returns {Promise<Object>} */
function post(path, obj) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj || {})
  });
}

/** Number formatting for the inspector. @param {*} v @param {number} [d=2] */
function num(v, d) {
  if (v === null || v === undefined || (typeof v === 'number' && !isFinite(v))) return '--';
  if (typeof v !== 'number') return String(v);
  return v.toFixed(d === undefined ? 2 : d);
}

/** Fill a <dl class="kv"> from an array of [key, value] pairs.
 * @param {string} sel selector @param {Array[]} pairs */
function kv(sel, pairs) {
  const el = $(sel);
  let html = '';
  pairs.forEach(p => {
    html += '<dt>' + esc(p[0]) + '</dt><dd>' + (p[2] ? p[1] : esc(String(p[1]))) + '</dd>';
  });
  el.innerHTML = html;
}

/** HTML-escape. @param {string} s @returns {string} */
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Deep-set a value at a dotted path inside a plain object.
 * @param {Object} obj @param {string[]} path @param {*} val */
function deepSet(obj, path, val) {
  let o = obj;
  for (let i = 0; i < path.length - 1; i++) {
    if (typeof o[path[i]] !== 'object' || o[path[i]] === null) o[path[i]] = {};
    o = o[path[i]];
  }
  o[path[path.length - 1]] = val;
}

/* ============================ config form ============================== */

const formRegistry = [];   // {path:string[], input:HTMLElement, kind:string}

/** Enum label table for a weight-dict path, or null.
 * @param {string[]} path @returns {?Object<number,string>} */
function enumTableFor(path) {
  const p = path.join('.');
  if (p.indexOf('platform_mix') === 0 || p.indexOf('sensor.rcs_factor') === 0) return S.typeNames;
  if (p.indexOf('intent_mix') === 0) return S.intentNames;
  if (p.indexOf('encounter.geometry_weights') === 0) return S.geomNames;
  return null;
}

/** Build the editable configuration form.
 * @param {HTMLElement} host container @param {Object} obj config dict
 * @param {string[]} path current key path */
function buildForm(host, obj, path) {
  Object.keys(obj).forEach(k => {
    const v = obj[k];
    const p = path.concat([k]);
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      const sec = document.createElement('div');
      sec.className = 'section';
      const h = document.createElement('h2');
      h.textContent = k.replace(/_/g, ' ');
      const body = document.createElement('div');
      body.className = 'body';
      sec.appendChild(h); sec.appendChild(body);
      h.addEventListener('click', () => sec.classList.toggle('collapsed'));
      if (p.length > 1) sec.classList.add('collapsed');
      host.appendChild(sec);
      buildForm(body, v, p);
      return;
    }
    const row = document.createElement('div');
    row.className = 'row';
    const lab = document.createElement('label');
    const table = enumTableFor(path);
    lab.textContent = (table && table[k] !== undefined) ? table[k] : k.replace(/_/g, ' ');
    lab.title = p.join('.');
    const id = 'cfg_' + p.join('_');
    lab.htmlFor = id;
    let inp;
    let kind;
    if (typeof v === 'boolean') {
      inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = v;
      kind = 'bool';
    } else if (typeof v === 'number') {
      inp = document.createElement('input'); inp.type = 'number';
      inp.step = Number.isInteger(v) ? '1' : 'any';
      inp.value = String(v); kind = 'num';
    } else if (Array.isArray(v)) {
      inp = document.createElement('input'); inp.type = 'text';
      inp.value = JSON.stringify(v); kind = 'json';
    } else {
      inp = document.createElement('input'); inp.type = 'text';
      inp.value = v === null ? '' : String(v); kind = 'str';
    }
    inp.id = id;
    row.appendChild(lab); row.appendChild(inp);
    host.appendChild(row);
    formRegistry.push({ path: p, input: inp, kind: kind });
  });
}

/** Render the whole config form from a config dict. @param {Object} cfg */
function renderConfigForm(cfg) {
  const host = $('#cfgForm');
  host.innerHTML = '';
  formRegistry.length = 0;
  S.config = JSON.parse(JSON.stringify(cfg));
  buildForm(host, S.config, []);
}

/** Read the form back into a config dict. @returns {Object} */
function readConfigForm() {
  const out = JSON.parse(JSON.stringify(S.config || {}));
  formRegistry.forEach(f => {
    let v;
    if (f.kind === 'bool') v = f.input.checked;
    else if (f.kind === 'num') { v = parseFloat(f.input.value); if (!isFinite(v)) v = 0; }
    else if (f.kind === 'json') { try { v = JSON.parse(f.input.value); } catch (e) { v = []; } }
    else v = f.input.value;
    deepSet(out, f.path, v);
  });
  return out;
}

/* ============================== boot =================================== */

let renderer = null;

/** Load metadata, presets and defaults, then wire everything up. */
async function boot() {
  const cv = $('#view');
  renderer = new Renderer(cv);
  wireCanvas(cv);
  wireChrome();
  wireLive();
  wireBatch();
  wireReplay();
  wireDiag();
  window.addEventListener('resize', () => renderer.resize());
  // A window resize is not the only way the canvas box changes size, and it is
  // not even the common one: if the element has no layout when the Renderer is
  // constructed (hidden tab, zero-size viewport) getBoundingClientRect returns
  // 0, the backing store clamps to 2x2 and NOTHING ever recomputes it -- a
  // 2x2 buffer stretched over a 696x386 box, surviving New simulation and
  // Play. Observing the element itself covers every case; the window listener
  // stays as the fallback for browsers without ResizeObserver.
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => renderer.resize()).observe(cv);
  }

  try {
    const m = await api('/api/meta');
    S.meta = m;
    window.PRISM_CONST = m.constants || {};
    (m.platforms || []).forEach(p => { S.colors[p.value] = p.color; S.typeNames[p.value] = p.name; });
    (m.intents || []).forEach(p => { S.intentNames[p.value] = p.name; });
    (m.modes || []).forEach(p => { S.modeNames[p.value] = p.name; });
    (m.risks || []).forEach(p => { S.riskNames[p.value] = p.name; });
    (m.geometries || []).forEach(p => { S.geomNames[p.value] = p.name; });
    renderBridge(m.bridge || {});
    renderLegend();
    setStatus('ready');
  } catch (e) {
    setStatus('cannot reach /api/meta - ' + e.message, true);
  }

  try {
    const d = await api('/api/config/defaults');
    renderConfigForm(d.config);
    $('#cfgProblems').textContent = (d.problems && d.problems.length)
      ? ('config problems: ' + d.problems.join('; ')) : '';
  } catch (e) { setStatus('defaults: ' + e.message, true); }

  try { await loadPresets(); } catch (e) { setStatus('presets: ' + e.message, true); }
  try { await loadRuns(); } catch (e) { /* the Logs dir may simply be empty */ }

  requestAnimationFrame(loop);
  setInterval(updatePanels, 120);
}

/** Show the subsystem bridge table in the inspector. @param {Object} bridge */
function renderBridge(bridge) {
  const pairs = [];
  let missing = 0, total = 0;
  Object.keys(bridge).forEach(k => {
    total++;
    const b = bridge[k];
    const ok = b.imported && b.symbol;
    if (!ok) missing++;
    pairs.push([k.replace('simulation.', 'sim.').replace('storage.', 'st.')
      .replace('analysis.', 'an.').replace('scenario.', 'sc.'),
      '<span style="color:' + (ok ? 'var(--ok)' : 'var(--bad)') + '">' +
      (ok ? esc(b.symbol) : (b.imported ? 'no symbol' : 'not importable')) +
      '</span>', true]);
  });
  kv('#kvBridge', pairs);
  const chip = $('#chipBridge');
  chip.textContent = 'bridge: ' + (total - missing) + '/' + total;
  chip.className = 'chip ' + (missing === 0 ? 'ok' : (missing >= total ? 'bad' : ''));
  chip.title = missing ? 'missing subsystems disable the features that need them' : 'all subsystems present';
}

/** Draw the platform colour legend over the canvas. */
function renderLegend() {
  let html = '';
  Object.keys(S.typeNames).forEach(k => {
    html += '<div><span class="swatch" style="background:' + S.colorOf(+k) + '"></span>' +
      '<b>' + esc(S.typeNames[k]) + '</b></div>';
  });
  html += '<div style="margin-top:4px"><span class="swatch" style="background:#fff"></span>' +
    '<b>ownship</b></div>';
  $('#legend').innerHTML = html;
}

/* ============================ render loop ============================== */

/** requestAnimationFrame body: draw and measure fps. @param {number} ts */
function loop(ts) {
  renderer.resize();
  renderer.draw(S);
  S._fpsN++;
  if (ts - S._fpsT > 500) {
    S.fps = Math.round(S._fpsN * 1000 / (ts - S._fpsT));
    S._fpsT = ts; S._fpsN = 0;
    $('#chipFps').textContent = S.fps + ' fps';
  }
  requestAnimationFrame(loop);
}

/* ============================== frames ================================= */

/** Ingest a new frame into the state and the trail buffers. @param {Object} fr */
function onFrame(fr) {
  if (fr.error) setStatus('engine: ' + fr.error, true);
  S.frame = fr;
  if (fr.static) S.staticCtx = fr.static;
  if (fr.n_steps) S.nSteps = fr.n_steps;
  if (fr.playing !== undefined) S.playing = !!fr.playing;
  if (fr.speed) S.speed = fr.speed;
  accumulateTrails(fr);
  if (!S.fitted) fitToWorld();
}

/** Append the frame's positions to the trail buffers and prune by age.
 * @param {Object} fr */
function accumulateTrails(fr) {
  const t = fr.t || 0;
  const keep = Math.max(0, S.trailSeconds);
  (fr.agents || []).forEach(a => {
    if (a.type !== undefined && a.type !== null && a.type >= 0) S.types.set(a.id, a.type);
    const prev = S.lastTrackId.get(a.id);
    if (a.track_id !== undefined && a.track_id >= 0) {
      if (prev !== undefined && prev !== a.track_id) S.freshTracks.set(a.id, t);
      S.lastTrackId.set(a.id, a.track_id);
    } else if (a.track_id === -1) {
      S.lastTrackId.set(a.id, -1);
    }
    if (a.x === null || a.x === undefined || !isFinite(a.x)) return;
    let arr = S.trails.get(a.id);
    if (!arr) { arr = []; S.trails.set(a.id, arr); }
    const last = arr[arr.length - 1];
    if (!last || last[2] !== t) arr.push([a.x, a.y, t]);
    while (arr.length && (t - arr[0][2]) > keep) arr.shift();
  });
  const own = fr.own;
  if (own && isFinite(own.x)) {
    const last = S.ownTrail[S.ownTrail.length - 1];
    if (!last || last[2] !== t) S.ownTrail.push([own.x, own.y, t]);
    while (S.ownTrail.length && (t - S.ownTrail[0][2]) > keep) S.ownTrail.shift();
  }
}

/** Clear every trail/track buffer (on reset, new sim or replay load). */
function clearTrails() {
  S.trails.clear(); S.ownTrail.length = 0;
  S.freshTracks.clear(); S.lastTrackId.clear();
}

/** Fit the view to the world bounds of the current frame. */
function fitToWorld() {
  let b = S.staticCtx && S.staticCtx.bounds;
  if (!b && S.replay && S.replay.bounds) b = S.replay.bounds;
  if (!b && S.frame && S.frame.own) {
    const o = S.frame.own;
    b = [o.x - 2500, o.y - 2500, o.x + 2500, o.y + 2500];
  }
  if (!b) return;
  renderer.fit(b, S.view);
  S.fitted = true;
}

/* ============================ SSE stream =============================== */

/** Open the Server-Sent-Events frame stream for a live session.
 * @param {string} simId */
function openStream(simId) {
  closeStream();
  S.esErrors = 0;
  const es = new EventSource('/api/sim/stream?sim_id=' + encodeURIComponent(simId));
  es.addEventListener('frame', ev => {
    S.esErrors = 0;
    try { onFrame(JSON.parse(ev.data)); } catch (e) { /* malformed chunk */ }
  });
  es.addEventListener('gone', () => {
    setStatus('simulation session ended');
    closeStream();
  });
  es.onopen = () => { $('#chipStream').textContent = 'stream: live'; S.esErrors = 0; };
  es.onerror = () => {
    S.esErrors++;
    /* A non-200 or wrong-MIME response is fatal for EventSource: readyState
       goes to CLOSED and the browser will NOT retry. Say so instead of
       showing "reconnecting" forever. */
    if (es.readyState === EventSource.CLOSED) {
      closeStream();
      setStatus('frame stream refused by the server - the session is gone; ' +
        'press "New simulation"', true);
      return;
    }
    $('#chipStream').textContent = 'stream: reconnecting (' + S.esErrors + ')';
    if (S.esErrors > 8) {
      closeStream();
      setStatus('frame stream gave up after 8 failed reconnects', true);
    }
  };
  S.es = es;
}

/** Close the SSE stream if one is open. */
function closeStream() {
  if (S.es) { try { S.es.close(); } catch (e) { /* already closed */ } }
  S.es = null;
  $('#chipStream').textContent = 'stream: idle';
}

/* ============================ canvas input ============================= */

/** Attach pan / zoom / hover / select handlers. @param {HTMLCanvasElement} cv */
function wireCanvas(cv) {
  let dragging = false, lastX = 0, lastY = 0, moved = 0;

  cv.addEventListener('mousedown', e => {
    dragging = true; moved = 0;
    lastX = e.offsetX; lastY = e.offsetY;
    cv.classList.add('dragging');
  });
  window.addEventListener('mouseup', e => {
    if (!dragging) return;
    dragging = false; cv.classList.remove('dragging');
    if (moved < 4) {
      const r = cv.getBoundingClientRect();
      const id = renderer.hitTest(e.clientX - r.left, e.clientY - r.top, S);
      S.selected = id;
      if (id === null) $('#selHint').textContent = 'Click an agent on the canvas.';
    }
  });
  window.addEventListener('mousemove', e => {
    const r = cv.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (dragging) {
      const dx = sx - lastX, dy = sy - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      if (S.view.follow) { S.view.follow = false; syncViewButtons(); }
      renderer.panBy(dx, dy, S.view);
      lastX = sx; lastY = sy;
      return;
    }
    if (sx < 0 || sy < 0 || sx > renderer.W || sy > renderer.H) {
      renderer.hoverId = null; return;
    }
    renderer.hoverId = renderer.hitTest(sx, sy, S);
    const w = renderer.s2w(sx, sy);
    $('#statusView').textContent =
      'cursor ' + w[0].toFixed(0) + ' E, ' + w[1].toFixed(0) + ' N m   |   ' +
      'scale ' + (1 / S.view.scale).toFixed(1) + ' m/px' +
      (S.view.headingUp ? '   |   heading-up' : '   |   north-up');
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const f = Math.exp(-e.deltaY * 0.0016);
    renderer.zoomAt(e.clientX - r.left, e.clientY - r.top, f, S.view);
  }, { passive: false });
  cv.addEventListener('dblclick', () => fitToWorld());
}

/** Reflect the view flags on their buttons. */
function syncViewButtons() {
  $('#btnFollow').classList.toggle('on', !!S.view.follow);
  $('#btnHeadingUp').classList.toggle('on', !!S.view.headingUp);
  $$('#layerToggles button').forEach(b => {
    b.classList.toggle('on', !!S.layers[b.dataset.layer]);
  });
}

/* ============================== chrome ================================= */

/** Tabs, collapsible sections, layer toggles, keyboard shortcuts. */
function wireChrome() {
  $$('#tabs button').forEach(b => {
    b.addEventListener('click', () => {
      $$('#tabs button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      $$('#drawer .panel').forEach(p => p.classList.remove('active'));
      $('#panel-' + b.dataset.tab).classList.add('active');
      S.tab = b.dataset.tab;
      if (S.tab === 'replay' || S.tab === 'diag') loadRuns().catch(() => {});
      if (S.tab === 'diag' && S.diag) drawDiag();
    });
  });

  $$('aside .section > h2').forEach(h => {
    h.addEventListener('click', () => h.parentElement.classList.toggle('collapsed'));
  });

  $$('#layerToggles button').forEach(b => {
    b.addEventListener('click', () => {
      S.layers[b.dataset.layer] = S.layers[b.dataset.layer] ? 0 : 1;
      syncViewButtons();
    });
  });

  $('#btnFollow').addEventListener('click', () => {
    S.view.follow = !S.view.follow; syncViewButtons();
  });
  $('#btnHeadingUp').addEventListener('click', () => {
    S.view.headingUp = !S.view.headingUp; syncViewButtons();
  });
  $('#btnFit').addEventListener('click', () => fitToWorld());
  $('#trailLen').addEventListener('change', e => {
    S.trailSeconds = Math.max(0, parseFloat(e.target.value) || 0);
  });

  $('#btnHelp').addEventListener('click', () => $('#help').classList.toggle('show'));
  $('#btnHelpClose').addEventListener('click', () => $('#help').classList.remove('show'));

  $('#btnLoadDefaults').addEventListener('click', async () => {
    try {
      const d = await api('/api/config/defaults');
      renderConfigForm(d.config);
      setStatus('loaded default configuration');
    } catch (e) { setStatus(e.message, true); }
  });
  $('#btnLoadPreset').addEventListener('click', () => {
    const name = $('#presetSel').value;
    const p = S.presets.find(x => x.name === name);
    if (!p) { setStatus('preset not found: ' + name, true); return; }
    renderConfigForm(p.config);
    setStatus('loaded preset "' + name + '"');
  });
  $('#btnNewSim').addEventListener('click', newSim);

  document.addEventListener('keydown', ev => {
    const tag = (ev.target && ev.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    let handled = true;
    switch (ev.key) {
      case ' ': togglePlay(); break;
      case 'ArrowRight': stepBy(1); break;
      case 'ArrowLeft': stepBy(-1); break;
      case 'ArrowUp': changeSpeed(2); break;
      case 'ArrowDown': changeSpeed(0.5); break;
      case 'r': case 'R': resetSim(); break;
      case 'g': case 'G': S.layers.gt = S.layers.gt ? 0 : 1; syncViewButtons(); break;
      case 'o': case 'O': S.layers.obs = S.layers.obs ? 0 : 1; syncViewButtons(); break;
      case 'e': case 'E': S.layers.ell = S.layers.ell ? 0 : 1; syncViewButtons(); break;
      case 'f': case 'F': S.layers.fan = S.layers.fan ? 0 : 1; syncViewButtons(); break;
      case 'c': case 'C': S.view.follow = !S.view.follow; syncViewButtons(); break;
      case 'h': case 'H': S.view.headingUp = !S.view.headingUp; syncViewButtons(); break;
      case '0': fitToWorld(); break;
      case '?': case '/': $('#help').classList.toggle('show'); break;
      case 'Escape': $('#help').classList.remove('show'); break;
      default: handled = false;
    }
    if (handled) ev.preventDefault();
  });
  syncViewButtons();
}

/* ============================ live control ============================= */

/** Load the preset list into both preset selects. */
async function loadPresets() {
  const d = await api('/api/presets');
  S.presets = d.presets || [];
  ['#presetSel', '#genPreset'].forEach(sel => {
    const el = $(sel);
    el.innerHTML = '';
    if (sel === '#genPreset') {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(use left config panel)';
      el.appendChild(o);
    }
    S.presets.forEach(p => {
      const o = document.createElement('option');
      o.value = p.name; o.textContent = p.name + '  [' + p.source + ']';
      el.appendChild(o);
    });
  });
}

/** Create a new live simulation from the left-panel configuration. */
async function newSim() {
  try {
    setStatus('creating simulation...');
    const cfg = readConfigForm();
    const d = await post('/api/sim/new', { config: cfg });
    S.sim_id = d.sim_id;
    S.fan = null; S.selected = null; S.fitted = false;
    clearTrails();
    onFrame(d.frame0);
    $('#chipSim').textContent = 'sim ' + d.sim_id;
    openStream(d.sim_id);
    S.mode = 'live';
    setStatus('simulation ready - press play (space)');
  } catch (e) {
    setStatus('new sim failed - ' + e.message, true);
  }
}

/** Send a transport command to the live session.
 * @param {string} action @param {number} [speed] */
async function control(action, speed) {
  if (!S.sim_id) { setStatus('no live simulation', true); return; }
  try {
    const d = await post('/api/sim/control',
      { sim_id: S.sim_id, action: action, speed: speed });
    S.playing = !!d.playing; S.speed = d.speed;
    if (d.error) setStatus('engine: ' + d.error, true);
    if (d.frame && action === 'reset') { clearTrails(); onFrame(d.frame); }
  } catch (e) { setStatus(e.message, true); }
}

/** Toggle play/pause on whichever timeline is active. */
function togglePlay() {
  if (S.mode === 'replay') { replayToggle(); return; }
  control('toggle');
}

/** Step the active timeline. @param {number} dir +1 forward, -1 backward */
function stepBy(dir) {
  if (S.mode === 'replay') {
    setReplayIndex(S.replayIdx + dir);
    return;
  }
  if (dir > 0) control('step');
  else setStatus('the live engine cannot step backwards - use Reset, or the Replay tab');
}

/** Multiply the playback speed. @param {number} f */
function changeSpeed(f) {
  if (S.mode === 'replay') return;
  const v = Math.max(0.05, Math.min(50, (S.speed || 1) * f));
  $('#speedSel').value = String(v);
  control('speed', v);
}

/** Reset the live simulation to t = 0. */
function resetSim() {
  if (S.mode === 'replay') { setReplayIndex(0); return; }
  S.fan = null; clearTrails();
  control('reset');
}

/** Wire the Live drawer panel. */
function wireLive() {
  $('#btnPlay').addEventListener('click', togglePlay);
  $('#btnStepF').addEventListener('click', () => stepBy(1));
  $('#btnStepB').addEventListener('click', () => stepBy(-1));
  $('#btnReset').addEventListener('click', resetSim);
  $('#speedSel').addEventListener('change', e => control('speed', parseFloat(e.target.value)));
  $('#btnFanClear').addEventListener('click', () => { S.fan = null; });
  $('#btnFan').addEventListener('click', async () => {
    if (!S.sim_id) { setStatus('no live simulation', true); return; }
    try {
      setStatus('rolling out branches...');
      const d = await post('/api/rollout', {
        sim_id: S.sim_id,
        n_branches: parseInt($('#fanN').value, 10) || 16,
        horizon_s: parseFloat($('#fanH').value) || 15
      });
      S.fan = d; S.layers.fan = 1; syncViewButtons();
      const nb = (d.agents && d.agents[0] && d.agents[0].branches.length) || 0;
      const spread = (d.spread_m === null || d.spread_m === undefined)
        ? '--' : d.spread_m.toFixed(1) + ' m';
      setStatus('rollout fan: ' + (d.agents ? d.agents.length : 0) +
        ' agents x ' + nb + ' branches over ' + d.horizon_s + ' s' +
        '   |   branch spread ' + spread +
        (d.warning ? ('   |   ' + d.warning) : ''), !!d.warning);
      $('#fanHint').textContent = d.warning ||
        ('Mean std-dev of the branch end points is ' + spread +
         '. A spread near zero means the futures are deterministic and a CV ' +
         'extrapolator would win (doc 3.3).');
    } catch (e) { setStatus('rollout failed - ' + e.message, true); }
  });
}

/* ============================== panels ================================= */

/** Refresh the inspector, the live table and the transport labels (10 Hz). */
function updatePanels() {
  const fr = S.frame;
  $('#btnPlay').textContent = (S.mode === 'replay')
    ? (S.replayPlaying ? 'Pause' : 'Play')
    : (S.playing ? 'Pause' : 'Play');

  if (!fr) { return; }
  const own = fr.own || {};
  kv('#kvOwn', [
    ['t [s]', num(fr.t, 2)],
    ['x [m]', num(own.x, 1)],
    ['y [m]', num(own.y, 1)],
    ['v [m/s]', num(own.v, 2)],
    ['psi', prismUtil.fmtDeg(own.psi)],
    ['vx,vy', num(own.vx, 1) + ', ' + num(own.vy, 1)],
    ['omega', own.omega === null || own.omega === undefined
      ? '--' : prismUtil.fmtDeg(own.omega) + '/s']
  ]);

  const w = fr.wind || {};
  const st = S.staticCtx || {};
  kv('#kvEnv', [
    ['wind [m/s]', num(w.speed, 2)],
    ['wind dir', prismUtil.fmtDeg(w.dir)],
    ['agents', String((fr.agents || []).length)],
    ['detected', String((fr.agents || []).filter(a => a.det).length)],
    ['corridors', String((st.corridors || []).length)],
    ['no-fly zones', String((st.nfz || []).length)],
    ['bounds [m]', st.bounds ? st.bounds.map(v => v.toFixed(0)).join(', ') : '--']
  ]);

  kv('#kvSim', [
    ['mode', S.mode],
    ['sim id', S.sim_id || '--'],
    ['step', (fr.step || 0) + (S.nSteps ? ' / ' + S.nSteps : '')],
    ['speed', 'x' + num(S.speed, 2)],
    ['fan', S.fan ? ((S.fan.agents || []).length + ' agents') : 'none']
  ]);

  updateSelected();
  updateAgentTable();
}

/** Fill the selected-agent inspector block. */
function updateSelected() {
  const fr = S.frame;
  if (!fr || S.selected === null || S.selected === undefined) {
    kv('#kvSel', []);
    return;
  }
  const a = (fr.agents || []).find(x => x.id === S.selected);
  if (!a) { kv('#kvSel', []); $('#selHint').textContent = 'agent left the scene'; return; }
  const g = prismUtil.cpaOf(fr.own, a);
  const omegaMax = (a.omega_max === null || a.omega_max === undefined)
    ? null : a.omega_max;
  const util = (omegaMax && a.omega !== null && a.omega !== undefined && omegaMax > 0)
    ? (Math.abs(a.omega) / omegaMax * 100).toFixed(0) + ' %' : '--';
  $('#selHint').textContent = '';
  kv('#kvSel', [
    ['id', '<span class="swatch" style="background:' + S.colorOf(a.type) + '"></span>#' + a.id, true],
    ['platform', S.typeNames[a.type] || a.type],
    ['intent', S.intentNames[a.intent] || (a.intent >= 0 ? a.intent : '--')],
    ['mode', S.modeNames[a.mode] || (a.mode >= 0 ? a.mode : '--')],
    ['dwell left', a.mode_timer === null || a.mode_timer === undefined
      ? '--' : num(a.mode_timer, 2) + ' s'],
    ['v [m/s]', num(a.v, 2)],
    ['psi', prismUtil.fmtDeg(a.psi)],
    ['omega', a.omega === null || a.omega === undefined
      ? '--' : prismUtil.fmtDeg(a.omega) + '/s'],
    ['omega max', omegaMax === null ? '--' : prismUtil.fmtDeg(omegaMax) + '/s'],
    ['omega used', util],
    ['track id', a.track_id === undefined ? '--' : String(a.track_id)],
    ['track age', a.track_age === undefined ? '--' : String(a.track_age)],
    ['detected', a.det ? 'yes' : 'NO (coasting)'],
    ['range', prismUtil.fmtM(g.range)],
    ['bearing', prismUtil.fmtDeg(g.bearing)],
    ['rel bearing', prismUtil.fmtDeg(g.relBearing)],
    ['CPA', g.cpa === null ? '--' : prismUtil.fmtM(g.cpa)],
    ['TCPA', g.tcpa === null ? '--' : num(g.tcpa, 1) + ' s'],
    ['sigma major', a.R ? num(prismUtil.eig2(a.R[0][0], a.R[0][1], a.R[1][1])[0], 1) + ' m' : '--'],
    ['sigma minor', a.R ? num(prismUtil.eig2(a.R[0][0], a.R[0][1], a.R[1][1])[1], 1) + ' m' : '--']
  ].concat(extraSelectedRows(a)));
}

/** Optional inspector rows that only exist when the engine supplies them.
 * @param {Object} a agent record of the current frame
 * @returns {Array[]} extra [key, value] pairs, possibly empty. */
function extraSelectedRows(a) {
  const rows = [];
  if (a.v_min !== undefined && a.v_min !== null) {
    rows.push(['v envelope', num(a.v_min, 1) + ' .. ' + num(a.v_max, 1) + ' m/s']);
  }
  if (a.can_hover !== undefined) rows.push(['can hover', a.can_hover ? 'yes' : 'no']);
  if (a.engine_cpa_m !== undefined && a.engine_cpa_m !== null) {
    rows.push(['CPA (engine)', prismUtil.fmtM(a.engine_cpa_m)]);
    rows.push(['TCPA (engine)', num(a.engine_tcpa_s, 1) + ' s']);
  }
  if (a.designed) rows.push(['designed', 'yes (inverse-designed encounter)']);
  if (a.ood) rows.push(['OOD', 'yes']);
  return rows;
}

/** Rebuild the live agent table, sorted by time to CPA. */
function updateAgentTable() {
  const fr = S.frame;
  const tb = $('#tblAgents tbody');
  if (!fr) { tb.innerHTML = ''; return; }
  const rows = (fr.agents || []).map(a => {
    const g = prismUtil.cpaOf(fr.own, a);
    return { a: a, g: g };
  });
  rows.sort((p, q) => {
    const tp = (p.g.tcpa === null || p.g.tcpa < 0) ? 1e9 : p.g.tcpa;
    const tq = (q.g.tcpa === null || q.g.tcpa < 0) ? 1e9 : q.g.tcpa;
    return tp - tq;
  });
  const nmac = (window.PRISM_CONST && window.PRISM_CONST.NMAC_RADIUS_M) || 152.4;
  let html = '';
  rows.forEach(r => {
    const a = r.a, g = r.g;
    const sel = (a.id === S.selected) ? ' class="sel"' : '';
    const danger = (g.cpa !== null && g.cpa < nmac) ? ' style="color:var(--bad)"' : '';
    html += '<tr' + sel + ' data-id="' + a.id + '">' +
      '<td class="l"><span class="swatch" style="background:' + S.colorOf(a.type) + '"></span>' + a.id + '</td>' +
      '<td class="l">' + esc((S.typeNames[a.type] || '?').slice(0, 12)) + '</td>' +
      '<td class="l">' + esc((S.intentNames[a.intent] || '--').slice(0, 12)) + '</td>' +
      '<td class="l">' + esc(S.modeNames[a.mode] || '--') + '</td>' +
      '<td>' + num(a.v, 1) + '</td>' +
      '<td>' + (isFinite(g.range) ? g.range.toFixed(0) : '--') + '</td>' +
      '<td>' + (g.tcpa === null ? '--' : g.tcpa.toFixed(1)) + '</td>' +
      '<td' + danger + '>' + (g.cpa === null ? '--' : g.cpa.toFixed(0)) + '</td>' +
      '<td>' + (a.det ? (a.track_id >= 0 ? a.track_id : '--') : 'x') + '</td>' +
      '</tr>';
  });
  if (tb.dataset.sig !== html) { tb.innerHTML = html; tb.dataset.sig = html; }
  Array.prototype.forEach.call(tb.querySelectorAll('tr'), tr => {
    tr.onclick = () => { S.selected = parseInt(tr.dataset.id, 10); };
  });
}

/* =============================== batch ================================= */

/** Wire the Batch drawer panel. */
function wireBatch() {
  $('#btnGenUseCfg').addEventListener('click', () => {
    $('#genPreset').value = '';
    setStatus('batch will use the left-panel configuration');
  });
  $('#btnGen').addEventListener('click', async () => {
    try {
      const body = {
        n_episodes: parseInt($('#genN').value, 10) || 1,
        tag: $('#genTag').value || 'run',
        seed: parseInt($('#genSeed').value, 10) || 0
      };
      const presetName = $('#genPreset').value;
      const useCfg = !presetName;
      if (useCfg) body.config = readConfigForm();
      else {
        const p = S.presets.find(x => x.name === presetName);
        body.config = p ? p.config : readConfigForm();
      }
      body.config = JSON.parse(JSON.stringify(body.config));
      body.config.seed = body.seed;
      const d = await post('/api/generate', body);
      S.job = d.job_id;
      $('#btnGen').disabled = true;
      $('#btnGenCancel').disabled = false;
      $('#genPathWrap').style.display = 'none';
      setStatus('generation job ' + d.job_id + ' started');
      pollJob();
    } catch (e) { setStatus('generate failed - ' + e.message, true); }
  });
  $('#btnGenCancel').addEventListener('click', async () => {
    if (!S.job) return;
    try { await post('/api/jobs/' + S.job + '/cancel', {}); }
    catch (e) { setStatus(e.message, true); }
  });
  $('#genPath').addEventListener('click', e => e.target.select());
}

/** Poll the active job until it terminates. */
function pollJob() {
  if (S.jobTimer) clearInterval(S.jobTimer);
  S.jobTimer = setInterval(async () => {
    if (!S.job) return;
    try {
      const j = await api('/api/jobs/' + S.job);
      $('#genBar').style.width = (100 * (j.progress || 0)).toFixed(1) + '%';
      $('#genLog').textContent = (j.log || []).join('\n');
      $('#genLog').scrollTop = $('#genLog').scrollHeight;
      kv('#kvJob', [
        ['state', j.state],
        ['episodes', j.done + ' / ' + j.n_episodes],
        ['elapsed', num(j.elapsed_s, 1) + ' s'],
        ['eta', j.eta_s === null || j.eta_s === undefined ? '--' : num(j.eta_s, 1) + ' s'],
        ['rate', num(j.rate_eps, 2) + ' ep/s'],
        ['message', (j.message || '').slice(0, 60)]
      ]);
      if (['done', 'error', 'cancelled'].indexOf(j.state) >= 0) {
        clearInterval(S.jobTimer); S.jobTimer = 0;
        $('#btnGen').disabled = false;
        $('#btnGenCancel').disabled = true;
        if (j.run_path) {
          $('#genPathWrap').style.display = '';
          $('#genPath').value = j.run_path;
        }
        setStatus('job ' + j.state + ': ' + (j.message || ''), j.state === 'error');
        loadRuns().catch(() => {});
      }
    } catch (e) {
      clearInterval(S.jobTimer); S.jobTimer = 0;
      setStatus('job polling failed - ' + e.message, true);
      $('#btnGen').disabled = false;
    }
  }, 500);
}

/* =============================== replay ================================ */

/** Refresh the run list shared by the Replay and Diagnostics tabs. */
async function loadRuns() {
  const d = await api('/api/runs');
  S.runs = d.runs || [];
  ['#runSel', '#diagRun'].forEach(sel => {
    const el = $(sel);
    const prev = el.value;
    el.innerHTML = '';
    S.runs.forEach(r => {
      const o = document.createElement('option');
      o.value = r.name;
      o.textContent = r.name + '  (' + r.n_episode_files + ' ep, ' + r.modified_str + ')';
      el.appendChild(o);
    });
    if (prev) el.value = prev;
    // Assigning a value that is not among the options DESELECTS the <select>
    // (selectedIndex -1, value ""), which happens whenever the previously
    // selected run disappears from Logs between refreshes -- and the list
    // auto-refreshes on every tab switch and after every generate job. The
    // empty value then reaches the API as '/api/runs//episode/0' (404
    // "no GET route") or {"run":""} (400 "illegal run name"), so "Load
    // episode" appears to do nothing and reports a routing error the operator
    // cannot act on. Fall back to the first run.
    if (el.selectedIndex < 0 && el.options.length) el.selectedIndex = 0;
  });
  $('#tabNote').textContent = S.runs.length
    ? (S.runs.length + ' runs under ' + d.root)
    : ('no runs yet under ' + d.root);
}

/** Wire the Replay drawer panel. */
function wireReplay() {
  $('#btnRunsRefresh').addEventListener('click', () =>
    loadRuns().then(() => setStatus('run list refreshed')).catch(e => setStatus(e.message, true)));
  $('#btnRunIndex').addEventListener('click', loadRunIndex);
  $('#btnRepLoad').addEventListener('click', () =>
    loadEpisode($('#runSel').value, parseInt($('#repIdx').value, 10) || 0));
  $('#btnRepPlay').addEventListener('click', replayToggle);
  $('#repScrub').addEventListener('input', e =>
    setReplayIndex(parseInt(e.target.value, 10) || 0));
  $$('#tblRuns th').forEach(th => {
    th.addEventListener('click', () => {
      const k = th.dataset.k;
      if (S.sortKey === k) S.sortDir = -S.sortDir; else { S.sortKey = k; S.sortDir = 1; }
      renderRunIndex();
    });
  });
}

/** Load the stratified episode index of the selected run. */
async function loadRunIndex() {
  const run = $('#runSel').value;
  if (!run) { setStatus('no run selected', true); return; }
  try {
    setStatus('loading index of ' + run + '...');
    S.runIndex = await api('/api/runs/' + encodeURIComponent(run) + '/index');
    renderRunIndex();
    setStatus('index: ' + S.runIndex.episodes.length + ' episodes (source: ' +
      S.runIndex.source + ')');
  } catch (e) { setStatus('index failed - ' + e.message, true); }
}

/** Render (and sort) the episode index table. */
function renderRunIndex() {
  const tb = $('#tblRuns tbody');
  if (!S.runIndex) { tb.innerHTML = ''; return; }
  const rows = S.runIndex.episodes.slice();
  if (S.sortKey) {
    rows.sort((a, b) => {
      let x = a[S.sortKey], y = b[S.sortKey];
      if (Array.isArray(x)) x = x.join(',');
      if (Array.isArray(y)) y = y.join(',');
      if (x === null || x === undefined) x = -Infinity;
      if (y === null || y === undefined) y = -Infinity;
      if (typeof x === 'boolean') x = x ? 1 : 0;
      if (typeof y === 'boolean') y = y ? 1 : 0;
      return (x > y ? 1 : (x < y ? -1 : 0)) * S.sortDir;
    });
  }
  let html = '';
  rows.forEach(r => {
    html += '<tr data-row="' + r.episode_index + '">' +
      '<td>' + r.episode_index + '</td>' +
      '<td>' + (r.n_agents !== undefined ? r.n_agents : '--') + '</td>' +
      '<td>' + num(r.density, 0) + '</td>' +
      '<td>' + num(r.wind_speed, 1) + '</td>' +
      '<td>' + (r.cpa_min === null || r.cpa_min === undefined ? '--' : num(r.cpa_min, 0)) + '</td>' +
      '<td>' + (r.tcpa_first === null || r.tcpa_first === undefined ? '--' : num(r.tcpa_first, 1)) + '</td>' +
      '<td>' + (r.min_separation_m === null || r.min_separation_m === undefined
        ? '--' : num(r.min_separation_m, 0)) + '</td>' +
      '<td class="l">' + esc(r.risk_name || '--') + '</td>' +
      '<td class="l">' + esc((r.geometry_names || []).join('/') || '--') + '</td>' +
      '<td>' + (r.nmac_occurred ? '<span style="color:var(--bad)">Y</span>' : '.') + '</td>' +
      '<td>' + (r.dwc_violated ? '<span style="color:var(--warn)">Y</span>' : '.') + '</td>' +
      '</tr>';
  });
  tb.innerHTML = html;
  Array.prototype.forEach.call(tb.querySelectorAll('tr'), tr => {
    tr.onclick = () => {
      Array.prototype.forEach.call(tb.querySelectorAll('tr'), x => x.classList.remove('sel'));
      tr.classList.add('sel');
      const i = parseInt(tr.dataset.row, 10);
      $('#repIdx').value = String(i);
      loadEpisode(S.runIndex.run, i);
    };
  });
}

/** Load one stored episode into the replay timeline.
 * @param {string} run @param {number} idx */
async function loadEpisode(run, idx) {
  if (!run) { setStatus('no run selected', true); return; }
  try {
    setStatus('loading ' + run + ' episode ' + idx + '...');
    const d = await api('/api/runs/' + encodeURIComponent(run) + '/episode/' + idx);
    S.replay = d;
    S.mode = 'replay';
    replayStop();
    closeStream();
    S.staticCtx = { bounds: d.bounds, corridors: [], nfz: [] };
    S.fitted = false;
    (d.agent_ids || []).forEach((aid, j) => {
      if (aid >= 0) S.types.set(aid, (d.gt_type || [])[j]);
    });
    $('#repScrub').max = String(Math.max(0, d.frames.length - 1));
    setReplayIndex(0);
    fitToWorld();
    const m = d.meta || {};
    kv('#kvReplay', [
      ['run', run],
      ['episode', String(idx)],
      ['steps', String(d.n_steps)],
      ['slots', String(d.n_agents)],
      ['risk', m.risk_name || '--'],
      ['min sep', m.min_separation_m === null || m.min_separation_m === undefined
        ? '--' : num(m.min_separation_m, 0) + ' m'],
      ['nmac', m.nmac_occurred ? 'YES' : 'no'],
      ['dwc', m.dwc_violated ? 'YES' : 'no'],
      ['wind', num(m.wind_speed, 1) + ' m/s @ ' + prismUtil.fmtDeg(m.wind_dir || 0)],
      ['file', (d.file || '').split('/').pop()]
    ]);
    setStatus('replay loaded: ' + d.frames.length + ' frames');
  } catch (e) { setStatus('episode load failed - ' + e.message, true); }
}

/** Show a given replay frame index, rebuilding the trails. @param {number} i */
function setReplayIndex(i) {
  if (!S.replay) return;
  const n = S.replay.frames.length;
  const idx = Math.max(0, Math.min(n - 1, i));
  S.replayIdx = idx;
  $('#repScrub').value = String(idx);
  clearTrails();
  const dt = S.replay.dt || 0.5;
  const back = Math.max(0, idx - Math.ceil(S.trailSeconds / dt));
  for (let k = back; k <= idx; k++) accumulateTrails(S.replay.frames[k]);
  const fr = JSON.parse(JSON.stringify(S.replay.frames[idx]));
  fr.n_steps = n;
  fr.step = idx;
  S.frame = fr;
  S.nSteps = n;
}

/** Toggle replay playback. */
function replayToggle() {
  if (S.replayPlaying) { replayStop(); return; }
  if (!S.replay) { setStatus('load an episode first', true); return; }
  S.replayPlaying = true;
  const dt = (S.replay.dt || 0.5) * 1000;
  S.replayTimer = setInterval(() => {
    if (S.replayIdx >= S.replay.frames.length - 1) { replayStop(); return; }
    setReplayIndex(S.replayIdx + 1);
  }, Math.max(30, dt));
}

/** Stop replay playback. */
function replayStop() {
  S.replayPlaying = false;
  if (S.replayTimer) { clearInterval(S.replayTimer); S.replayTimer = 0; }
}

/* ============================ diagnostics ============================== */

/** Wire the Diagnostics drawer panel. */
function wireDiag() {
  $('#btnDiag').addEventListener('click', async () => {
    const run = $('#diagRun').value;
    if (!run) { setStatus('no run selected', true); return; }
    try {
      setStatus('running CV baseline over ' + run + ' (this reads every episode)...');
      $('#btnDiag').disabled = true;
      S.diag = await post('/api/diagnose', { run: run });
      drawDiag();
      setStatus('CV baseline verdict: ' + S.diag.verdict);
    } catch (e) {
      setStatus('diagnose failed - ' + e.message, true);
    } finally { $('#btnDiag').disabled = false; }
  });
  window.addEventListener('resize', () => { if (S.diag) drawDiag(); });
}

/** Render the diagnosis verdict, tables and canvas charts. */
function drawDiag() {
  const d = S.diag;
  if (!d) return;
  const v = $('#diagVerdict');
  v.textContent = d.verdict || '--';
  v.className = 'verdict ' + (d.verdict || 'NO_DATA');
  const nmac = d.nmac_radius_m || 152.4;
  kv('#kvDiag', [
    ['run', d.run || '--'],
    ['source', d.source || '--'],
    ['episodes', String(d.n_episodes_scanned !== undefined ? d.n_episodes_scanned : '--')],
    ['samples', String(d.n_samples !== undefined ? d.n_samples : '--')],
    ['median FDE 15 s', d.median_fde_15s_m === null || d.median_fde_15s_m === undefined
      ? '--' : num(d.median_fde_15s_m, 1) + ' m'],
    ['NMAC radius', num(nmac, 1) + ' m'],
    ['ratio to NMAC', d.ratio_to_nmac === null || d.ratio_to_nmac === undefined
      ? '--' : num(d.ratio_to_nmac, 2)],
    ['OK band', d.bands ? (d.bands.too_easy_below + ' .. ' + d.bands.too_random_above) : '--']
  ]);
  $('#diagNote').textContent = (d.note || '') +
    (d.module_error ? ('  [analysis.cv_baseline raised ' + d.module_error + ']') : '');
  Charts.histograms($('#diagHist'), d.horizons || [], nmac);
  Charts.breakdown($('#diagBreak'),
    { platform: d.by_platform || {}, intent: d.by_intent || {} }, nmac);
}

/* =============================== start ================================= */

window.addEventListener('DOMContentLoaded', boot);
