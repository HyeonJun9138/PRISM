# PRISM `dataGen`

**2D traffic simulator and trajectory-prediction dataset generator.**

This package supplies the simulator and observation tracks used by PRISM:

```
2D simulator -> multi-branch prediction -> time-indexed risk -> fixed Frenet planner
^^^^^^^^^^^^
this program
```

The revised paper studies observation-based forecasting and planning for
**non-cooperative** aerial traffic. The predictor uses current sensor tracks,
not prior object-type, identity or intent labels. The experiments model
horizontal motion; they are not real-traffic or 3D flight validation.
This program builds the simulator and the dataset. It does not train anything.

---

## Simulation scope

Type-specific motion, maneuver and reaction rules create a range of futures
from shared observed histories. These are explicit simulator assumptions,
not a claim of complete real-world traffic coverage. The CV diagnostic
characterizes dataset difficulty before predictor training; it does not
establish comparative performance or safety on its own.

---

## Requirements

| | |
|---|---|
| OS | Windows |
| Python | 3.10.9 — `python` |
| Dependencies | **numpy only** (2.2.6), plus the Python standard library |
| Explicitly not used | pandas, torch, scipy, matplotlib, flask, fastapi, pyarrow |

The HTTP server is stdlib `http.server`. The GUI is plain Canvas 2D with **zero**
external JS/CSS — no CDN, no build step.

---

## Layout

Only `main.py` sits at the top level. Everything else is a function-named
subpackage.

```
$PRISM_ROOT/dataGen/
  main.py           <- the only top-level .py file; CLI entry point
  core/             types, constants, rng, state, kinematics, integrator
  agents/           base, platforms, intent, maneuver, behavior
  environment/      wind, world, encounter
  sensors/          model, tracker
  scenario/         config, presets, builder
  simulation/       engine, episode, rollout
  storage/          schema, writer, manifest      (named 'storage', not 'io')
  analysis/         cv_baseline, metrics
  gui/              server, api, static/{index.html,app.js,render.js,style.css}
  config/           defaults.json, presets/*.json
  tests/            test_*.py  (standalone runners, no pytest)
```

### Which doc section each subpackage implements

| subpackage | doc | role |
|---|---|---|
| `core/` | 3.7, 9.4 | Units, enums, RNG, state, kinematics. The shared contract every other package imports. The integrator enforces the doc 3.3(a) "only as much as physically possible" constraint. |
| `agents/` | 3.2, 3.3 | Objects as *purpose × character × performance*. `platforms.py` = the 3.2(3) performance table; `intent.py` = the 3.2(1) purpose types; `maneuver.py` + `behavior.py` = the 3.3 stochastic decision rules that **create** the multimodality, including the 3.3(b) manoeuvre-dwell constraint. |
| `environment/` | 3.4, 3.6 | `wind.py` = steady wind (the systematic CV bias and the physical origin of aleatoric uncertainty); `encounter.py` = **inverse-designed** encounter geometry — sample `d*`/`t*`, then solve for the initial state; `world.py` = corridors and no-fly zones. |
| `sensors/` | 3.5 | `Pd < 1`, polar noise → Cartesian, asynchronous update rates, latency + jitter, track drop and re-acquisition. Misses are **masked, never interpolated**. |
| `scenario/` | 8.1 | Config dataclasses and the dataset splits, cut on **parameter axes** rather than on seeds. |
| `simulation/` | 3.3, 10 | Engine and episode loop; `rollout.py` generates the branch rollouts that are the **true distribution** calibration is scored against. |
| `storage/` | 4 | The on-disk contract. GT and OBS stored **separately**: input is OBS, ground truth is GT. |
| `analysis/` | 8.3, 9 | `cv_baseline.py` = the CV baseline in its dual role (performance floor + data-quality diagnostic); `metrics.py` = ADE/FDE, NLL/CRPS, coverage/PIT, NMAC/DWC. |
| `gui/` | — | Stdlib-only inspection GUI. Not a doc section: it exists so that doc 10 step 2 can be judged by eye as well as by number. |

---

## Conventions (non-negotiable — every module agrees on these)

- **SI units everywhere**: metres, m/s, radians, seconds, kilograms.
- **Local tangent plane**: `+X` = East, `+Y` = North.
- `psi` = heading measured **CCW from +X** (standard math convention, *not*
  compass), wrapped to `(-pi, pi]`.
- Positive turn rate `omega` = **left** turn = CCW = increasing `psi`.
- Integration at `SIM_DT = 0.05 s` (20 Hz); logging at `LOG_DT = 0.5 s` (2 Hz);
  `LOG_DECIMATION = 10`. **Integrate fine, log coarse.**
- Episode 60 s → **120 logged steps**. History window 10 s = 20 steps.
  Prediction horizon 15 s = 30 steps.
- `float32` on disk, `float64` for internal computation.

---

## Quick start

```bash
cd $PRISM_ROOT/dataGen

# 1. is the code healthy?
python main.py selftest

# 2. does one episode look sane?
python main.py demo

# 3. build a small dataset
python main.py generate --preset train --episodes 5 --tag smoke

# 4. is what landed on disk valid?
python main.py verify --run latest

# 5. is the data hard enough to justify the paper?  (doc 8.3 — the real gate)
python main.py diagnose --preset all --episodes 24
```

Below, `python` is shorthand for that full interpreter path.

---

## Subcommands

### `gui` — inspect the simulator visually (default with no arguments)

```bash
python main.py                      # identical to: python main.py gui
python main.py gui --port 8770
python main.py gui --port 8899 --no-open
```

Serves on **loopback only** (`127.0.0.1`) and opens a browser at the URL it
prints. If the port is busy it probes the next 40, then lets the OS assign one,
so the printed URL is authoritative. `--no-open` suppresses the browser.

Live simulation with SSE streaming, branch-rollout fan overlay, sensor
covariance ellipses, batch generation with a progress bar, dataset replay and
the CV diagnostic — all in the browser.

### `generate` — build a dataset run

```bash
python main.py generate --preset train    --episodes 5000 --tag v1
python main.py generate --preset test_ood --episodes 500  --tag v1 --rollouts 50
python main.py generate --preset test_ood --episodes 100  --tag eval --rollouts 64 --branch-t 25
```

| flag | default | meaning |
|---|---|---|
| `--preset` | `train` | `train` / `valid` / `test_id` / `test_ood` |
| `--episodes` | `100` | number of episodes |
| `--tag` | `run` | free-text tag folded into the run id |
| `--rollouts K` | `0` (off) | also write `K` branch rollouts per episode |
| `--branch-t` | `30.0` | fork time [s] for `--rollouts` |

Writes to `$PRISM_ROOT/Logs/Dataset/<timestamp>_<preset>_<tag>/` with a console progress
bar showing rate, ETA and a running NMAC count. The bar repaints in place on a
terminal and degrades to periodic lines when stderr is redirected.

**`--rollouts` is the doc 3.3 true distribution**: the same initial condition
and the same past, re-rolled with different *decision* seeds. This is what
calibration is scored against, and it is the reason the study uses a simulator
at all — a real trajectory's future happens exactly once (doc `[1]`). Doc 3.7
asks for 50–100 branches, evaluation set only.

### `diagnose` — the doc 8.3 CV baseline verdict

```bash
python main.py diagnose --preset all --episodes 24      # generate in memory
python main.py diagnose --run latest                    # score a written run
python main.py diagnose --run 20260812_172146_train_smoke
python main.py diagnose --run latest --json report.json
```

Doc 8.3 gives the CV baseline two jobs. Job one is the performance floor every
reviewer asks about first. Job two — this subcommand — is data-quality
diagnosis:

| CV 15 s median error | meaning | action |
|---|---|---|
| too small | data is too easy | increase turn frequency |
| too large | data is too random | lengthen manoeuvre dwell |
| same order as the NMAC radius (152.4 m) | **OK** | proceed |

`--run` accepts `latest`, a bare run id, or a full path. `--velocity-source gt`
(the default) uses logged ground-truth velocity — the *best case* for CV, which
is the honest diagnostic; `diff` uses a backward difference of positions.

**Exit code**: 0 when the dataset gate passes **or is not applicable**, 1 when
it genuinely fails. See "The dataset gate" below — a `train`-only run reports
`N/A`, not `FAIL`, and that is deliberate.

### `verify` — check a written run against the storage contract

```bash
python main.py verify --run latest
python main.py verify --run latest --shallow          # index only, no NPZ reopen
python main.py verify --run latest --episodes 50      # cap the deep check
```

Reopens every episode NPZ, runs `storage.schema.validate`, and cross-checks the
index against the files on disk (orphans, missing entries, duplicate ids, `T` vs
the manifest, `meta.n_agents` vs slots actually used). Prints a summary and
exits 1 on any problem.

It also reports the run **status**. A run killed part-way through (Ctrl-C, or a
crash) is stamped `aborted` and its surviving episodes are still readable and
still internally consistent — so `verify` says so explicitly rather than calling
it "complete":

```
OK: the 2 episodes present are consistent with the storage contract,
    but the run status is 'aborted' -- it is a PARTIAL dataset, not a finished one.
```

### `demo` — one episode, headless, no disk writes

```bash
python main.py demo
python main.py demo --preset test_ood --seed 12345
```

The fastest end-to-end check that the whole stack is wired together: scenario
build → agent decisions → wind → sensor → tracker → logging → metrics, plus the
schema validation and a per-episode CV baseline.

### `platforms` — the performance and stability catalogue

```bash
python main.py platforms
```

### `selftest` — run every test file

```bash
python main.py selftest
python main.py selftest -k sensors
```

Each `tests/test_*.py` is a standalone runner executed as a subprocess (no
pytest — the environment may not import it).

---

## Output dataset format

### Run directory

```
$PRISM_ROOT/Logs/Dataset/20260812_172146_train_smoke/
  manifest.json          config snapshot, timing block, splits, environment,
                         source fingerprint (SHA-256 over every .py), array spec
  config.json            the ScenarioConfig, byte-identical to the manifest copy
  index.jsonl            one EpisodeMeta per line, append-only
  README.txt             array dictionary generated with the run
  episodes/ep_000000.npz compressed arrays, one file per episode
  rollouts/ep_000000.npz branch rollouts (only when --rollouts was used)
  preview/ep_000000.json decimated strict-JSON replay for the GUI
```

Crash-safe: whole-file writes are `.tmp` → `fsync` → `os.replace`, and the
episode NPZ is written **before** its index line, so a killed run never has an
index entry pointing at a missing file.

### Episode arrays (`episodes/ep_NNNNNN.npz`)

`T` = logged steps (120 for a 60 s episode), `N` = agent slots.

**Ground truth — the label**

| array | shape | dtype |
|---|---|---|
| `t` | `[T]` | float32 |
| `own_pos` / `own_vel` | `[T,2]` | float32 |
| `own_psi` | `[T]` | float32 |
| `gt_valid` | `[T,N]` | bool |
| `gt_pos` / `gt_vel` | `[T,N,2]` | float32 |
| `gt_psi` | `[T,N]` | float32 |
| `gt_mode` / `gt_intent` | `[T,N]` | int8 |
| `gt_type` | `[N]` | int8 |
| `agent_ids` | `[N]` | int32 |

**Observation — the model input**

| array | shape | dtype |
|---|---|---|
| `obs_detected` | `[T,N]` | bool |
| `obs_pos` | `[T,N,2]` | float32 |
| `obs_R` | `[T,N,2,2]` | float32 |
| `obs_t_valid` | `[T,N]` | float32 |
| `obs_track_id` | `[T,N]` | int32 |
| `obs_track_age` | `[T,N]` | int32 |

**Sentinel and NaN policy — load-bearing, do not "clean" it**

- `obs_pos` / `obs_R` / `obs_t_valid` are **NaN** wherever `obs_detected` is
  False. **Never interpolate.** Doc 3.5 requires the model to learn "no
  information", so that a real track dropout inflates its predicted sigma
  rather than silently inventing a position.
- **`obs_t_valid` is the time the measurement was actually true**, and it is
  always EARLIER than `t`. The sensor runs at 5 Hz behind a 0.2 ± 0.05 s
  latency and the log runs at 2 Hz, so the measurement folded into a logged
  frame is 0.05–0.78 s stale (measured). `obs_R` covers **only** the polar
  measurement noise at `t_valid`; it does not include the staleness.
  A consumer that ignores this treats a stale position as a current one and
  the input layer's own uncertainty comes out ~50 % optimistic in variance:

  | | NEES (want 2.0) | 90 % coverage (want 0.90) |
  |---|---|---|
  | `obs_pos − gt_pos(t)` against `obs_R` | 2.55 | 0.848 |
  | `obs_pos − gt_pos(obs_t_valid)` against `obs_R` | **1.96** | **0.903** |

  i.e. the whole discrepancy is time alignment and it is fully removable from
  what is on disk. Propagate the target back to `obs_t_valid` (or add the
  staleness term to `R`) before scoring anything calibration-related.
- `obs_track_id` is `-1` when no track is live; `obs_track_age == 0` is an exact
  synonym for that. Age counts **logged steps**. A re-acquired target gets a
  **new** track id — ids are never reused, so a model cannot assume continuity.
- `gt_mode` / `gt_intent` use `-1` for unknown.

### Branch rollouts (`rollouts/ep_NNNNNN.npz`, `--rollouts` only)

`B` = agent slots, `K` = branches, `H` = 30 prediction steps.

| array | shape | dtype |
|---|---|---|
| `origin_step` | `[B]` | int32 |
| `agent_slot` | `[B]` | int32 |
| `branch_seed` | `[B,K]` | int64 |
| `pos` | `[B,K,H,2]` | float32 |
| `valid` | `[B,K,H]` | bool |

### `index.jsonl`

One `EpisodeMeta` per line: `episode_index`, `seed`, `density`,
`realised_density_per_km2`, `wind_speed`, `wind_dir`, `n_agents`,
`geometry_type`, `cpa_dist`, `tcpa`, `min_separation_m`, `risk_level`,
`platform_counts`, `intent_counts`, `nmac_occurred`, `dwc_violated`,
`dwc_alerted`, `duration_s`.

`geometry_type`, `cpa_dist` and `tcpa` are **lists** — one entry per designed
encounter (doc 3.4 samples **0–3** of them; zero is deliberate, see *Risk
stratification* below). They record **design intent**: agents evolve freely
from `t=0`, so the *realised* closest approach is `min_separation_m`, which is
generally not `cpa_dist`.

Three fields are easy to misread and are worth stating outright:

- **`density` is the doc 8.1 split axis, not a scene property.** It is defined
  over a 0.25 km² reference footprint while the arena is 36 km², so it is
  ~60× the density the episode actually contains. `realised_density_per_km2`
  (`n_agents / 36 km²`, typically 0.14–0.69) is the number to quote in any
  spacing-related paper text.
- **`min_separation_m` is measured at the 20 Hz integration rate**, not from
  the 2 Hz arrays. A minimum taken over `gt_pos` is biased HIGH by up to half
  a logging interval of closure — measured at 19.2 m, 12.6 % of the NMAC
  radius — and the bias is one-sided, so a true NMAC just inside the radius
  would be labelled safe. `nmac_occurred` follows from the fine measurement.
- **`dwc_violated` means well clear was actually LOST** (achieved
  `min_separation_m < DWC_HMD_M`). `dwc_alerted` is the separate, much wider
  *alerting* predicate: the CV projection of the relative state was going to
  lose well clear within `tau_mod`. In a 36 km² arena the alert fires in
  essentially every episode, so the two must not be confused.

---

## Dataset splits (doc 8.1)

Splits are cut on **parameter axes**, not on seeds — splitting the same
scenario by seed alone would leak. Seed blocks are disjoint as well, so both
guarantees hold at once.

| split | seeds | density /km² | wind m/s | agents | intents | platforms |
|---|---|---|---|---|---|---|
| `train` | 0–99 999 | 20–60 | 0–8 | 5–15 | corridor, delivery WP, loiter | multirotor, fixed-wing UAS, eVTOL |
| `valid` | 100 000–199 999 | 20–60 | 0–8 | 5–15 | *same as train* | *same as train* |
| `test_id` | 200 000–299 999 | 20–60 | 0–8 | 5–15 | *same as train* | *same as train* |
| `test_ood` | 300 000–399 999 | **100** | **10–16** | **25** | **+ search wander, reactive avoid, passive drift** | **+ bird, GA, debris, unknown** |

`valid` and `test_id` are in-distribution controls, not generalisation tests.
`test_ood` is the doc 8.1 claim: an unseen aircraft class should make the model
flatten its type distribution and widen sigma, rather than force a
classification.

### The dataset gate

`diagnose` asks three questions, not one:

1. do the **HARD** classes genuinely defeat CV?
2. are the **EASY** classes genuinely easy (and not degenerate)?
3. is the separation between the bands large enough to see?

A `train`-only run contains **zero** hard samples — doc 8.1 deliberately keeps
bird, unknown, search-wander and reactive-avoid out of training. Question (1)
therefore has no data to answer it, and the gate reports **`N/A`** rather than
`FAIL`: blaming the data for a property of the split would be a category error.
An *empty* dataset still reports `FAIL`. Judge the gate with `--preset all`.

---

## Parameter provenance (doc 12)

> Doc 12: the greatest weakness of a simulation paper is the arbitrariness of
> its parameters. Every physical number must trace to a document, and anything
> that cannot must be marked **assumed** and entered into the sensitivity
> analysis.

`agents.platforms.PARAMETER_PROVENANCE` carries this table in machine-readable
form; `python main.py platforms` prints the live values.

### Standard-derived

| parameter | value | source |
|---|---|---|
| `G` | 9.806 65 m/s² | ISO 80000-3 / CGPM standard gravity |
| `NMAC_RADIUS_M` | 152.4 m (500 ft) | Common NMAC definition, horizontal component only (doc 9.4). 2D study — the vertical 100 ft criterion is not modelled, and the paper must say so. |
| standard rate turn | 3 °/s | Basic airmanship; the doc 7.2 reference for the RL action-space limit |
| `MULTIROTOR.v_max` | ≤ 44.7 m/s | Bounded by 14 CFR 107.51 ground-speed limit (100 mph) |
| `FIXED_WING_UAS.v_range` | 22–38 m/s | Bounded by 14 CFR 107.51 |

### Needs verification before publication

| parameter | value | status |
|---|---|---|
| `DWC_HMD_M` | 1219.2 m (4000 ft) | ⚠ **Commonly quoted RTCA DO-365 value, NOT yet checked against the standard's own text.** Marked in `core/constants.py`. Doc 12.5 checklist item 1. |
| `DWC_TAU_MOD_S` | 35.0 s | ⚠ Same caveat. |

### Assumed — bounded by a citeable limit

| parameter | value | basis |
|---|---|---|
| `EVTOL_UAM.bank_max` | passenger comfort | Ride-comfort limit; ties to the doc 7.2 defence paragraph |
| `GA.bank_max` | typical enroute | Typical enroute manoeuvring bank |

### Assumed — source outstanding (sensitivity-analysis candidates)

| parameter | basis | doc 12 item |
|---|---|---|
| `EVTOL_UAM.v_range` (32–76 m/s) | Manufacturer spec or NASA reference vehicle needed | 12.4 |
| `BIRD.v_range` (6–22 m/s) | Birdstrike literature needed | 12.4 |
| `BIRD.omega_max` | Avian avoidance-behaviour observation study needed | 12.4 |
| `GA.v_range` (42–67 m/s) | Light-aircraft POH cruise figures needed | 12.4 |
| `DEBRIS.mass`, `DEBRIS.drag_area` | No standard exists | — |
| `omega_dot_max` (all platforms) | Roll-onset rate; no public figure used | — |
| `wind_susceptibility` (all platforms) | Modelling device, not a measurement | — |
| `a_max`, `a_min` (all platforms) | Plausible longitudinal authority, unsourced | — |
| `UNKNOWN.factors` | **Deliberately arbitrary** — the purpose is to leave the trained envelope | — |
| `stability.dwell_*` | Manoeuvre commitment per platform. The **ordering** (eVTOL ≫ GA > fixed-wing > multirotor > bird) is the defensible claim, not the absolute seconds. | — |
| `stability.mode_authority`, `stability.diag_bias` | Modelling devices: how far a discrete manoeuvre overrides guidance, and the stickiness of the mode-transition matrix | — |

### Sensor parameters (doc 3.5) — all currently assumed

Doc 3.5 requires these to be set from DO-365-class surveillance performance
requirements and cited in doc 12. **They are placeholders until that check is
done.**

| parameter | default | unit |
|---|---|---|
| `sigma_r0_m` | 3.0 | m (range noise floor) |
| `sigma_r_frac` | 0.01 | + 1 % of range |
| `sigma_theta_rad` | 0.0087 | rad (0.5°) |
| `pd_max` | 0.95 | — |
| `r_full_m` / `r_max_m` | 800 / 3000 | m |
| `update_rate_hz` | 5.0 | Hz |
| `latency_s` / `latency_jitter_s` | 0.2 / 0.05 | s |
| `drop_after_misses` | 6 | consecutive misses → track dropped |

---

## Reproducibility

Reproducibility is a paper requirement, not a convenience.

- `core.rng.RNGHub` derives every stream deterministically from
  `(master_seed, stream_name)` via `SeedSequence(spawn_key=crc32(name))`.
  `zlib.crc32` is used rather than Python's `hash()`, which is randomised per
  process and would break run-to-run reproducibility.
- Named streams: `scenario`, `placement`, `platform`, `intent`,
  `mode_decision`, `branch_choice`, `wind`, `sensor_noise`, `sensor_detect`,
  `sensor_latency`.
- Branch rollouts re-seed **only** the decision streams, leaving placement,
  platform and intent identical. That is what makes a branch a genuine
  alternative future of the *same* scenario rather than a different scenario.
- Every run's `manifest.json` carries a SHA-256 fingerprint over all `.py`
  files, standing in for a git commit (this tree is not a repository).
- The same `(config, seed)` reproduces all 18 arrays bit-for-bit.

---

## Gotchas worth knowing

- **`storage` is not called `io`** — that would shadow the stdlib module.
- **`EpisodeArrays.from_npz_dict` does not copy.** It uses `np.asarray`, a
  no-op view when the dtype already matches, so
  `from_npz_dict(ep.to_npz_dict())` aliases the original. Call `.copy()`
  explicitly before perturbing.
- **`EncounterConfig.tcpa_max_s` must be ≤ `duration_s`**, or
  `ScenarioConfig.validate()` fails and `verify` rejects the whole run. Short
  debug configs need `tcpa_max_s` lowered to match.
- **Risk stratification is uneven, and the reason is the airspace, not a
  bug.** Measured over 80 episodes (20 per split, production config path):
  `HIGH 33, MEDIUM 46, LOW 1, NONE 0`; `min_separation_m` min 1 m, median
  197 m, max 1244 m. Three of four strata occur. `NONE` requires the ownship
  to keep 1219 m (`DWC_HMD_M`, a manned-aviation DAA threshold) from every
  agent for a whole minute, and a 55 m/s ownship sweeps a 12.7 km² corridor
  through a 36 km² arena holding 5–25 aircraft — so at doc 3.7 densities the
  sky is essentially never empty. It is a scale mismatch between DO-365 and
  low-altitude urban traffic, not a generator defect. Two things were fixed to
  get this far, and both matter: `n_designed_min` is now **0** (it was 1, so
  every episode was *guaranteed* a designed conflict and no nominal episode
  existed anywhere in the dataset), and `cpa_dist_max_m` is now **2500 m**, up
  from 600 m — the old band sat entirely inside `DWC_HMD_M`, so every designed
  encounter had already lost well clear before the simulation started.
  `EncounterConfig.validate()` now rejects a `cpa_dist_max_m` at or below the
  DWC threshold, and `verify` prints a `** WARNING` naming any stratum that
  never occurred. NMAC rate fell from 0.58 to 0.41 as a side effect.
  Enlarging the arena would populate `NONE` but was measured and rejected: at
  64 km² the detection rate drops from 0.46 to 0.31 and at 100 km² to 0.22,
  and the observation layer is the model's input.
- **`density_per_km2` is a split axis, not a realised areal density.** Doc 3.7
  (5–15 agents) and doc 8.1 (20–60 /km²) are simultaneously satisfiable only at
  a 0.25 km² reference footprint, while the arena is 36 km². Quote
  `Scenario.realised_density_per_km2` in any spacing-related paper text.
- **Manoeuvre commitment is per-platform.** An eVTOL holds a manoeuvre for
  ~12–15 s and only biases its corridor guidance; a bird re-decides every
  ~1–2.5 s and fully commits. There is deliberately **no** global dwell floor.
- **Mode dwell is truncated at the 15 s prediction horizon**
  (`ModeMachine._draw_dwell`). A dwell that outlives the horizon means no
  decision epoch falls inside the prediction window, and for an agent whose
  intent has no continuous stochastic term the empirical future collapses to a
  literal point mass — measured at 200 branches producing 2 distinct endpoints
  and a spread of 4e-4 m. This does **not** destabilise the stable platforms:
  what governs how long an aircraft flies straight is the diagonal of its
  platform-biased transition matrix, not the dwell tail. An eVTOL in corridor
  cruise re-decides with P(stay CV) = 0.909, giving an expected uninterrupted
  cruise leg of ~148 s against a 60 s episode, and a CV→CV re-latch has an
  empty parameter payload, i.e. it is a no-op on the trajectory. The truncated
  tail is **redrawn uniformly** in `[dwell_min_s, 15 s]`, never clamped to
  15 s: clamping put an atom on exactly one dwell length, which for the eVTOL
  (mean 25 s) carried most of the mass, put its mode changes on an exact 15 s
  grid and resonated with the branch times.
- **Reactive avoidance triggers on the well-clear criterion.** An agent breaks
  away when the developing geometry would cost DAA well clear within the
  well-clear lookahead (projected miss < `DWC_HMD_M`, CPA within
  `DWC_TAU_MOD_S`, range < 2500 m). There is a structural ceiling on how often
  that can fire: only ~76 % of `REACTIVE_AVOID` agents ever come within
  2500 m of the ownship and only ~52 % ever project a CPA inside the HMD, so
  the raw rate is 0.41 reactions/agent/episode with 31 % of agents reacting.
  What is asserted by `selftest` is the number that is actually under the
  model's control: **100 % (33/33) of agents whose geometry qualifies do
  react**. Forcing the raw rate to 1.0 would mean aiming every reactive agent
  at the ownship, which fabricates the interaction instead of letting it
  emerge from the geometry — the opposite of doc 3.3.
- **Platform and intent are not independent.**
  `agents.intent.PLATFORM_INTENT_COMPATIBILITY` stops a passenger eVTOL or a
  manned light aircraft being given `SEARCH_WANDER` (doc 3.2's random zigzag)
  and stops a bird transiting a corridor or flying a delivery route. Drawing
  them independently made the platform label a *mixture* rather than a class
  — the eVTOL's 15 s CV error was 17 m on a corridor and 179 m
  search-wandering — and depressed the platform-axis band separation from
  4.01× to 2.50×. Rejected draws are **redrawn from the same intent mix
  restricted to the compatible set**, not mapped to a fixed fallback.

---

## Verified status

Everything below was executed on this machine, not assumed.

```
python main.py selftest                                    10/10 test files pass (57.5 s)
python main.py demo                                        OK, schema + meta validate clean
python main.py generate --preset train --episodes 5 --tag accept   OK (1 NMAC of 5)
python main.py verify   --run latest                       OK, contract-consistent
python main.py diagnose --run latest                       AGGREGATE VERDICT: OK, gate N/A (train-only)
python main.py diagnose --preset all --episodes 24         AGGREGATE VERDICT: OK, DATASET GATE: PASS
```

CV baseline over all four splits, 24 episodes — the doc 8.3 numbers:

| horizon | samples | median error | p90 | vs NMAC radius |
|---|---|---|---|---|
| 5 s | 37 840 | 10.1 m | 42.9 m | 0.07 |
| 10 s | 34 400 | 37.4 m | 143.1 m | 0.25 |
| **15 s** | **30 960** | **75.9 m** | **270.3 m** | **0.50** |

Per class at 15 s, each judged against **its own** expected band — a low CV
error on an eVTOL is a pass, the same number on a bird would be a failure:

| axis | class | expected | band | median | p90 | verdict |
|---|---|---|---|---|---|---|
| platform | Bird | HARD | 76–457 m | 179.2 | 320.5 | OK |
| platform | Unknown | HARD | 76–457 m | 188.8 | 443.9 | OK |
| platform | Fixed-wing UAS | MEDIUM | 38–457 m | 103.4 | 357.0 | OK |
| platform | Multirotor | MEDIUM | 38–457 m | 45.0 | 143.0 | OK |
| platform | General aviation | EASY | 8–152 m | 86.1 | 253.7 | OK |
| platform | eVTOL UAM | EASY | 8–152 m | 38.4 | 218.0 | OK |
| platform | Debris | EASY | 0–152 m | 5.1 | 36.4 | OK |
| intent | Search wander | HARD | 76–457 m | 134.1 | 296.5 | OK |
| intent | Reactive avoid | HARD | 76–457 m | 101.5 | 273.2 | OK |
| intent | Loiter / hover | MEDIUM | 38–457 m | 145.1 | 379.4 | OK |
| intent | Delivery waypoints | MEDIUM | 38–457 m | 96.5 | 291.1 | OK |
| intent | Corridor transit | EASY | 8–152 m | 16.0 | 94.8 | OK |
| intent | Passive drift | EASY | 0–152 m | 5.1 | 36.4 | OK |

One class sits close to an edge and is worth knowing about before it surprises
someone: **Multirotor** in the `train` split alone measures 40.8 m at 15 s over
27 270 samples (30 episodes) against a MEDIUM lower bound of 38.1 m — a 7 %
margin. A 5-episode `diagnose --run latest` can and does report it as
`TOO_EASY` (30.9 m over 2 070 samples). That is sampling noise, not a data
defect, and it is why `MIN_SAMPLES_FOR_VERDICT` exists; judge the class on
`diagnose --preset all --episodes 24` or larger, where it reads 45.0 m / OK.
It was deliberately **not** tuned: at the canonical sample size it is not a
failure, and doc 8.3 only prescribes changing turn frequency for a class that
genuinely is one.

Band separation at 15 s: **4.29×** by platform (hard 181.6 m / easy 42.3 m) and
**8.15×** by intent (hard 114.6 m / easy 14.1 m), against a 2.0× requirement.
Every class lands inside its own expected difficulty band, so **no difficulty
tuning was applied** — the platform ratio rose from 2.50× only because the
platform × intent confound was removed, not because any class was made more or
less erratic. The eVTOL still flies like an aircraft: 18.1 m median CV error on
a corridor, the most predictable powered class in the dataset.

The 15 s median sits at roughly half the NMAC radius — the same order of
magnitude doc 8.3 asks for, and the hard classes exceed it. The data is hard
enough to justify a probabilistic predictor.

GUI: every GET endpoint returns HTTP 200 with valid JSON or HTML; error
responses are JSON envelopes rather than HTML 500 pages; path traversal is
rejected. `POST /api/diagnose {"run":"latest"}` resolves, matching the CLI.
`POST /api/generate` rejects unknown top-level keys (400) instead of silently
running the default episode count, and accepts `episodes` as an alias for
`n_episodes`. `POST /api/sim/control {"action":"step"}` returns the frame
AFTER the step, not before it.
