"""PRISM dataGen -- 2D traffic simulator and trajectory-prediction dataset generator.

This is the ONLY top-level module in the package. Everything else lives in a
function-named subpackage; this file exists to put ``$PRISM_ROOT/dataGen`` on
``sys.path`` and dispatch to it.

WHAT THIS PROGRAM IS
--------------------------------------------------------------------------
PRISM is a study in *calibrated* probabilistic trajectory prediction for
RL-based collision avoidance against NON-COOPERATIVE aerial traffic in
low-altitude airspace (논문관련.txt section [0]). The full pipeline is::

    2D simulator -> trajectory prediction model -> BEV risk map -> PPO policy
    ^^^^^^^^^^^^^
    this program (stages 1-3 of doc [10])

This program builds only the simulator and the dataset generator. It produces
episodes of ownship + 5-15 non-cooperative agents, logs ground truth and
degraded sensor observations SEPARATELY (doc [4]), and writes them to
``$PRISM_ROOT/Logs`` for the downstream predictor to train on.

THE ONE DESIGN RULE THAT GOVERNS EVERYTHING (doc 3.1)
--------------------------------------------------------------------------
The simulator's purpose is NOT to look realistic. It is to embed structure a
constant-velocity (CV) extrapolator CANNOT predict. If the trajectories are too
straight, CV wins and the paper's contribution collapses. Multimodality
(branching futures) therefore arises from STOCHASTIC DECISION RULES given to
the agents (doc 3.3) -- never from hand-placed waypoints on a map.

``python main.py diagnose`` is the guard on that rule and the first thing to
run after ANY change to the simulator (doc [10] step 2). It flies episodes,
extrapolates every traffic agent with a constant-velocity model, and reports
whether the data is hard enough to justify a probabilistic predictor.

HOW THE PIECES FIT / WHICH DOC SECTION EACH SUBPACKAGE IMPLEMENTS
--------------------------------------------------------------------------
=================  ========  ==================================================
subpackage         doc       role
=================  ========  ==================================================
core/              3.7, 9.4  units, enums, RNG, state, kinematics. The shared
                             contract every other package imports. Enforces the
                             doc 3.3(a) "only as much as physically possible"
                             constraint inside the integrator.
agents/            3.2, 3.3  objects as purpose x character x performance.
                             platforms.py = the 3.2(3) performance table;
                             intent.py = the 3.2(1) purpose types;
                             maneuver.py + behavior.py = the 3.3 stochastic
                             decision rules that CREATE the multimodality,
                             including the 3.3(b) manoeuvre-dwell constraint.
environment/       3.4, 3.6  wind.py = 3.6 steady wind (the systematic CV bias
                             and the physical origin of aleatoric uncertainty);
                             encounter.py = 3.4 INVERSE-DESIGNED encounter
                             geometry (sample d*/t*, solve for the initial
                             state); world.py = corridors and no-fly zones.
sensors/           3.5       Pd < 1, polar noise -> Cartesian, async update
                             rates, latency + jitter, track drop/re-acquire.
                             Misses are MASKED, never interpolated (doc 3.5).
scenario/          8.1       config dataclasses and the dataset splits cut on
                             PARAMETER AXES (density / intent / platform /
                             wind), not on seeds -- doc 8.1 leakage rule.
simulation/        3.3, 10   engine + episode loop; rollout.py generates the
                             branch rollouts that are the TRUE distribution
                             calibration is scored against (doc 3.3, 1).
storage/           4         the on-disk contract. GT and OBS stored
                             separately: input is OBS, ground truth is GT.
analysis/          8.3, 9    cv_baseline.py = the doc 8.3 CV baseline in its
                             dual role (performance floor + data-quality
                             diagnostic); metrics.py = doc 9.1-9.4 (ADE/FDE,
                             NLL/CRPS, coverage/PIT, NMAC/DWC).
gui/               --        stdlib-only inspection GUI. Not a doc section; it
                             exists so doc 10 step 2 ("if this looks wrong
                             everything downstream shakes") can be judged by
                             eye as well as by number.
=================  ========  ==================================================

SUBCOMMANDS
--------------------------------------------------------------------------
::

    python main.py                                  # same as: gui
    python main.py gui       [--port 8770] [--no-open]
    python main.py generate  --preset train --episodes 100 --tag v1 [--rollouts 50]
    python main.py diagnose  [--run latest | --preset all --episodes 20]
    python main.py verify    --run latest
    python main.py demo      [--preset train] [--seed N]
    python main.py platforms
    python main.py selftest  [-k PATTERN]

Units are SI throughout, as everywhere in dataGen: metres, m/s, radians,
seconds, kilograms. Headings are CCW from +X (East); positive turn rate is a
left turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------
# shared console helpers
# --------------------------------------------------------------------------


class Progress:
    """A console progress bar that degrades gracefully when piped.

    Writes to stderr so that stdout stays a clean, parseable result stream.
    On a terminal it repaints one line with ``\\r``; when stderr is redirected
    it emits a periodic plain line instead, because a redirected ``\\r`` just
    produces an unreadable smear of half-overwritten text in the log file.
    """

    def __init__(self, total: int, label: str = "", width: int = 28) -> None:
        """Create a bar.

        Args:
            total: number of units of work [-].
            label: short prefix shown before the bar.
            width: bar width in characters [-].
        """
        self.total = max(1, int(total))
        self.label = label
        self.width = int(width)
        self.t0 = time.time()
        self.n = 0
        self.tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._last_line = 0.0
        self._final_emitted = False

    def update(self, n: int, suffix: str = "") -> None:
        """Advance the bar to ``n`` completed units.

        Args:
            n: units completed so far [-].
            suffix: extra text appended after the counters.
        """
        self.n = int(n)
        frac = min(1.0, self.n / self.total)
        elapsed = time.time() - self.t0
        rate = (self.n / elapsed) if elapsed > 0 and self.n else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        msg = (f"{self.label}[{bar}] {frac * 100:5.1f}%  {self.n}/{self.total}"
               f"  {rate:5.2f}/s  eta {self._hms(eta)}")
        if suffix:
            msg += f"  {suffix}"
        if self.tty:
            sys.stderr.write("\r" + msg + "  ")
            sys.stderr.flush()
        else:
            # Redirected: emit a line every few seconds, and exactly ONE final
            # line -- done() re-enters update(), so without the latch the
            # completion line is printed twice.
            now = time.time()
            complete = self.n >= self.total
            if complete and self._final_emitted:
                return
            if complete or now - self._last_line >= 5.0:
                self._last_line = now
                self._final_emitted = self._final_emitted or complete
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()

    def done(self, suffix: str = "") -> None:
        """Finish the bar and leave the cursor on a fresh line."""
        self.update(self.total, suffix)
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()

    @staticmethod
    def _hms(seconds: float) -> str:
        """Format a duration [s] as ``MM:SS`` (or ``HH:MM:SS`` past an hour)."""
        s = max(0, int(round(seconds)))
        if s >= 3600:
            return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return f"{s // 60:02d}:{s % 60:02d}"


def _resolve_run(spec: str) -> str:
    """Resolve a ``--run`` argument to an absolute run directory.

    Accepts the three things a user naturally types: the word ``latest``, a
    bare run id (the directory name under ``LOGS_ROOT``), or a full path.

    Args:
        spec: ``'latest'``, a run id, or a path.

    Returns:
        Absolute run directory with forward slashes.

    Raises:
        SystemExit: if nothing matches, listing what IS available -- an
            unhelpful "not found" for a mistyped run id wastes more time than
            the listing costs.
    """
    from core.constants import LOGS_ROOT
    from storage.writer import latest_run, list_runs

    spec = (spec or "latest").strip().strip('"').replace("\\", "/")
    if spec == "latest":
        run = latest_run()
        if not run:
            raise SystemExit(
                f"no runs found under {LOGS_ROOT}\n"
                f"generate one first:  python main.py generate "
                f"--preset train --episodes 5 --tag smoke")
        return run.replace("\\", "/")
    if os.path.isdir(spec):
        return os.path.abspath(spec).replace("\\", "/")
    candidate = os.path.join(LOGS_ROOT, spec).replace("\\", "/")
    if os.path.isdir(candidate):
        return candidate
    available = [r.get("run_id", "?") for r in list_runs()][:12]
    listing = "\n  ".join(available) if available else "(none)"
    raise SystemExit(f"run not found: {spec}\navailable runs:\n  {listing}")


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def _write_branch_rollouts(writer, engine, episode_index: int,
                           n_branches: int, branch_t: float) -> int:
    """Take branch rollouts from a finished episode and write them.

    This is the doc 3.3 "true distribution": the same initial condition and
    the same past, re-rolled with different decision seeds. It is what
    calibration is scored against, and it is the reason the simulator exists
    rather than a real-track dataset (doc [1]: a real future happens once).

    ``simulation.rollout.branch_rollout`` returns ``[n_branches, n_agents, H,
    2]`` but ``storage.writer.ROLLOUT_ARRAY_SPEC`` is keyed ``[B, K, H, 2]``
    with ``B`` = rollout groups and ``K`` = branches, so the first two axes are
    transposed here: one group per agent slot, all sharing this branch point.

    Args:
        writer: an open :class:`storage.writer.DatasetWriter`.
        engine: the engine that produced the episode; it is reset and replayed
            to ``branch_t`` (deterministic, so the past matches the log).
        episode_index: index [-] this rollout set belongs to.
        n_branches: branches to roll [-]; doc 3.7 asks for 50-100.
        branch_t: fork time [s].

    Returns:
        Number of agent slots written [-].
    """
    import numpy as np

    from core.constants import LOG_DT, PRED_STEPS
    from simulation.rollout import branch_rollout, branch_seed_block

    engine.reset()
    seeds = branch_seed_block(int(engine.cfg.seed) + int(round(branch_t)),
                              n_branches)
    # require_divergence=False: a legitimately degenerate slot (an agent that
    # is already retired, or a committed loiter orbit with no live decision)
    # must not abort a whole dataset build.
    pos = branch_rollout(engine, branch_t, n_branches,
                         horizon_s=PRED_STEPS * LOG_DT,
                         branch_seeds=seeds, require_divergence=False)
    pos = np.asarray(pos)                      # [K, N, H, 2]
    grouped = np.transpose(pos, (1, 0, 2, 3))  # [B=N, K, H, 2]
    n_slots = grouped.shape[0]
    step = int(round(branch_t / LOG_DT))
    arrays = {
        "origin_step": np.full((n_slots,), step, dtype=np.int32),
        "agent_slot": np.arange(n_slots, dtype=np.int32),
        "branch_seed": np.tile(np.asarray(seeds, dtype=np.int64),
                               (n_slots, 1)),
        "pos": grouped.astype(np.float32),
        "valid": np.isfinite(grouped).all(axis=-1),
    }
    writer.write_rollouts(episode_index, arrays)
    return n_slots


def _cmd_generate(args: argparse.Namespace) -> int:
    """Build a dataset run under ``LOGS_ROOT``.

    Args:
        args: parsed arguments (``preset`` / ``episodes`` / ``tag`` /
            ``rollouts`` / ``branch_t``).

    Returns:
        Process exit code [-]: 0 on success.
    """
    from scenario.presets import config_for_episode, make_preset, seed_block
    from simulation.engine import SimulationEngine
    from storage.writer import DatasetWriter

    n = int(args.episodes)
    k = int(args.rollouts)
    start = int(getattr(args, "start_index", 0))
    writer = DatasetWriter()
    run_dir = writer.open_run(make_preset(args.preset), preset=args.preset,
                              tag=args.tag)
    print(f"run       : {run_dir}")
    print(f"preset    : {args.preset}   episodes: {n}   start index: {start}"
          + (f"   branch rollouts: {k} @ t={args.branch_t:g} s" if k else ""))

    bar = Progress(n, label="  ")
    nmac = 0
    rollout_slots = 0
    try:
        for j in range(n):
            i = start + j
            cfg = config_for_episode(args.preset, i)
            engine = SimulationEngine(cfg, episode_index=i)
            ep, meta = engine.run_episode()
            writer.write_episode(ep, meta)
            nmac += int(bool(meta.nmac_occurred))
            if k:
                rollout_slots += _write_branch_rollouts(
                    writer, engine, i, k, float(args.branch_t))
            bar.update(j + 1, suffix=f"nmac {nmac}")
        bar.done(suffix=f"nmac {nmac}")
        # seed_block() returns the half-open block [start, end) of this split;
        # add_split wants a scalar base, so record the start and keep the full
        # block alongside it as provenance (doc 8.1: the splits are separated
        # by seed block AND by parameter axis; the manifest must show both).
        block = seed_block(args.preset)
        writer.add_split(args.preset, n_episodes=n,
                         seed_base=int(block[0]),
                         seed_block=[int(block[0]), int(block[1])],
                         episode_range=[start, start + n - 1],
                         notes=f"generated by main.py generate --tag {args.tag}")
    except BaseException as exc:
        # A run that died halfway must NOT be stamped 'complete'. An
        # unconditional close() in a finally block would do exactly that, and
        # a later training job would silently train on a truncated dataset.
        # BaseException, not Exception, so that a Ctrl-C during a 5000-episode
        # build -- much the likeliest way this loop ends early -- is also
        # recorded honestly. The episodes already written stay readable:
        # index.jsonl is append-only and fsynced per episode.
        writer.close(status="aborted")
        print(f"\nABORTED after {bar.n}/{n} episodes "
              f"({type(exc).__name__}); run marked 'aborted': {run_dir}",
              file=sys.stderr)
        raise
    writer.close()

    print(f"\nepisodes written : {n}"
          f"   ({nmac} with an NMAC, {100.0 * nmac / max(1, n):.0f}%)")
    if k:
        print(f"rollout sets     : {n} x {rollout_slots // max(1, n)} agent slots"
              f" x {k} branches")
    print(f"run written      : {run_dir}")
    print("\nnext:")
    print(f"  python main.py verify   --run latest")
    print(f"  python main.py diagnose --run latest")
    return 0


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------


def _episode_source(preset: str, i: int) -> tuple[str, int]:
    """Resolve episode ``i`` of a ``--preset`` selection to ``(split, index)``.

    ``'all'`` round-robins the four splits. That is the only selection over
    which the dataset gate is meaningful: doc 8.1 deliberately keeps the HARD
    classes (BIRD, UNKNOWN, SEARCH_WANDER, REACTIVE_AVOID) out of training, so
    a diagnosis of ``train`` alone has an empty HARD band -- correctly, but it
    cannot answer "do the hard classes defeat CV?".

    Args:
        preset: a split name, or ``'all'``.
        i: episode counter [-].

    Returns:
        ``(split_name, episode_index_within_that_split)``.
    """
    from scenario.presets import PRESET_NAMES

    if preset != "all":
        return preset, i
    return PRESET_NAMES[i % len(PRESET_NAMES)], i // len(PRESET_NAMES)


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Run the doc 8.3 CV difficulty diagnostic and print the full report.

    Doc 8.3 gives the CV baseline two jobs. Job one is the performance floor
    every reviewer asks about first. Job two -- the one this subcommand serves
    -- is data-quality diagnosis: CV error too small means the data is too
    easy (add turns), too large means it is too random (lengthen manoeuvre
    dwell), and the target is that the 15 s median lands within the same order
    of magnitude as the NMAC radius.

    Args:
        args: parsed arguments (``run`` / ``preset`` / ``episodes`` /
            ``velocity_source`` / ``json``).

    Returns:
        Process exit code [-]: 0 when the dataset gate passes or is not
        applicable to this data, 1 when it genuinely fails.
    """
    from analysis import cv_baseline as cvb

    if args.run:
        run_dir = _resolve_run(args.run)
        report = cvb.run_diagnostic(
            run_dir,
            velocity_source=args.velocity_source,
            max_episodes=args.episodes if args.episodes > 0 else None,
        )
    else:
        from scenario.presets import config_for_episode
        from simulation.episode import run_episode

        merged = cvb.CVErrors.empty(cvb.DEFAULT_HORIZONS_S)
        n = int(args.episodes) if args.episodes > 0 else 20
        bar = Progress(n, label="  ")
        for i in range(n):
            preset, index = _episode_source(args.preset, i)
            cfg = config_for_episode(preset, index)
            ep, _meta = run_episode(cfg, episode_index=i, validate=False)
            merged.extend(cvb.cv_errors(
                ep, cvb.DEFAULT_HORIZONS_S, velocity_source=args.velocity_source))
            bar.update(i + 1, suffix=preset)
        bar.done()
        report = cvb.diagnose(merged)
        report["preset"] = str(args.preset)
        report["velocity_source"] = str(args.velocity_source)

    print(cvb.format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwritten: {args.json}")

    gate = report.get("gate") or {}
    status = gate.get("status", "PASS" if gate.get("passed") else "FAIL")
    if status == "N/A":
        print("\nnote: the gate is not applicable to single-band data. To "
              "evaluate it,\n      run:  python main.py diagnose --preset all "
              "--episodes 24")
    return 0 if status in ("PASS", "N/A") else 1


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-open a written run and check it against the storage contract.

    Args:
        args: parsed arguments (``run`` / ``shallow`` / ``episodes``).

    Returns:
        Process exit code [-]: 0 when the run is clean, 1 when it is not.
    """
    from storage.manifest import summarize_run, verify_run

    run_dir = _resolve_run(args.run)
    print(f"verifying: {run_dir}")
    problems = verify_run(run_dir, deep=not args.shallow,
                          max_episodes=args.episodes if args.episodes > 0 else None)
    summary = {}
    try:
        summary = summarize_run(run_dir)
    except Exception as exc:                      # noqa: BLE001 - report, not raise
        print(f"  (summary unavailable: {type(exc).__name__}: {exc})")

    if summary:
        # Surface the manifest status. verify_run() checks that whatever IS on
        # disk is self-consistent, which a half-finished run can perfectly well
        # be -- so "consistent" must not be reported as "finished".
        status = str(summary.get("status", "unknown"))
        note = "" if status == "complete" else "   <-- NOT a finished dataset"
        print(f"  status        : {status}{note}")
        print(f"  episodes      : {summary.get('n_episodes', '?')}")
        print(f"  NMAC rate     : {_pct(summary.get('nmac_rate'))}")
        print(f"  DWC violation : {_pct(summary.get('dwc_rate'))}")
        sep = summary.get("min_separation_m") or {}
        if isinstance(sep, dict) and sep:
            print(f"  min separation: min {_num(sep.get('min'))} m,"
                  f" median {_num(sep.get('median'))} m,"
                  f" max {_num(sep.get('max'))} m")
        risk = summary.get("risk_level_counts")
        if risk:
            from analysis.metrics import risk_stratification
            from core.types import RiskLevel
            levels: list[int] = []
            for k, v in sorted(risk.items()):
                levels.extend([int(k)] * int(v))
            strat = risk_stratification(levels)
            print(f"  risk levels   : {strat['counts']}")
            if strat["empty"]:
                # Doc 3.4 asks for a risk-STRATIFIED dataset. Empty strata are
                # expected on a handful of episodes and are a real finding on a
                # large run, so this warns rather than failing -- see the
                # RISK_STRATIFICATION note in README.md.
                print(f"  ** WARNING    : risk strata never observed: "
                      f"{', '.join(strat['empty'])} "
                      f"(over {strat['n_episodes']} episodes)")
        for key, label in (("n_rollout_files", "rollout files"),
                           ("n_preview_files", "preview files")):
            if summary.get(key):
                print(f"  {label:<14}: {summary[key]}")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems[:40]:
            print(f"  - {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    status = str(summary.get("status", "unknown")) if summary else "unknown"
    if status == "complete":
        print("\nOK: run is complete and consistent with the storage contract")
    else:
        print(f"\nOK: the {summary.get('n_episodes', '?')} episodes present are "
              f"consistent with the storage contract,\n    but the run status is "
              f"'{status}' -- it is a PARTIAL dataset, not a finished one.")
    return 0


def _pct(value) -> str:
    """Format a 0-1 fraction as a percentage, or ``'?'`` when absent."""
    try:
        return f"{100.0 * float(value):.0f}%"
    except (TypeError, ValueError):
        return "?"


def _num(value) -> str:
    """Format a float to one decimal, or ``'?'`` when absent."""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "?"


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run one episode headless and print a human-readable summary.

    Nothing is written to disk. This is the fastest end-to-end check that the
    whole stack -- scenario build, agent decisions, wind, sensor, tracker,
    logging, metrics -- is wired together and producing sane numbers.

    Args:
        args: parsed arguments (``preset`` / ``seed`` / ``index``).

    Returns:
        Process exit code [-]: 0 on success.
    """
    import numpy as np

    from analysis.cv_baseline import DEFAULT_HORIZONS_S, cv_errors
    from core.constants import LOG_DT, NMAC_RADIUS_M, SIM_DT
    from core.types import INTENT_NAMES, TYPE_NAMES, GeometryType, RiskLevel
    from scenario.presets import config_for_episode
    from simulation.episode import run_episode
    from storage.schema import validate as validate_arrays

    cfg = config_for_episode(args.preset, int(args.index))
    seed = int(args.seed) if args.seed is not None else int(cfg.seed)
    t0 = time.time()
    ep, meta = run_episode(cfg, seed=seed, episode_index=int(args.index))
    dt = time.time() - t0

    n_slots = int(ep.gt_valid.shape[1])
    used = [j for j in range(n_slots) if bool(ep.gt_valid[:, j].any())]
    det_rate = (float(ep.obs_detected[ep.gt_valid].mean())
                if ep.gt_valid.any() else float("nan"))

    print("PRISM dataGen -- headless episode demo")
    print("=" * 74)
    print(f"preset          : {args.preset}    seed: {seed}")
    print(f"wall time       : {dt:.2f} s")
    print(f"duration        : {meta.duration_s:g} s at {1 / SIM_DT:g} Hz "
          f"integration, logged at {1 / LOG_DT:g} Hz")
    print(f"logged steps    : {ep.t.shape[0]}   "
          f"t = {ep.t[0]:g} .. {ep.t[-1]:g} s")
    print(f"agents          : {meta.n_agents} traffic + 1 ownship "
          f"({len(used)} slots used of {n_slots})")
    print(f"density         : {meta.density:g} /km^2      "
          f"wind: {meta.wind_speed:.1f} m/s from "
          f"{np.degrees(meta.wind_dir):.0f} deg (math convention)")

    print("\nTRAFFIC MIX")
    print("-" * 74)
    plat = meta.platform_counts or {}
    inte = meta.intent_counts or {}
    print("  platforms : " + ", ".join(
        f"{TYPE_NAMES.get(int(k), k)} x{v}" for k, v in sorted(plat.items())) or "  (none)")
    print("  intents   : " + ", ".join(
        f"{INTENT_NAMES.get(int(k), k)} x{v}" for k, v in sorted(inte.items())) or "  (none)")

    print("\nSAFETY (doc 9.4)")
    print("-" * 74)
    # geometry_type / cpa_dist / tcpa are PER DESIGNED ENCOUNTER (doc 3.4
    # inverse design samples 1-3 of them), so they are parallel lists.
    geo = list(meta.geometry_type or [])
    cpa = list(meta.cpa_dist or [])
    tcp = list(meta.tcpa or [])
    if geo:
        for j, g in enumerate(geo):
            name = GeometryType(int(g)).name
            d = _num(cpa[j]) if j < len(cpa) else "?"
            s = _num(tcp[j]) if j < len(tcp) else "?"
            print(f"  designed encounter {j}: {name:<11} "
                  f"cpa*={d:>7} m   tcpa*={s:>6} s   (DESIGN INTENT; agents "
                  f"then evolve freely)")
    else:
        print("  designed encounter : none")
    print(f"  realised min sep   : {meta.min_separation_m:.1f} m "
          f"(NMAC radius {NMAC_RADIUS_M:.1f} m)")
    print(f"  NMAC occurred      : {bool(meta.nmac_occurred)}")
    print(f"  DWC violated       : {bool(meta.dwc_violated)}")
    print(f"  risk level         : {meta.risk_level} "
          f"({RiskLevel(int(meta.risk_level)).name})")

    print("\nSENSOR (doc 3.5)")
    print("-" * 74)
    print(f"  detected fraction of live agent-steps : {det_rate * 100:.1f}%")
    print(f"  obs_pos NaN where undetected          : "
          f"{bool(np.isnan(ep.obs_pos[~ep.obs_detected]).all())}  "
          f"(misses are masked, never interpolated)")
    live_tracks = int(np.unique(ep.obs_track_id[ep.obs_track_id >= 0]).size)
    print(f"  distinct track ids opened             : {live_tracks}")

    err = cv_errors(ep, DEFAULT_HORIZONS_S)
    print("\nCV BASELINE ON THIS EPISODE (doc 8.3)")
    print("-" * 74)
    for h in DEFAULT_HORIZONS_S:
        e = np.asarray(err.errors.get(float(h), np.zeros(0)))
        if e.size:
            print(f"  {h:>4.0f} s : n={e.size:<6d} median {np.median(e):7.1f} m"
                  f"   p90 {np.percentile(e, 90):7.1f} m")

    problems = validate_arrays(ep)
    print("\nSTORAGE CONTRACT (doc 4)")
    print("-" * 74)
    print(f"  schema.validate : {'OK' if not problems else problems}")
    print(f"  meta.validate   : {'OK' if not meta.validate() else meta.validate()}")
    print("=" * 74)
    return 0 if not problems else 1


# --------------------------------------------------------------------------
# platforms
# --------------------------------------------------------------------------


def _cmd_platforms(args: argparse.Namespace) -> int:
    """Print the platform catalogue with its manoeuvre-stability profiles."""
    from agents.platforms import INTENT_DWELL_MULTIPLIER, envelope_summary_table
    from analysis.cv_baseline import band_of_intent, band_of_platform
    from core.types import INTENT_NAMES, IntentType

    rows = envelope_summary_table()
    print("PLATFORM CATALOGUE (mid-band envelope + stability profile)")
    print("-" * 108)
    print(f"{'platform':<16}{'split':<7}{'v[m/s]':>12}{'turn@cru':>10}"
          f"{'dwell_min':>10}{'dwell_mu':>10}{'turn_frac':>12}{'auth':>7}"
          f"{'diag':>7}{'expect':>8}")
    for r in rows:
        print(f"{r['label']:<16}{r['split']:<7}"
              f"{r['v_min_ms']:>5.0f}-{r['v_max_ms']:<6.0f}"
              f"{r['turn_rate_at_cruise_deg_s']:>10.1f}"
              f"{r['dwell_min_s']:>10.1f}{r['dwell_mean_s']:>10.1f}"
              f"{r['turn_frac_lo']:>7.2f}-{r['turn_frac_hi']:<4.2f}"
              f"{r['mode_authority']:>7.2f}{r['diag_bias']:>7.2f}"
              f"{band_of_platform(r['platform']):>8}")
    print()
    print("INTENT DWELL MULTIPLIERS")
    print("-" * 108)
    for intent in IntentType:
        print(f"  {INTENT_NAMES[int(intent)]:<20}"
              f"x{INTENT_DWELL_MULTIPLIER[int(intent)]:<6.2f}"
              f"expected {band_of_intent(int(intent))}")
    print()
    print("dwell_min / dwell_mu [s] are the PLATFORM manoeuvre-commitment floor")
    print("and mean; the mean is multiplied by the intent factor above, and the")
    print("floor is applied after that multiplication.")
    return 0


# --------------------------------------------------------------------------
# gui / selftest
# --------------------------------------------------------------------------


def _cmd_gui(args: argparse.Namespace) -> int:
    """Serve the local GUI (stdlib http.server; loopback only)."""
    from gui.server import main as server_main

    argv = ["--port", str(args.port)]
    if not args.no_open:
        argv.append("--open")
    return int(server_main(argv))


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Run every ``tests/test_*.py`` in a subprocess and summarise.

    Each test file is a standalone runner that bootstraps ``sys.path`` itself,
    so they are executed as scripts rather than collected -- no pytest, no
    import side effects between suites.

    Args:
        args: parsed arguments (``pattern``).

    Returns:
        Process exit code [-]: 0 when every file passes.
    """
    import glob
    import subprocess

    paths = sorted(glob.glob(os.path.join(_ROOT, "tests", "test_*.py")))
    if args.pattern:
        paths = [p for p in paths if args.pattern in os.path.basename(p)]
    if not paths:
        print("no test files matched", file=sys.stderr)
        return 2
    failures: list[str] = []
    t0 = time.time()
    for path in paths:
        name = os.path.basename(path)
        started = time.time()
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, cwd=_ROOT)
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"{status}  {name:<24} {time.time() - started:6.1f}s  "
              f"{tail[-1] if tail else ''}")
        if proc.returncode != 0:
            failures.append(name)
            print((proc.stdout or "") + (proc.stderr or ""))
    print(f"\n{len(paths) - len(failures)}/{len(paths)} test files passed "
          f"in {time.time() - t0:.1f}s")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _generate_split(preset: str, n: int, tag: str, rollouts: int,
                    branch_t: float) -> tuple[str, dict]:
    """Generate one split (the ``generate`` loop, callable from ``dataset``).

    Returns:
        ``(run_dir, stats)`` with stats keys ``n / nmac / reactions``.
    """
    from scenario.presets import config_for_episode, make_preset, seed_block
    from simulation.engine import SimulationEngine
    from storage.writer import DatasetWriter

    writer = DatasetWriter()
    run_dir = writer.open_run(make_preset(preset), preset=preset, tag=tag)
    bar = Progress(n, label=f"  {preset:9s}")
    nmac = 0
    reactions = 0
    try:
        for i in range(n):
            cfg = config_for_episode(preset, i)
            engine = SimulationEngine(cfg, episode_index=i)
            ep, meta = engine.run_episode()
            writer.write_episode(ep, meta)
            nmac += int(bool(meta.nmac_occurred))
            reactions += int(meta.reaction_events)
            if rollouts:
                _write_branch_rollouts(writer, engine, i, rollouts,
                                       float(branch_t))
            bar.update(i + 1, suffix=f"nmac {nmac} react {reactions}")
        bar.done(suffix=f"nmac {nmac} react {reactions}")
        block = seed_block(preset)
        writer.add_split(preset, n_episodes=n, seed_base=int(block[0]),
                         seed_block=[int(block[0]), int(block[1])],
                         episode_range=[0, n - 1],
                         notes=f"generated by main.py dataset --tag {tag}")
    except BaseException:
        writer.close(status="aborted")
        raise
    writer.close()
    return run_dir, {"n": n, "nmac": nmac, "reactions": reactions}


def _cmd_dataset(args: argparse.Namespace) -> int:
    """One-command, training-ready dataset build (doc 10 steps 1-3).

    Generates all four doc 8.1 splits, writes branch rollouts for the eval
    splits, deep-verifies every run, runs the CV difficulty gate over the
    UNION of the splits, asks modelTraining to actually consume a sample
    batch, and stamps everything into one collection manifest. Exit code 0
    only when every stage is green -- the user never has to look inside.
    """
    import json as _json
    import subprocess
    import time as _time

    from analysis import cv_baseline as cvb
    from core.constants import LOGS_ROOT
    from storage.manifest import verify_run
    from storage.writer import iter_run

    n_train = int(args.episodes)
    n_valid = max(1, int(round(n_train * float(args.valid_frac))))
    n_test = max(1, int(round(n_train * float(args.test_frac))))
    n_ood = max(1, int(round(n_train * float(args.ood_frac))))
    k = int(args.rollouts)
    plan = [("train", n_train, 0), ("valid", n_valid, k),
            ("test_id", n_test, k), ("test_ood", n_ood, k)]

    print("PRISM auto dataset build")
    print(f"  episodes  : train {n_train} / valid {n_valid} / "
          f"test_id {n_test} / test_ood {n_ood}")
    print(f"  rollouts  : {k} branches @ t={args.branch_t:g} s (eval splits)")
    t_start = _time.time()

    # -- 1. generate ------------------------------------------------------
    runs: dict[str, str] = {}
    stats: dict[str, dict] = {}
    for preset, n, kk in plan:
        run_dir, st = _generate_split(preset, n, args.tag, kk, args.branch_t)
        runs[preset] = run_dir
        stats[preset] = st

    # -- 2. deep verify every run ----------------------------------------
    print("\nverifying:")
    verify_problems: dict[str, list[str]] = {}
    for preset, run_dir in runs.items():
        problems = verify_run(run_dir)
        verify_problems[preset] = problems
        print(f"  {preset:9s}: {'OK' if not problems else problems[:3]}")
    verify_ok = not any(verify_problems.values())

    # -- 3. CV gate over the union of all four splits ---------------------
    print("\nCV difficulty gate (union of all splits):")
    merged = cvb.CVErrors.empty()
    for preset, run_dir in runs.items():
        for _meta, ep in iter_run(run_dir):
            merged = merged.extend(cvb.cv_errors(ep))
    report = cvb.diagnose(merged)
    gate = report.get("gate", {})
    gate_ok = bool(gate.get("passed"))
    sep = report.get("band_separation", {})
    print(f"  gate      : {'PASS' if gate_ok else 'FAIL'}")
    for axis in ("platform", "intent"):
        rec = sep.get(axis) or {}
        if "ratio" in rec:
            print(f"  {axis:9s}: band separation {rec['ratio']:.2f}x")
    for reason in gate.get("reasons", [])[:6]:
        print(f"  ! {reason}")

    # -- 4. can the trainer actually eat this? ----------------------------
    print("\ntraining-readiness (modelTraining consumes a real batch):")
    trainer_main = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "modelTraining", "main.py")
    training_ready = False
    training_note = "modelTraining not found"
    if os.path.exists(trainer_main) and not args.skip_training_check:
        cmd = [sys.executable, trainer_main, "validate-data",
               "--run", runs["train"], "--run", runs["valid"]]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=600)
            tail = (out.stdout or "").strip().splitlines()
            payload = _json.loads(tail[-1]) if tail else {}
            training_ready = bool(payload.get("ok"))
            training_note = (f"{payload.get('n_samples', '?')} samples, "
                             f"batch {payload.get('batch_shape', '?')}, "
                             f"forward OK" if training_ready
                             else (out.stderr or out.stdout or "")[-400:])
        except Exception as exc:  # noqa: BLE001 - readiness must never crash the build
            training_note = f"{type(exc).__name__}: {exc}"
    elif args.skip_training_check:
        training_note = "skipped by flag"
    print(f"  {'OK' if training_ready else 'NOT READY'}  {training_note}")

    # -- 5. collection manifest -------------------------------------------
    collection = {
        "tag": str(args.tag),
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(_time.time() - t_start, 1),
        "episodes": {p: s["n"] for p, s in stats.items()},
        "nmac": {p: s["nmac"] for p, s in stats.items()},
        "reaction_events": {p: s["reactions"] for p, s in stats.items()},
        "rollout_branches": k,
        "runs": runs,
        "verify_ok": verify_ok,
        "verify_problems": {p: v for p, v in verify_problems.items() if v},
        "gate": gate,
        "band_separation": sep,
        "training_ready": training_ready,
        "training_note": training_note,
    }
    name = f"dataset_{_time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.json"
    coll_path = os.path.join(LOGS_ROOT, name).replace("\\", "/")
    with open(coll_path, "w", encoding="utf-8") as f:
        _json.dump(collection, f, indent=2, ensure_ascii=False)

    all_ok = verify_ok and gate_ok and training_ready
    print("\n" + "=" * 60)
    print(f"DATASET {'READY FOR TRAINING' if all_ok else 'NOT READY'}"
          f"   ({collection['wall_s']:.0f} s)")
    print(f"manifest: {coll_path}")
    if all_ok:
        print("next: open the modelTraining GUI -- the four runs are already "
              "discoverable with train/val roles.")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    from scenario.presets import PRESET_NAMES

    ap = argparse.ArgumentParser(
        prog="main.py",
        description="PRISM dataGen: 2D traffic simulator and "
                    "trajectory-prediction dataset generator. "
                    "With no arguments, starts the GUI.")
    sub = ap.add_subparsers(dest="command", required=True)

    u = sub.add_parser("gui", help="serve the local inspection GUI (default)")
    u.add_argument("--port", type=int, default=8770)
    u.add_argument("--no-open", dest="no_open", action="store_true",
                   help="do not launch a browser")
    u.set_defaults(func=_cmd_gui)

    g = sub.add_parser("generate", help="write a dataset run to LOGS_ROOT")
    g.add_argument("--preset", type=str, default="train", choices=PRESET_NAMES)
    g.add_argument("--episodes", type=int, default=100)
    g.add_argument("--start-index", dest="start_index", type=int, default=0,
                   help="first episode index; lets parallel shard processes "
                        "cover disjoint index ranges of one split")
    g.add_argument("--tag", type=str, default="run")
    g.add_argument("--rollouts", type=int, default=0, metavar="K",
                   help="also write K branch rollouts per episode (doc 3.3 "
                        "true distribution; doc 3.7 asks 50-100, eval set only)")
    g.add_argument("--branch-t", dest="branch_t", type=float, default=30.0,
                   help="fork time [s] for --rollouts (default 30 s, which "
                        "leaves a full 15 s horizon inside a 60 s episode)")
    g.set_defaults(func=_cmd_generate)

    ds = sub.add_parser(
        "dataset",
        help="one-command training-ready dataset: all four splits + rollouts "
             "+ verify + CV gate + trainer readiness check")
    ds.add_argument("--episodes", type=int, default=1000,
                    help="TRAIN episodes; the other splits scale from this")
    ds.add_argument("--valid-frac", dest="valid_frac", type=float, default=0.15)
    ds.add_argument("--test-frac", dest="test_frac", type=float, default=0.15)
    ds.add_argument("--ood-frac", dest="ood_frac", type=float, default=0.20)
    ds.add_argument("--rollouts", type=int, default=60, metavar="K",
                    help="branch rollouts per eval episode (doc 3.7: 50-100)")
    ds.add_argument("--branch-t", dest="branch_t", type=float, default=30.0)
    ds.add_argument("--tag", type=str, default="auto")
    ds.add_argument("--skip-training-check", dest="skip_training_check",
                    action="store_true")
    ds.set_defaults(func=_cmd_dataset)

    d = sub.add_parser("diagnose", help="CV difficulty diagnostic (doc 8.3)")
    d.add_argument("--run", type=str, default="",
                   help="score a written run: 'latest', a run id, or a path")
    d.add_argument("--preset", type=str, default="all",
                   choices=tuple(PRESET_NAMES) + ("all",),
                   help="split to generate from when --run is not given; "
                        "'all' round-robins every split, which is the only "
                        "selection containing both difficulty bands")
    d.add_argument("--episodes", type=int, default=0,
                   help="episodes to generate (default 20), or a cap on how "
                        "many to read from --run (default all)")
    d.add_argument("--velocity-source", dest="velocity_source", type=str,
                   default="gt", choices=("gt", "diff"),
                   help="'gt' is the best case for CV and the honest diagnostic")
    d.add_argument("--json", type=str, default="",
                   help="also write the full report as JSON to this path")
    d.set_defaults(func=_cmd_diagnose)

    v = sub.add_parser("verify", help="check a written run against the schema")
    v.add_argument("--run", type=str, default="latest",
                   help="'latest', a run id, or a path")
    v.add_argument("--shallow", action="store_true",
                   help="index-only check; do not reopen every episode NPZ")
    v.add_argument("--episodes", type=int, default=0,
                   help="cap how many episodes to deep-check (default all)")
    v.set_defaults(func=_cmd_verify)

    m = sub.add_parser("demo", help="run one episode headless and summarise")
    m.add_argument("--preset", type=str, default="train", choices=PRESET_NAMES)
    m.add_argument("--seed", type=int, default=None)
    m.add_argument("--index", type=int, default=0,
                   help="episode index within the preset")
    m.set_defaults(func=_cmd_demo)

    p = sub.add_parser("platforms", help="print the platform/stability catalogue")
    p.set_defaults(func=_cmd_platforms)

    t = sub.add_parser("selftest", help="run the test suite")
    t.add_argument("-k", dest="pattern", type=str, default="",
                   help="only test files whose name contains this substring")
    t.set_defaults(func=_cmd_selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: argument vector; ``sys.argv[1:]`` when omitted. An empty vector
            starts the GUI, which is the useful thing to do when someone
            double-clicks the file or types ``python main.py``.

    Returns:
        Process exit code [-].
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["gui"]
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
