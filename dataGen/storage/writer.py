"""Crash-safe dataset writer and loader for PRISM episode runs.

Everything this module writes lands under ``<LOGS_ROOT>/<run_id>/`` with the
layout owned by :mod:`storage.manifest`. Two properties are non-negotiable:

**Crash safety.** Every file that must be read back in one piece (manifest,
config, README, episode NPZ, rollout NPZ, preview JSON) is written to
``<path>.tmp``, flushed, ``fsync``-ed and then moved into place with
:func:`os.replace`, which is atomic on both NTFS and POSIX. ``index.jsonl`` is
the one append-only file: each :class:`~storage.schema.EpisodeMeta` is appended
as a single line and flushed + fsync-ed immediately, so a run killed at any
moment leaves a valid, truncation-free partial dataset whose index lines all
refer to episode files that were already fully written (the episode is written
*before* its index line).

**Explicit GT / OBS separation (doc 4).** The NPZ key groups are the schema
names themselves: ``gt_*`` for the ground-truth layer, ``obs_*`` for the
observation layer, ``own_*`` for the ownship, plus ``t`` and ``agent_ids``.
Nothing merges the two layers, and the observation layer keeps its NaN gaps --
a miss is never interpolated (doc 3.5).

Units: positions [m], velocities [m/s], headings [rad] CCW from +X (East),
times [s], covariances [m^2].
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Iterator, Mapping

import numpy as np

from core.constants import LOG_DT, LOGS_ROOT
from core.types import (
    GEOMETRY_NAMES,
    INTENT_NAMES,
    MODE_NAMES,
    RISK_NAMES,
    TYPE_NAMES,
)
from scenario.config import ScenarioConfig, save_scenario_config
from storage import schema
from storage.manifest import (
    CONFIG_NAME,
    DATASET_FORMAT_VERSION,
    EPISODES_DIR,
    INDEX_NAME,
    MANIFEST_NAME,
    PREVIEW_DIR,
    README_NAME,
    ROLLOUTS_DIR,
    add_split,
    build_manifest,
    episode_index_from_name,
    episode_path,
    make_run_id,
    preview_path,
    read_manifest,
    rollout_path,
    write_manifest,
)
from storage.schema import EpisodeArrays, EpisodeMeta

__all__ = [
    "ROLLOUT_ARRAY_SPEC",
    "DatasetWriter",
    "read_episode",
    "read_rollouts",
    "read_index",
    "read_preview",
    "iter_run",
    "list_runs",
    "latest_run",
    "build_preview",
    "build_readme",
]


# --------------------------------------------------------------------------
# Branch-rollout array contract
# --------------------------------------------------------------------------

#: Recommended layout of a branch-rollout bundle (doc 3.3: "same initial
#: condition and same past track, re-rolled N=50..100 times with a different
#: seed; this empirical distribution is the ground truth for calibration
#: scoring"). Symbols:
#:     B -- number of branch points stored for this episode
#:     K -- number of rollouts per branch point (50..100)
#:     H -- rollout horizon in logged steps (normally PRED_STEPS = 30)
#:
#: :meth:`DatasetWriter.write_rollouts` accepts ANY mapping of numpy arrays so
#: that :mod:`simulation.rollout` stays free to add fields; this spec is the
#: shape/dtype contract that :func:`validate_rollouts` checks when the
#: recommended keys are present.
ROLLOUT_ARRAY_SPEC: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "origin_step": (np.int32, ("B",)),      # logged step index the branch starts at
    "agent_slot": (np.int32, ("B",)),       # EpisodeArrays slot index; -1 = whole world
    "branch_seed": (np.int64, ("B", "K")),  # seed used for each rollout
    "pos": (np.float32, ("B", "K", "H", 2)),   # future positions [m]
    "vel": (np.float32, ("B", "K", "H", 2)),   # future ground velocities [m/s]
    "psi": (np.float32, ("B", "K", "H")),      # future headings [rad]
    "mode": (np.int8, ("B", "K", "H")),        # ManeuverMode, -1 where invalid
    "valid": (np.bool_, ("B", "K", "H")),      # rollout sample exists
}


def validate_rollouts(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Check a rollout bundle against :data:`ROLLOUT_ARRAY_SPEC`.

    Only keys present in the mapping are checked, so a partial bundle (for
    example positions only) is accepted; unknown keys are ignored. All present
    recommended keys must agree on the symbolic dimensions B, K and H.

    Args:
        arrays: mapping of array name to numpy array.

    Returns:
        A list of problems; empty means the bundle is contract-compliant.
    """
    p: list[str] = []
    dims: dict[str, int] = {}
    for name, (dtype, shape) in ROLLOUT_ARRAY_SPEC.items():
        if name not in arrays:
            continue
        arr = np.asarray(arrays[name])
        if arr.dtype != np.dtype(dtype):
            p.append(f"rollouts.{name}: dtype {arr.dtype}, expected {np.dtype(dtype)}")
        if arr.ndim != len(shape):
            p.append(f"rollouts.{name}: ndim {arr.ndim}, expected {len(shape)}")
            continue
        for axis, sym in enumerate(shape):
            got = int(arr.shape[axis])
            if isinstance(sym, int):
                if got != sym:
                    p.append(f"rollouts.{name}: axis {axis} is {got}, expected {sym}")
            elif sym in dims:
                if dims[sym] != got:
                    p.append(
                        f"rollouts.{name}: axis {axis} is {got}, but {sym} was "
                        f"already fixed to {dims[sym]} by another array"
                    )
            else:
                dims[sym] = got
    return p


# --------------------------------------------------------------------------
# Low-level atomic IO
# --------------------------------------------------------------------------


def _join(*parts: str) -> str:
    """Join path components and normalise to forward slashes."""
    return os.path.join(*parts).replace("\\", "/")


def _atomic_write_text(path: str, text: str) -> None:
    """Write UTF-8 text atomically (``.tmp`` + fsync + :func:`os.replace`)."""
    tmp = path + ".tmp"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_savez(path: str, arrays: Mapping[str, np.ndarray],
                  compressed: bool = True) -> None:
    """Write a (compressed) NPZ atomically.

    ``np.savez*`` appends ``.npz`` when the target name lacks it, so the
    temporary file is named ``<path>.tmp.npz`` and then moved onto ``<path>``.

    Args:
        path: final destination, ending in ``.npz``.
        arrays: mapping of key to array; dtypes are preserved exactly.
        compressed: use deflate (True) or store raw (False).
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp, "wb") as fh:
        saver(fh, **dict(arrays))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_number(value: float, nd: int = 2) -> float | None:
    """Round a float for JSON; non-finite values become ``None``.

    JSON has no NaN literal that ``JSON.parse`` accepts, so the GUI preview
    encodes "no data" as ``null``. The NaN policy itself is unaffected -- the
    authoritative arrays keep real NaNs (doc 3.5).
    """
    v = float(value)
    return None if not math.isfinite(v) else round(v, nd)


# --------------------------------------------------------------------------
# Preview + README builders
# --------------------------------------------------------------------------


def build_preview(
    ep: EpisodeArrays,
    meta: EpisodeMeta,
    stride: int = 2,
    pos_decimals: int = 1,
    angle_decimals: int = 3,
) -> dict[str, Any]:
    """Build a small, JSON-safe, decimated replay record for the Canvas GUI.

    The GUI must load a run without numpy, so the preview flattens the episode
    into plain lists. Every ``stride``-th logged step is kept (``stride=2`` ->
    1 Hz from the 2 Hz log), and continuous values are rounded; NaN becomes
    ``null``.

    Args:
        ep: the episode arrays.
        meta: the matching metadata.
        stride: logged-step decimation factor [-]; 1 keeps every step.
        pos_decimals: decimal places kept for positions [m] and velocities.
        angle_decimals: decimal places kept for headings [rad].

    Returns:
        A JSON-serialisable dict with ``t``, ownship track, per-slot GT tracks
        and per-slot observation tracks.
    """
    stride = max(1, int(stride))
    idx = np.arange(0, ep.n_steps, stride, dtype=np.int64)
    t = ep.t.astype(np.float64)[idx]
    own_pos = ep.own_pos.astype(np.float64)[idx]
    own_vel = ep.own_vel.astype(np.float64)[idx]
    own_psi = ep.own_psi.astype(np.float64)[idx]

    used = ep.used_slots()
    agents: list[dict[str, Any]] = []
    for slot in range(ep.n_agents):
        if not bool(used[slot]):
            continue
        valid = ep.gt_valid[idx, slot]
        gpos = ep.gt_pos.astype(np.float64)[idx, slot]
        gvel = ep.gt_vel.astype(np.float64)[idx, slot]
        gpsi = ep.gt_psi.astype(np.float64)[idx, slot]
        det = ep.obs_detected[idx, slot]
        opos = ep.obs_pos.astype(np.float64)[idx, slot]
        agents.append({
            "slot": int(slot),
            "agent_id": int(ep.agent_ids[slot]),
            "type": int(ep.gt_type[slot]),
            "type_name": TYPE_NAMES.get(int(ep.gt_type[slot]), "?"),
            "valid": [int(v) for v in valid],
            "x": [_json_number(v, pos_decimals) for v in gpos[:, 0]],
            "y": [_json_number(v, pos_decimals) for v in gpos[:, 1]],
            "vx": [_json_number(v, pos_decimals) for v in gvel[:, 0]],
            "vy": [_json_number(v, pos_decimals) for v in gvel[:, 1]],
            "psi": [_json_number(v, angle_decimals) for v in gpsi],
            "mode": [int(v) for v in ep.gt_mode[idx, slot]],
            "intent": [int(v) for v in ep.gt_intent[idx, slot]],
            "detected": [int(v) for v in det],
            "obs_x": [_json_number(v, pos_decimals) for v in opos[:, 0]],
            "obs_y": [_json_number(v, pos_decimals) for v in opos[:, 1]],
            "track_id": [int(v) for v in ep.obs_track_id[idx, slot]],
        })

    return {
        "format_version": DATASET_FORMAT_VERSION,
        "episode_index": int(meta.episode_index),
        "seed": int(meta.seed),
        "duration_s": float(meta.duration_s),
        "log_dt_s": float(LOG_DT),
        "stride": stride,
        "preview_dt_s": float(LOG_DT) * stride,
        "n_steps": int(idx.size),
        "n_agents": int(len(agents)),
        "t": [_json_number(v, 3) for v in t],
        "own": {
            "x": [_json_number(v, pos_decimals) for v in own_pos[:, 0]],
            "y": [_json_number(v, pos_decimals) for v in own_pos[:, 1]],
            "vx": [_json_number(v, pos_decimals) for v in own_vel[:, 0]],
            "vy": [_json_number(v, pos_decimals) for v in own_vel[:, 1]],
            "psi": [_json_number(v, angle_decimals) for v in own_psi],
        },
        "agents": agents,
        "meta": meta.to_dict(),
        "names": {
            "type": {str(k): v for k, v in TYPE_NAMES.items()},
            "mode": {str(k): v for k, v in MODE_NAMES.items()},
            "intent": {str(k): v for k, v in INTENT_NAMES.items()},
            "geometry": {str(k): v for k, v in GEOMETRY_NAMES.items()},
            "risk": {str(k): v for k, v in RISK_NAMES.items()},
        },
    }


def build_readme(run_id: str, config: ScenarioConfig) -> str:
    """Compose the human-readable ``README.txt`` shipped inside every run.

    It documents every file, every array (shape, dtype, unit) and the NaN /
    sentinel policy, so the dataset is interpretable without reading any
    dataGen source.

    Args:
        run_id: the run identifier.
        config: the scenario configuration used for the run.

    Returns:
        The README text (UTF-8, ``\\n`` line endings).
    """
    T = int(config.n_logged_steps())
    unit = {
        "t": "s, since episode start",
        "own_pos": "m, (East, North)",
        "own_vel": "m/s, GROUND velocity (air velocity + wind)",
        "own_psi": "rad, CCW from +X(East), wrapped to (-pi, pi]",
        "gt_valid": "-, agent slot occupied and alive",
        "gt_pos": "m",
        "gt_vel": "m/s, GROUND velocity",
        "gt_psi": "rad, wrapped to (-pi, pi]",
        "gt_mode": "-, ManeuverMode enum (-1 = invalid)",
        "gt_intent": "-, IntentType enum (-1 = invalid)",
        "gt_type": "-, PlatformType enum (-1 = unused slot)",
        "obs_detected": "-, measurement exists at this step",
        "obs_pos": "m, sensor-measured position (NaN where not detected)",
        "obs_R": "m^2, 2x2 measurement covariance in world axes",
        "obs_t_valid": "s, time the measurement was TRUE (earlier than t)",
        "obs_track_id": "-, tracker id (-1 = no live track)",
        "obs_track_age": "steps, track lifetime (0 where no live track)",
        "agent_ids": "-, stable ground-truth agent id (-1 = unused slot)",
    }
    group_of = {
        "own": "OWNSHIP",
        "gt": "GROUND TRUTH layer  (targets / evaluation reference)",
        "obs": "OBSERVATION layer   (model input: noisy, missing, delayed)",
        "index": "INDEX / TIME",
    }
    lines: list[str] = []
    a = lines.append
    a("=" * 78)
    a(f"PRISM dataGen dataset run : {run_id}")
    a(f"format_version            : {DATASET_FORMAT_VERSION}")
    a("=" * 78)
    a("")
    a("This directory is one dataset run produced by $PRISM_ROOT/dataGen.")
    a("The simulator is a 2D low-altitude non-cooperative traffic model; the")
    a("data it produces is the training and evaluation set for a probabilistic")
    a("trajectory prediction model (PRISM, doc sections 3-5).")
    a("")
    a("-" * 78)
    a("FILES")
    a("-" * 78)
    a(f"  {MANIFEST_NAME:<28} full provenance: scenario configuration, master")
    a(f"  {'':<28} seed, per-split provenance, code fingerprint,")
    a(f"  {'':<28} python/numpy versions, wall-clock, episode counts.")
    a(f"  {CONFIG_NAME:<28} the exact ScenarioConfig used, round-trippable")
    a(f"  {'':<28} through scenario.config.load_scenario_config().")
    a(f"  {INDEX_NAME:<28} one JSON object per line, one line per episode:")
    a(f"  {'':<28} the EpisodeMeta table used for stratified splitting")
    a(f"  {'':<28} and analysis (doc 4 'Scenario meta').")
    a(f"  {EPISODES_DIR + '/ep_NNNNNN.npz':<28} the episode arrays (see below), deflate")
    a(f"  {'':<28} compressed. NNNNNN is the episode index.")
    a(f"  {ROLLOUTS_DIR + '/ep_NNNNNN_rollouts.npz':<28}")
    a(f"  {'':<28} branch rollouts for evaluation splits only")
    a(f"  {'':<28} (doc 3.3): same initial condition and same past")
    a(f"  {'':<28} track, re-rolled K times with different decision")
    a(f"  {'':<28} seeds. This empirical distribution is the")
    a(f"  {'':<28} calibration ground truth.")
    a(f"  {PREVIEW_DIR + '/ep_NNNNNN.json':<28} decimated plain-JSON copy for fast GUI replay.")
    a(f"  {'':<28} Lossy: rounded, subsampled, NaN encoded as null.")
    a(f"  {'':<28} NEVER use it for training or evaluation.")
    a(f"  {README_NAME:<28} this file.")
    a("")
    a("-" * 78)
    a("EPISODE ARRAYS   (numpy .npz, load with np.load(path))")
    a("-" * 78)
    a(f"  T = {T} logged steps  ({config.duration_s:g} s at LOG_DT = {LOG_DT:g} s, 2 Hz)")
    a("  N = agent slot capacity for the episode (padding slots exist)")
    a("")
    a("  Doc section 4 requires the ground-truth layer and the observation")
    a("  layer to be stored SEPARATELY. On disk that separation is the key")
    a("  prefix: gt_* is the truth, obs_* is what the ownship sensor saw.")
    a("  The learning input is obs_*; the target and the evaluation reference")
    a("  is gt_*. Never train on gt_* inputs by accident.")
    a("")
    last_group = ""
    for name, (dtype, shape) in schema.EPISODE_ARRAY_SPEC.items():
        g = ("gt" if name.startswith("gt_")
             else "obs" if name.startswith("obs_")
             else "own" if name.startswith("own_") else "index")
        if g != last_group:
            a("")
            a(f"  [{group_of[g]}]")
            last_group = g
        shape_s = "[" + ",".join(str(s) for s in shape) + "]"
        a(f"    {name:<14} {shape_s:<12} {np.dtype(dtype).name:<8} {unit.get(name, '')}")
    a("")
    a("-" * 78)
    a("NaN / SENTINEL POLICY   (enforced by storage.schema.validate)")
    a("-" * 78)
    a("  * gt_pos / gt_vel / gt_psi : finite where gt_valid, NaN elsewhere.")
    a("  * gt_mode / gt_intent      : valid enum where gt_valid, -1 elsewhere.")
    a("  * gt_type / agent_ids      : valid where the slot is ever used,")
    a("                               -1 for slots that are never valid.")
    a("  * obs_pos / obs_R          : finite where obs_detected, NaN elsewhere.")
    a("                               A GAP IS NEVER INTERPOLATED (doc 3.5).")
    a("                               The model must learn 'no information' so")
    a("                               that it widens sigma on a real track drop.")
    a("  * obs_track_id             : >= 0 wherever obs_detected is True; -1")
    a("                               when no live track exists. A track that")
    a("                               is re-acquired after a drop gets a NEW id.")
    a("                               A live track MAY have obs_detected=False")
    a("                               on a missed update (the coast case).")
    a("  * obs_track_age            : >= 0 steps; exactly 0 where track_id=-1.")
    a("  * obs_t_valid              : the time the measurement was actually")
    a("                               true. ALWAYS EARLIER than t: the sensor")
    a("                               runs at 5 Hz behind a 0.2 +- 0.05 s")
    a("                               latency and the log at 2 Hz, so the")
    a("                               value is 0.05-0.78 s stale. obs_R covers")
    a("                               ONLY the polar noise at t_valid, not the")
    a("                               staleness -- propagate the target back to")
    a("                               obs_t_valid before scoring calibration,")
    a("                               or the input layer's own covariance is")
    a("                               ~50% optimistic in variance.")
    a("  Padding slots never carry stale values.")
    a("")
    a("-" * 78)
    a("CONVENTIONS")
    a("-" * 78)
    a("  SI units throughout: metres, m/s, radians, seconds, kilograms.")
    a("  Local tangent plane, +X = East, +Y = North.")
    a("  psi is measured CCW from +X (standard math convention, NOT compass)")
    a("  and wrapped to (-pi, pi]. Positive turn rate = left = CCW.")
    a("  Continuous arrays are float32 on disk; internal computation is")
    a("  float64. Enum labels are int8, ids and counters int32.")
    a("")
    a("-" * 78)
    a("REPRODUCIBILITY")
    a("-" * 78)
    a("  An episode is fully determined by (config.json, episode seed). The")
    a("  seed of every episode is in index.jsonl; the master seed and the")
    a("  source-code SHA-256 fingerprint are in manifest.json.")
    a("  storage.manifest.verify_run(run_dir) re-opens every episode, runs the")
    a("  schema validator and cross-checks index.jsonl against the files on")
    a("  disk. Run it before using a dataset in a paper table.")
    a("")
    a(f"  scenario name  : {config.name}")
    a(f"  master seed    : {config.seed}")
    a(f"  duration       : {config.duration_s:g} s per episode")
    a(f"  area           : {config.area_km2:g} km^2 "
      f"(square side {config.region_side_m():.1f} m)")
    a(f"  density        : {config.density_per_km2:g} aircraft/km^2")
    a("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


class DatasetWriter:
    """Incremental, crash-safe writer for one dataset run.

    Typical use::

        w = DatasetWriter()
        run_dir = w.open_run(cfg, preset='urban_dense', tag='train')
        for i, (ep, meta) in enumerate(generator):
            w.write_episode(ep, meta)
        w.close()

    or as a context manager, which closes with status ``'aborted'`` if the
    body raised::

        with DatasetWriter() as w:
            w.open_run(cfg, preset='urban_dense', tag='train')
            ...

    The instance keeps ``index.jsonl`` open in append mode and flushes +
    fsyncs after every episode, so killing the process leaves a valid partial
    dataset: each index line is complete and its episode file already exists.
    """

    def __init__(self, logs_root: str = LOGS_ROOT) -> None:
        """Create a writer.

        Args:
            logs_root: dataset output root; defaults to
                :data:`core.constants.LOGS_ROOT` (``$PRISM_ROOT/Logs``).
        """
        self.logs_root: str = _join(logs_root)
        self.run_dir: str = ""
        self.run_id: str = ""
        self.manifest: dict[str, Any] = {}
        self.config: ScenarioConfig | None = None
        self._index_fh: Any = None
        self._t_open: float = 0.0
        self._n_episodes: int = 0
        self._n_index_lines: int = 0
        self._n_rollouts: int = 0
        self._n_previews: int = 0
        self._written_indices: set[int] = set()

    # -- lifecycle --------------------------------------------------------

    def open_run(
        self,
        config: ScenarioConfig,
        preset: str = "default",
        tag: str = "run",
        run_id: str | None = None,
        master_seed: int | None = None,
        splits: Mapping[str, Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
        include_fingerprint: bool = True,
    ) -> str:
        """Create the run directory tree and write the initial provenance files.

        Writes ``manifest.json`` (status ``'open'``), ``config.json`` and
        ``README.txt`` immediately, so a run that dies after one episode is
        still self-describing.

        Args:
            config: the scenario configuration for this run.
            preset: preset name, embedded in ``run_id``.
            tag: free-form tag, embedded in ``run_id``.
            run_id: explicit run id; when None it is built from the current
                local time, ``preset`` and ``tag``.
            master_seed: top-level seed [-]; defaults to ``config.seed``.
            splits: initial per-split provenance table.
            extra: arbitrary extra JSON-safe manifest fields.
            include_fingerprint: hash the dataGen source into the manifest.

        Returns:
            The absolute run directory path (forward slashes).

        Raises:
            RuntimeError: if a run is already open on this writer.
            FileExistsError: if the target run directory already exists.
        """
        if self._index_fh is not None:
            raise RuntimeError("DatasetWriter: a run is already open; call close() first")
        self.run_id = run_id or make_run_id(preset, tag)
        self.run_dir = _join(self.logs_root, self.run_id)
        if os.path.exists(self.run_dir):
            raise FileExistsError(f"run directory already exists: {self.run_dir}")
        for sub in (EPISODES_DIR, ROLLOUTS_DIR, PREVIEW_DIR):
            os.makedirs(_join(self.run_dir, sub), exist_ok=True)

        self.config = config
        self._t_open = time.time()
        self._n_episodes = 0
        self._n_index_lines = 0
        self._n_rollouts = 0
        self._n_previews = 0
        self._written_indices = set()

        save_scenario_config(config, _join(self.run_dir, CONFIG_NAME))
        self.manifest = build_manifest(
            config=config,
            run_id=self.run_id,
            preset=preset,
            tag=tag,
            master_seed=master_seed,
            splits=splits,
            extra=extra,
            include_fingerprint=include_fingerprint,
        )
        write_manifest(self.run_dir, self.manifest)
        _atomic_write_text(_join(self.run_dir, README_NAME),
                           build_readme(self.run_id, config))

        # line-buffered append; the handle stays open for the whole run
        self._index_fh = open(_join(self.run_dir, INDEX_NAME), "a",
                              encoding="utf-8", newline="\n")
        return self.run_dir

    def add_split(self, name: str, **fields: Any) -> None:
        """Record per-split provenance (see :func:`storage.manifest.add_split`).

        The manifest on disk is rewritten immediately so the information
        survives a crash.
        """
        self._require_open()
        add_split(self.manifest, name, **fields)
        write_manifest(self.run_dir, self.manifest)

    def close(self, status: str = "complete", extra: Mapping[str, Any] | None = None) -> str:
        """Finalise the run: flush the index and rewrite the manifest.

        Args:
            status: ``'complete'`` (default) or ``'aborted'``.
            extra: extra JSON-safe fields merged into ``manifest['extra']``.

        Returns:
            The manifest path, or ``''`` if no run was open.
        """
        if self._index_fh is None:
            return ""
        try:
            self._index_fh.flush()
            os.fsync(self._index_fh.fileno())
        finally:
            self._index_fh.close()
            self._index_fh = None
        self.manifest["status"] = str(status)
        self.manifest["finished_utc"] = _utc_now_iso()
        self.manifest["wall_clock_s"] = round(time.time() - self._t_open, 3)
        self.manifest["counts"] = {
            "n_episodes": int(self._n_episodes),
            "n_rollout_files": int(self._n_rollouts),
            "n_preview_files": int(self._n_previews),
            "n_index_lines": int(self._n_index_lines),
        }
        if extra:
            self.manifest.setdefault("extra", {}).update(dict(extra))
        path = write_manifest(self.run_dir, self.manifest)
        return path

    def __enter__(self) -> "DatasetWriter":
        """Enter the context manager (the run must still be opened explicitly)."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the run; status is ``'aborted'`` when the body raised."""
        self.close("complete" if exc_type is None else "aborted")
        return False

    # -- writing ----------------------------------------------------------

    def write_episode(
        self,
        ep: EpisodeArrays,
        meta: EpisodeMeta,
        write_preview: bool = True,
        append_index: bool = True,
        validate: bool = True,
        preview_stride: int = 2,
        compressed: bool = True,
    ) -> str:
        """Write one episode, its preview and its index line.

        Ordering matters for crash safety: the NPZ is written and moved into
        place first, the preview second, and only then is the index line
        appended and fsynced. A process killed at any point therefore never
        leaves an index line without its episode file.

        Args:
            ep: the episode arrays.
            meta: the matching metadata; ``meta.episode_index`` names the file.
            write_preview: also write the decimated GUI preview JSON.
            append_index: also append the metadata line to ``index.jsonl``.
            validate: run :func:`storage.schema.validate` first and raise on
                any problem. Turn off only for deliberate negative tests.
            preview_stride: logged-step decimation of the preview [-].
            compressed: deflate the NPZ (True) or store raw (False).

        Returns:
            The episode NPZ path.

        Raises:
            RuntimeError: if no run is open.
            ValueError: if validation fails or the index already holds this
                episode index.
        """
        self._require_open()
        if validate:
            problems = schema.validate(ep)
            problems.extend(f"meta: {c}" for c in meta.validate())
            if problems:
                raise ValueError(
                    "episode %d failed validation:\n  %s"
                    % (meta.episode_index, "\n  ".join(problems))
                )
        idx = int(meta.episode_index)
        if idx in self._written_indices:
            raise ValueError(f"episode index {idx} has already been written to this run")

        path = episode_path(self.run_dir, idx)
        _atomic_savez(path, ep.to_npz_dict(), compressed=compressed)

        if write_preview:
            _atomic_write_text(
                preview_path(self.run_dir, idx),
                json.dumps(build_preview(ep, meta, stride=preview_stride),
                           ensure_ascii=False, allow_nan=False),
            )
            self._n_previews += 1

        self._written_indices.add(idx)
        self._n_episodes += 1
        if append_index:
            self.append_index(meta)
        return path

    def append_index(self, meta: EpisodeMeta) -> None:
        """Append one :class:`EpisodeMeta` line to ``index.jsonl`` and fsync.

        One JSON object per line, no trailing commas: the file stays readable
        even if the process is killed mid-run, and it can be streamed without
        loading the whole table.

        Args:
            meta: the episode metadata to append.

        Raises:
            RuntimeError: if no run is open.
        """
        self._require_open()
        line = json.dumps(meta.to_dict(), ensure_ascii=False,
                          sort_keys=False, allow_nan=False)
        self._index_fh.write(line + "\n")
        self._index_fh.flush()
        os.fsync(self._index_fh.fileno())
        self._written_indices.add(int(meta.episode_index))
        self._n_index_lines += 1

    def write_rollouts(
        self,
        episode_index: int,
        arrays: Mapping[str, np.ndarray],
        validate: bool = True,
        compressed: bool = True,
    ) -> str:
        """Write the branch-rollout bundle of one episode (eval splits only).

        Doc 3.3: from the same initial condition and the same past track, the
        world is re-rolled K times with different decision seeds; that
        empirical distribution is the calibration ground truth. The bundle
        layout is :data:`ROLLOUT_ARRAY_SPEC`, but any extra arrays are stored
        verbatim so :mod:`simulation.rollout` can extend it.

        Args:
            episode_index: index [-] of the episode the rollouts belong to.
            arrays: mapping of array name to numpy array.
            validate: check the recommended keys against the spec and raise.
            compressed: deflate the NPZ.

        Returns:
            The rollout NPZ path.

        Raises:
            RuntimeError: if no run is open.
            ValueError: if validation fails or the mapping is empty.
        """
        self._require_open()
        if not arrays:
            raise ValueError("write_rollouts: empty array mapping")
        if validate:
            problems = validate_rollouts(arrays)
            if problems:
                raise ValueError(
                    "rollouts for episode %d failed validation:\n  %s"
                    % (episode_index, "\n  ".join(problems))
                )
        path = rollout_path(self.run_dir, int(episode_index))
        _atomic_savez(path, {k: np.asarray(v) for k, v in arrays.items()},
                      compressed=compressed)
        self._n_rollouts += 1
        return path

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        """Raise :class:`RuntimeError` unless a run is currently open."""
        if self._index_fh is None or not self.run_dir:
            raise RuntimeError("DatasetWriter: no run is open; call open_run() first")

    @property
    def n_episodes(self) -> int:
        """Number of episodes written to the currently open run [-]."""
        return self._n_episodes


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with second resolution."""
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def read_episode(path: str, validate: bool = False) -> EpisodeArrays:
    """Load one episode NPZ back into :class:`EpisodeArrays`.

    dtypes are restored exactly as declared in
    :data:`storage.schema.EPISODE_ARRAY_SPEC`, including the NaN pattern of
    the observation layer -- nothing is filled in on load.

    Args:
        path: path of the ``ep_NNNNNN.npz`` file.
        validate: also run :func:`storage.schema.validate` and raise on any
            problem.

    Returns:
        The episode arrays.

    Raises:
        KeyError: if the file is missing a schema array.
        ValueError: if ``validate`` is True and the episode is inconsistent.
    """
    with np.load(path, allow_pickle=False) as npz:
        missing = [k for k in schema.EPISODE_ARRAY_SPEC if k not in npz.files]
        if missing:
            raise KeyError(f"{path}: missing array(s) {missing}")
        ep = EpisodeArrays.from_npz_dict({k: npz[k] for k in schema.EPISODE_ARRAY_SPEC})
    if validate:
        problems = schema.validate(ep)
        if problems:
            raise ValueError(f"{path} failed validation:\n  " + "\n  ".join(problems))
    return ep


def read_rollouts(path: str) -> dict[str, np.ndarray]:
    """Load a branch-rollout NPZ as a plain ``{name: array}`` dict.

    Args:
        path: path of the ``ep_NNNNNN_rollouts.npz`` file.

    Returns:
        Every array in the file, dtypes preserved.
    """
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def read_preview(path: str) -> dict[str, Any]:
    """Load a decimated preview JSON (the GUI replay record)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_index(run_dir: str) -> list[EpisodeMeta]:
    """Read ``index.jsonl`` of a run into a list of :class:`EpisodeMeta`.

    Blank lines are skipped and a trailing partial line (possible only if the
    process died mid-write, which fsync makes very unlikely) is ignored rather
    than raising, so a killed run stays loadable.

    Args:
        run_dir: the run directory.

    Returns:
        Metadata in file order (which is generation order).
    """
    path = _join(run_dir, INDEX_NAME)
    out: list[EpisodeMeta] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(EpisodeMeta.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
    return out


def iter_run(
    run_dir: str,
    validate: bool = False,
    with_meta: bool = True,
) -> Iterator[tuple[EpisodeMeta | None, EpisodeArrays]]:
    """Iterate over every episode of a run in index order, lazily.

    Only one episode is held in memory at a time, so a 5,000-episode run can
    be streamed on a laptop.

    Args:
        run_dir: the run directory.
        validate: run the schema validator on each episode and raise on error.
        with_meta: pair each episode with its ``index.jsonl`` metadata; when
            an episode file has no index line the metadata is ``None``.

    Yields:
        ``(meta_or_None, episode_arrays)`` tuples.
    """
    run_dir = _join(run_dir)
    metas = {int(m.episode_index): m for m in read_index(run_dir)} if with_meta else {}
    ep_dir = _join(run_dir, EPISODES_DIR)
    if not os.path.isdir(ep_dir):
        return
    names = sorted(fn for fn in os.listdir(ep_dir) if episode_index_from_name(fn) >= 0)
    for fn in names:
        idx = episode_index_from_name(fn)
        ep = read_episode(_join(ep_dir, fn), validate=validate)
        yield (metas.get(idx), ep)


def list_runs(logs_root: str = LOGS_ROOT) -> list[dict[str, Any]]:
    """List the dataset runs under ``logs_root``, newest first.

    A directory counts as a run when it contains a ``manifest.json``. Runs
    with an unreadable manifest are still listed, with ``status='unreadable'``,
    so a partially written run is visible in the GUI instead of silently
    disappearing.

    Args:
        logs_root: dataset root; defaults to ``$PRISM_ROOT/Logs``.

    Returns:
        A list of ``{'run_id', 'run_dir', 'status', 'created_utc', 'preset',
        'tag', 'n_episodes', 'n_index_lines', 'format_version'}`` dicts sorted
        by ``run_id`` descending (which is chronological, because the id
        starts with a sortable timestamp).
    """
    root = _join(logs_root)
    out: list[dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root), reverse=True):
        run_dir = _join(root, name)
        if not os.path.isdir(run_dir):
            continue
        man_path = _join(run_dir, MANIFEST_NAME)
        if not os.path.isfile(man_path):
            continue
        n_files = 0
        ep_dir = _join(run_dir, EPISODES_DIR)
        if os.path.isdir(ep_dir):
            n_files = sum(1 for fn in os.listdir(ep_dir)
                          if episode_index_from_name(fn) >= 0)
        try:
            man = read_manifest(run_dir)
        except (OSError, json.JSONDecodeError):
            out.append({
                "run_id": name, "run_dir": run_dir, "status": "unreadable",
                "created_utc": "", "preset": "", "tag": "",
                "n_episodes": n_files, "n_index_lines": 0, "format_version": "",
            })
            continue
        out.append({
            "run_id": man.get("run_id", name),
            "run_dir": run_dir,
            "status": man.get("status", "unknown"),
            "created_utc": man.get("created_utc", ""),
            "preset": man.get("preset", ""),
            "tag": man.get("tag", ""),
            "n_episodes": n_files,
            "n_index_lines": int(man.get("counts", {}).get("n_index_lines", 0)),
            "format_version": man.get("format_version", ""),
        })
    return out


def latest_run(logs_root: str = LOGS_ROOT) -> str:
    """Return the newest run directory under ``logs_root``, or ``''`` if none."""
    runs = list_runs(logs_root)
    return runs[0]["run_dir"] if runs else ""
