"""Run layout, manifest construction and whole-run verification.

Reproducibility is an explicit paper requirement (doc 10: "each stage's
artefacts are stored together with seed and configuration so the result stays
reproducible"). This module owns three things:

1. **The on-disk layout of a run** -- directory and file naming. Both
   :mod:`storage.writer` and any external consumer import the naming helpers
   from here so that exactly one module decides where a file lives.
2. **The manifest** -- a JSON document capturing everything needed to explain
   or re-create a run: the full scenario configuration, the master seeds, the
   per-split provenance, code and environment versions, episode counts and
   wall-clock timings.
3. **Verification** -- :func:`verify_run` re-opens every episode file, runs
   :func:`storage.schema.validate` on it, and cross-checks the ``index.jsonl``
   table against the files actually present.

Layout produced (doc 4 requires GT and OBS to be explicitly separated; that
separation lives in the ``gt_*`` / ``obs_*`` key groups inside each NPZ)::

    <LOGS_ROOT>/<run_id>/
        manifest.json                       this module
        config.json                         the exact ScenarioConfig used
        index.jsonl                         one EpisodeMeta per line
        README.txt                          human-readable array dictionary
        episodes/ep_000000.npz              GT + OBS arrays
        rollouts/ep_000123_rollouts.npz     branch rollouts (eval splits only)
        preview/ep_000000.json              decimated JSON for the GUI

``run_id`` is ``YYYYmmdd_HHMMSS_<preset>_<tag>``.

Units: every physical quantity recorded in the manifest is SI (metres,
seconds, radians, m/s); wall-clock fields are seconds and ISO-8601 UTC
timestamps.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from core.constants import (
    HIST_STEPS,
    LOG_DECIMATION,
    LOG_DT,
    LOGS_ROOT,
    PRED_STEPS,
    SIM_DT,
)
from scenario.config import ScenarioConfig, load_scenario_config
from storage import schema
from storage.schema import EpisodeArrays, EpisodeMeta

__all__ = [
    "DATASET_FORMAT_VERSION",
    "MANIFEST_NAME",
    "CONFIG_NAME",
    "INDEX_NAME",
    "README_NAME",
    "EPISODES_DIR",
    "ROLLOUTS_DIR",
    "PREVIEW_DIR",
    "episode_filename",
    "rollout_filename",
    "preview_filename",
    "episode_path",
    "rollout_path",
    "preview_path",
    "episode_index_from_name",
    "make_run_id",
    "slugify",
    "environment_info",
    "source_fingerprint",
    "build_manifest",
    "add_split",
    "write_manifest",
    "read_manifest",
    "validate_manifest",
    "verify_run",
    "summarize_run",
]


# --------------------------------------------------------------------------
# Format version and layout constants
# --------------------------------------------------------------------------

#: Version of the on-disk dataset format. Bump the minor part when a new
#: optional array/key is added, the major part when an existing key changes
#: meaning, dtype or shape (that invalidates every previously written run).
DATASET_FORMAT_VERSION: str = "1.0.0"

MANIFEST_NAME: str = "manifest.json"
CONFIG_NAME: str = "config.json"
INDEX_NAME: str = "index.jsonl"
README_NAME: str = "README.txt"
EPISODES_DIR: str = "episodes"
ROLLOUTS_DIR: str = "rollouts"
PREVIEW_DIR: str = "preview"

#: Stem format for per-episode files: 6 digits supports 10^6 episodes, well
#: above the 5,000 episodes of doc 3.7, and sorts lexicographically.
_EPISODE_STEM_FMT: str = "ep_{:06d}"

_EPISODE_RE = re.compile(r"^ep_(\d{6})\.npz$")
_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


def slugify(text: str, fallback: str = "none") -> str:
    """Return a filesystem-safe token.

    Any run of characters outside ``[0-9A-Za-z._-]`` collapses to a single
    hyphen; leading/trailing separators are stripped.

    Args:
        text: arbitrary label, e.g. a preset name typed in the GUI.
        fallback: value returned when ``text`` reduces to the empty string.

    Returns:
        A token safe to embed in a Windows or POSIX path component.
    """
    s = _SLUG_RE.sub("-", str(text)).strip("-._")
    return s if s else fallback


def episode_filename(episode_index: int) -> str:
    """Return ``ep_000123.npz`` for episode index 123."""
    return _EPISODE_STEM_FMT.format(int(episode_index)) + ".npz"


def rollout_filename(episode_index: int) -> str:
    """Return ``ep_000123_rollouts.npz`` for episode index 123."""
    return _EPISODE_STEM_FMT.format(int(episode_index)) + "_rollouts.npz"


def preview_filename(episode_index: int) -> str:
    """Return ``ep_000123.json`` for episode index 123."""
    return _EPISODE_STEM_FMT.format(int(episode_index)) + ".json"


def episode_path(run_dir: str, episode_index: int) -> str:
    """Absolute path (forward slashes) of an episode NPZ inside ``run_dir``."""
    return _join(run_dir, EPISODES_DIR, episode_filename(episode_index))


def rollout_path(run_dir: str, episode_index: int) -> str:
    """Absolute path of a branch-rollout NPZ inside ``run_dir``."""
    return _join(run_dir, ROLLOUTS_DIR, rollout_filename(episode_index))


def preview_path(run_dir: str, episode_index: int) -> str:
    """Absolute path of a decimated GUI preview JSON inside ``run_dir``."""
    return _join(run_dir, PREVIEW_DIR, preview_filename(episode_index))


def episode_index_from_name(name: str) -> int:
    """Parse ``ep_000123.npz`` back to ``123``; return ``-1`` if it does not match."""
    m = _EPISODE_RE.match(os.path.basename(str(name)))
    return int(m.group(1)) if m else -1


def _join(*parts: str) -> str:
    """Join path parts and normalise to forward slashes (Windows-safe)."""
    return os.path.join(*parts).replace("\\", "/")


def make_run_id(preset: str, tag: str = "", when: _dt.datetime | None = None) -> str:
    """Build the run identifier ``YYYYmmdd_HHMMSS_<preset>_<tag>``.

    Args:
        preset: scenario preset name (slugified).
        tag: free-form run tag such as ``train`` / ``testood`` (slugified).
        when: local timestamp to use; defaults to :func:`datetime.now`.

    Returns:
        A directory-name-safe run identifier.
    """
    ts = (when or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slugify(preset, 'preset')}_{slugify(tag, 'run')}"


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    """Capture interpreter / platform / library versions for the manifest.

    Returns:
        A JSON-safe dict. Only numpy is reported because numpy + stdlib is the
        entire dependency set of dataGen.
    """
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable.replace("\\", "/"),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def _package_root() -> str:
    """Directory holding the dataGen packages (parent of ``storage/``)."""
    return _join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def source_fingerprint(root: str | None = None) -> dict[str, Any]:
    """Hash every ``.py`` file under ``root`` into one reproducibility digest.

    dataGen is not guaranteed to live in a version-control checkout, so the
    manifest records a content hash of the generating code instead of a commit
    id. Two runs with the same fingerprint were produced by byte-identical
    source.

    Args:
        root: directory to walk; defaults to the dataGen package root.

    Returns:
        ``{'root', 'sha256', 'n_files', 'n_bytes'}``. ``sha256`` is ``''`` when
        no source file could be read.
    """
    base = root or _package_root()
    h = hashlib.sha256()
    n_files = 0
    n_bytes = 0
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in {"__pycache__", ".git", ".idea"})
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                paths.append(_join(dirpath, fn))
    for path in sorted(paths):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(path, base).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        n_files += 1
        n_bytes += len(data)
    return {
        "root": base,
        "sha256": h.hexdigest() if n_files else "",
        "n_files": n_files,
        "n_bytes": n_bytes,
    }


# --------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------


def build_manifest(
    config: ScenarioConfig,
    run_id: str,
    preset: str = "",
    tag: str = "",
    master_seed: int | None = None,
    splits: Mapping[str, Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
    source_root: str | None = None,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    """Assemble the manifest dictionary for a freshly opened run.

    Args:
        config: the exact :class:`~scenario.config.ScenarioConfig` used.
        run_id: run identifier, normally from :func:`make_run_id`.
        preset: preset name the config was loaded from (informational).
        tag: free-form run tag (informational).
        master_seed: top-level seed [-]; defaults to ``config.seed``.
        splits: per-split provenance, ``{split_name: {...}}``. Use
            :func:`add_split` to fill this in incrementally.
        extra: arbitrary extra JSON-safe fields merged under ``'extra'``.
        source_root: directory to fingerprint; defaults to the package root.
        include_fingerprint: set False to skip source hashing (fast tests).

    Returns:
        A JSON-safe manifest dict with ``status == 'open'``. Counts and
        wall-clock end fields are filled in by :mod:`storage.writer` at close.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    man: dict[str, Any] = {
        "format_version": DATASET_FORMAT_VERSION,
        "run_id": str(run_id),
        "preset": str(preset),
        "tag": str(tag),
        "status": "open",
        "created_utc": now.isoformat(timespec="seconds"),
        "finished_utc": None,
        "wall_clock_s": 0.0,
        "master_seed": int(config.seed if master_seed is None else master_seed),
        "config": config.to_dict(),
        "timing": {
            "sim_dt_s": float(SIM_DT),
            "log_dt_s": float(LOG_DT),
            "log_decimation": int(LOG_DECIMATION),
            "hist_steps": int(HIST_STEPS),
            "pred_steps": int(PRED_STEPS),
            "duration_s": float(config.duration_s),
            "n_logged_steps": int(config.n_logged_steps()),
        },
        "counts": {
            "n_episodes": 0,
            "n_rollout_files": 0,
            "n_preview_files": 0,
            "n_index_lines": 0,
        },
        "splits": {str(k): dict(v) for k, v in (splits or {}).items()},
        "environment": environment_info(),
        "source": (source_fingerprint(source_root) if include_fingerprint
                   else {"root": source_root or _package_root(),
                         "sha256": "", "n_files": 0, "n_bytes": 0}),
        "layout": {
            "manifest": MANIFEST_NAME,
            "config": CONFIG_NAME,
            "index": INDEX_NAME,
            "readme": README_NAME,
            "episodes_dir": EPISODES_DIR,
            "rollouts_dir": ROLLOUTS_DIR,
            "preview_dir": PREVIEW_DIR,
            "episode_file": episode_filename(0),
            "rollout_file": rollout_filename(0),
            "preview_file": preview_filename(0),
        },
        "array_spec": {
            name: {
                "dtype": np.dtype(dtype).name,
                "shape": [str(s) for s in shape],
                "group": ("gt" if name.startswith("gt_")
                          else "obs" if name.startswith("obs_")
                          else "own" if name.startswith("own_")
                          else "index"),
            }
            for name, (dtype, shape) in schema.EPISODE_ARRAY_SPEC.items()
        },
        "extra": dict(extra or {}),
    }
    return man


def add_split(
    man: dict[str, Any],
    name: str,
    n_episodes: int = 0,
    seed_base: int | None = None,
    episode_range: Sequence[int] | None = None,
    notes: str = "",
    **fields: Any,
) -> dict[str, Any]:
    """Record per-split provenance in an existing manifest (in place).

    Args:
        man: manifest dict from :func:`build_manifest`.
        name: split name, e.g. ``'train'``, ``'valid'``, ``'test_id'``,
            ``'test_ood'`` (doc 8.1).
        n_episodes: number of episodes [-] belonging to this split.
        seed_base: base seed [-] from which the split's episode seeds derive.
        episode_range: ``[first, last_exclusive)`` episode index range.
        notes: free-form description of what makes this split different.
        **fields: any additional JSON-safe provenance fields.

    Returns:
        The same manifest dict, for chaining.
    """
    entry: dict[str, Any] = {
        "n_episodes": int(n_episodes),
        "seed_base": None if seed_base is None else int(seed_base),
        "episode_range": None if episode_range is None else [int(v) for v in episode_range],
        "notes": str(notes),
    }
    entry.update(fields)
    man.setdefault("splits", {})[str(name)] = entry
    return man


def write_manifest(run_dir: str, man: Mapping[str, Any]) -> str:
    """Atomically write ``manifest.json`` into ``run_dir``.

    Args:
        run_dir: run directory.
        man: manifest dict.

    Returns:
        The path written.
    """
    path = _join(run_dir, MANIFEST_NAME)
    _atomic_write_text(path, json.dumps(man, indent=2, ensure_ascii=False,
                                        sort_keys=False, allow_nan=False) + "\n")
    return path


def read_manifest(run_dir: str) -> dict[str, Any]:
    """Read ``manifest.json`` from ``run_dir``.

    Raises:
        FileNotFoundError: if the manifest is missing.
    """
    path = _join(run_dir, MANIFEST_NAME)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write_text(path: str, text: str) -> None:
    """Write text via ``<path>.tmp`` + :func:`os.replace` so readers never
    observe a half-written file (crash safety, doc 10)."""
    tmp = path + ".tmp"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

_REQUIRED_MANIFEST_KEYS = (
    "format_version", "run_id", "status", "created_utc", "master_seed",
    "config", "timing", "counts", "environment", "source", "layout",
    "array_spec",
)


def validate_manifest(man: Mapping[str, Any]) -> list[str]:
    """Check a manifest dict for structural completeness.

    Args:
        man: the manifest to check.

    Returns:
        A list of human-readable problems; empty means the manifest is usable.
    """
    p: list[str] = []
    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in man:
            p.append(f"manifest: missing key '{key}'")
    if p:
        return p

    fv = str(man["format_version"])
    if fv.split(".")[0] != DATASET_FORMAT_VERSION.split(".")[0]:
        p.append(
            f"manifest: format_version {fv} is incompatible with reader "
            f"{DATASET_FORMAT_VERSION} (major version differs)"
        )
    if str(man["status"]) not in {"open", "complete", "aborted"}:
        p.append(f"manifest: unknown status '{man['status']}'")

    try:
        cfg = ScenarioConfig.from_dict(man["config"])
    except Exception as exc:                      # noqa: BLE001 - report, not raise
        p.append(f"manifest: config does not deserialise ({exc!r})")
    else:
        p.extend(f"manifest.config: {c}" for c in cfg.validate())

    timing = man.get("timing", {})
    if abs(float(timing.get("sim_dt_s", SIM_DT)) - SIM_DT) > 1e-12:
        p.append("manifest.timing: sim_dt_s differs from core.constants.SIM_DT "
                 "-- the dataset was generated by an incompatible build")
    if abs(float(timing.get("log_dt_s", LOG_DT)) - LOG_DT) > 1e-12:
        p.append("manifest.timing: log_dt_s differs from core.constants.LOG_DT "
                 "-- the dataset was generated by an incompatible build")

    spec = man.get("array_spec", {})
    for name, (dtype, shape) in schema.EPISODE_ARRAY_SPEC.items():
        if name not in spec:
            p.append(f"manifest.array_spec: missing array '{name}'")
            continue
        got = spec[name]
        if str(got.get("dtype")) != np.dtype(dtype).name:
            p.append(f"manifest.array_spec['{name}']: dtype "
                     f"{got.get('dtype')} != {np.dtype(dtype).name}")
        if [str(s) for s in shape] != [str(s) for s in got.get("shape", [])]:
            p.append(f"manifest.array_spec['{name}']: shape "
                     f"{got.get('shape')} != {[str(s) for s in shape]}")

    counts = man.get("counts", {})
    for key in ("n_episodes", "n_index_lines"):
        if int(counts.get(key, 0)) < 0:
            p.append(f"manifest.counts['{key}'] must be >= 0")
    return p


# --------------------------------------------------------------------------
# Whole-run verification
# --------------------------------------------------------------------------


def _read_index_lines(run_dir: str) -> tuple[list[EpisodeMeta], list[str]]:
    """Parse ``index.jsonl``; returns (metas, problems)."""
    path = _join(run_dir, INDEX_NAME)
    metas: list[EpisodeMeta] = []
    problems: list[str] = []
    if not os.path.isfile(path):
        problems.append(f"{INDEX_NAME}: missing")
        return metas, problems
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{INDEX_NAME}:{lineno}: not valid JSON ({exc.msg})")
                continue
            try:
                metas.append(EpisodeMeta.from_dict(d))
            except Exception as exc:              # noqa: BLE001
                problems.append(f"{INDEX_NAME}:{lineno}: not an EpisodeMeta ({exc!r})")
    return metas, problems


def _list_episode_files(run_dir: str) -> list[tuple[int, str]]:
    """Return ``[(episode_index, path)]`` sorted by index for ``episodes/``."""
    d = _join(run_dir, EPISODES_DIR)
    out: list[tuple[int, str]] = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        idx = episode_index_from_name(fn)
        if idx >= 0:
            out.append((idx, _join(d, fn)))
    out.sort(key=lambda kv: kv[0])
    return out


def verify_run(
    run_dir: str,
    deep: bool = True,
    max_episodes: int | None = None,
) -> list[str]:
    """Re-open a finished run and check that it is internally consistent.

    Checks performed:
        * ``manifest.json`` exists, deserialises and passes
          :func:`validate_manifest`.
        * ``config.json`` exists, deserialises and matches the config embedded
          in the manifest.
        * ``index.jsonl`` exists, every line is a valid
          :class:`~storage.schema.EpisodeMeta`, and each passes
          ``EpisodeMeta.validate``.
        * The number of index lines equals the number of episode NPZ files,
          episode indices are unique, and every index entry has a file.
        * (``deep``) every episode NPZ loads, contains exactly the schema key
          groups (``gt_*`` / ``obs_*`` / ownship / index arrays) and passes
          :func:`storage.schema.validate`.
        * Episode shapes agree with the manifest timing block.

    Args:
        run_dir: the ``<LOGS_ROOT>/<run_id>`` directory.
        deep: when False, skip loading the episode arrays (structure only).
        max_episodes: if given, only deep-check the first N episodes.

    Returns:
        A list of human-readable problems. Empty means the run is verified.
    """
    problems: list[str] = []
    run_dir = _join(run_dir)
    if not os.path.isdir(run_dir):
        return [f"run_dir: '{run_dir}' is not a directory"]

    # -- manifest ---------------------------------------------------------
    man: dict[str, Any] | None = None
    try:
        man = read_manifest(run_dir)
    except FileNotFoundError:
        problems.append(f"{MANIFEST_NAME}: missing")
    except json.JSONDecodeError as exc:
        problems.append(f"{MANIFEST_NAME}: not valid JSON ({exc.msg})")
    if man is not None:
        problems.extend(validate_manifest(man))

    # -- config -----------------------------------------------------------
    cfg_path = _join(run_dir, CONFIG_NAME)
    cfg: ScenarioConfig | None = None
    if not os.path.isfile(cfg_path):
        problems.append(f"{CONFIG_NAME}: missing")
    else:
        try:
            cfg = load_scenario_config(cfg_path)
        except Exception as exc:                  # noqa: BLE001
            problems.append(f"{CONFIG_NAME}: does not load ({exc!r})")
        else:
            problems.extend(f"{CONFIG_NAME}: {c}" for c in cfg.validate())
            if man is not None and cfg.to_dict() != man.get("config"):
                problems.append(
                    f"{CONFIG_NAME} differs from the config embedded in "
                    f"{MANIFEST_NAME} -- provenance is broken"
                )

    # -- index ------------------------------------------------------------
    metas, index_problems = _read_index_lines(run_dir)
    problems.extend(index_problems)
    for m in metas:
        problems.extend(f"{INDEX_NAME}[ep {m.episode_index}]: {c}"
                        for c in m.validate())

    files = _list_episode_files(run_dir)
    file_indices = [i for i, _ in files]
    if len(set(file_indices)) != len(file_indices):
        problems.append(f"{EPISODES_DIR}/: duplicate episode index among files")
    meta_indices = [int(m.episode_index) for m in metas]
    if len(set(meta_indices)) != len(meta_indices):
        problems.append(f"{INDEX_NAME}: duplicate episode_index among lines")

    if len(metas) != len(files):
        problems.append(
            f"{INDEX_NAME} has {len(metas)} line(s) but {EPISODES_DIR}/ holds "
            f"{len(files)} episode file(s)"
        )
    missing = sorted(set(meta_indices) - set(file_indices))
    if missing:
        problems.append(f"{INDEX_NAME}: no episode file for index(es) {missing}")
    orphan = sorted(set(file_indices) - set(meta_indices))
    if orphan:
        problems.append(f"{EPISODES_DIR}/: file(s) {orphan} absent from {INDEX_NAME}")

    if man is not None:
        n_declared = int(man.get("counts", {}).get("n_episodes", -1))
        if n_declared >= 0 and n_declared != len(files):
            problems.append(
                f"{MANIFEST_NAME}: counts.n_episodes={n_declared} but "
                f"{len(files)} episode file(s) exist"
            )

    if not deep:
        return problems

    # -- episodes ---------------------------------------------------------
    expected_T = None
    if man is not None:
        try:
            expected_T = int(man["timing"]["n_logged_steps"])
        except (KeyError, TypeError, ValueError):
            expected_T = None
    meta_by_index = {int(m.episode_index): m for m in metas}
    checked = files if max_episodes is None else files[: int(max_episodes)]
    for idx, path in checked:
        try:
            with np.load(path, allow_pickle=False) as npz:
                keys = set(npz.files)
                missing_keys = set(schema.EPISODE_ARRAY_SPEC) - keys
                if missing_keys:
                    problems.append(
                        f"{episode_filename(idx)}: missing array(s) "
                        f"{sorted(missing_keys)}"
                    )
                    continue
                ep = EpisodeArrays.from_npz_dict(
                    {k: npz[k] for k in schema.EPISODE_ARRAY_SPEC}
                )
        except Exception as exc:                  # noqa: BLE001
            problems.append(f"{episode_filename(idx)}: does not load ({exc!r})")
            continue
        problems.extend(f"{episode_filename(idx)}: {c}" for c in schema.validate(ep))
        if expected_T is not None and ep.n_steps != expected_T:
            problems.append(
                f"{episode_filename(idx)}: T={ep.n_steps} but the manifest "
                f"declares n_logged_steps={expected_T}"
            )
        m = meta_by_index.get(idx)
        if m is not None:
            n_used = int(np.count_nonzero(ep.used_slots()))
            if int(m.n_agents) != n_used:
                problems.append(
                    f"{episode_filename(idx)}: meta.n_agents={m.n_agents} but "
                    f"{n_used} slot(s) are ever valid"
                )
    return problems


def summarize_run(run_dir: str) -> dict[str, Any]:
    """Summarise a run from its index table without loading any array.

    Args:
        run_dir: the run directory.

    Returns:
        ``{'run_id', 'run_dir', 'status', 'n_episodes', 'n_rollout_files',
        'n_preview_files', 'duration_s', 'nmac_rate', 'dwc_rate',
        'risk_level_counts', 'platform_counts', 'intent_counts',
        'min_separation_m': {'min','median','max'}}``. Missing pieces degrade
        gracefully to defaults rather than raising.
    """
    run_dir = _join(run_dir)
    try:
        man = read_manifest(run_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        man = {}
    metas, _ = _read_index_lines(run_dir)
    n = len(metas)

    def _count_dir(sub: str) -> int:
        d = _join(run_dir, sub)
        return len(os.listdir(d)) if os.path.isdir(d) else 0

    risk_counts: dict[int, int] = {}
    plat_counts: dict[int, int] = {}
    intent_counts: dict[int, int] = {}
    seps: list[float] = []
    for m in metas:
        risk_counts[int(m.risk_level)] = risk_counts.get(int(m.risk_level), 0) + 1
        for k, v in m.platform_counts.items():
            plat_counts[int(k)] = plat_counts.get(int(k), 0) + int(v)
        for k, v in m.intent_counts.items():
            intent_counts[int(k)] = intent_counts.get(int(k), 0) + int(v)
        if np.isfinite(m.min_separation_m):
            seps.append(float(m.min_separation_m))
    sep_arr = np.asarray(seps, dtype=np.float64)
    return {
        "run_id": man.get("run_id", os.path.basename(run_dir)),
        "run_dir": run_dir,
        "status": man.get("status", "unknown"),
        "format_version": man.get("format_version", ""),
        "created_utc": man.get("created_utc", ""),
        "n_episodes": n,
        "n_rollout_files": _count_dir(ROLLOUTS_DIR),
        "n_preview_files": _count_dir(PREVIEW_DIR),
        "duration_s": float(metas[0].duration_s) if n else 0.0,
        "nmac_rate": (float(sum(1 for m in metas if m.nmac_occurred)) / n) if n else 0.0,
        "dwc_rate": (float(sum(1 for m in metas if m.dwc_violated)) / n) if n else 0.0,
        "risk_level_counts": dict(sorted(risk_counts.items())),
        "platform_counts": dict(sorted(plat_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "min_separation_m": {
            "min": float(np.min(sep_arr)) if sep_arr.size else None,
            "median": float(np.median(sep_arr)) if sep_arr.size else None,
            "max": float(np.max(sep_arr)) if sep_arr.size else None,
        },
    }
