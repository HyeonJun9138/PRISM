"""Experiment splits of doc 8.1, cut along PARAMETER axes -- never by seed.

Doc 8.1 is explicit that splitting the same scenario family by seed alone is a
LEAKAGE BUG: the test set would then be drawn from exactly the distribution the
model was trained on, and "generalisation" would measure nothing. The splits
here are therefore cut along four parameter axes

    * traffic density        [aircraft / km^2]
    * wind speed band        [m/s]
    * the SUPPORT of the intent mix
    * the SUPPORT of the platform mix

and the seed blocks are disjoint on top of that, so a replay key can never be
shared between two splits either. Both properties are machine-checkable:

    :func:`assert_no_seed_overlap`          -- seed blocks are pairwise disjoint
    :func:`assert_parameter_axis_separation` -- test_ood really is off-manifold

``valid`` and ``test_id`` are *deliberately* identical in distribution to
``train`` (doc 8.1: "동일 분포, 다른 시드"). They are in-distribution controls;
the generalisation claim rests on ``test_ood`` alone. This module makes that
asymmetry explicit rather than leaving it to a comment.

Area / density reconciliation (a modelling decision recorded here on purpose)
---------------------------------------------------------------------------
Doc 3.7 asks for 5-15 traffic agents and doc 8.1 asks for 20-60 aircraft/km^2.
Those two statements are simultaneously true at exactly one area:

    5 agents  / 20 per km^2 = 0.25 km^2
    15 agents / 60 per km^2 = 0.25 km^2

So ``ScenarioConfig.area_km2`` is used here as the **density reference
footprint** (0.25 km^2 = a 500 m box travelling with the ownship), which is
what makes ``ScenarioConfig.n_background()`` reproduce doc 3.7's agent count
across doc 8.1's density band without the ``n_agents_min/max`` clamp doing all
the work. The *simulated arena* is a separate, much larger region owned by
:mod:`scenario.builder` (``ARENA_AREA_KM2``), because a 55 m/s ownship cannot
fly a 60 s cruise leg inside a 500 m box.

``density_per_km2`` is therefore the SPLIT AXIS -- the knob that sets how many
aircraft an episode contains -- and not the areal density realised inside the
arena, which is ``n_agents / ARENA_AREA_KM2`` and is reported separately by
``Scenario.realised_density_per_km2``. Taking 20-60 /km^2 literally would put
an aircraft every ~150 m, at which point an NMAC is unavoidable in every
episode and doc 3.4's stratification has nothing left to stratify.

Units are SI throughout: metres, seconds, radians, m/s.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from core.rng import RNGHub
from core.types import IntentType, PlatformType
from scenario.config import (
    EncounterConfig,
    IntentMixConfig,
    PlatformMixConfig,
    ScenarioConfig,
    WindConfig,
    load_scenario_config,
    save_scenario_config,
)

__all__ = [
    "PRESET_NAMES",
    "SEED_BLOCK_SIZE",
    "DENSITY_REFERENCE_AREA_KM2",
    "SplitSpec",
    "SPLIT_SPECS",
    "PRESETS_DIR",
    "split_spec",
    "seed_block",
    "episode_seed",
    "split_of_seed",
    "make_preset",
    "config_for_episode",
    "assert_no_seed_overlap",
    "assert_parameter_axis_separation",
    "parameter_axis_summary",
    "preset_path",
    "write_presets",
    "load_preset",
]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Canonical split names, in pipeline order.
PRESET_NAMES: tuple[str, ...] = ("train", "valid", "test_id", "test_ood")

#: Width [-] of each seed block. Blocks are ``[k*SIZE, (k+1)*SIZE)``.
SEED_BLOCK_SIZE: int = 100_000

#: Footprint [km^2] over which doc 8.1's traffic density is defined.
#: rev.3 densification: 0.5 km^2 maps the 20-60 /km^2 band to 10-30 traffic
#: agents (clamped [12, 26]), doubling scene density over the original
#: 0.25 km^2 / 5-15 reconciliation. See the module docstring.
DENSITY_REFERENCE_AREA_KM2: float = 0.5

#: Directory the preset JSON files are written to / read from.
PRESETS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                 "config", "presets")
).replace("\\", "/")


# --------------------------------------------------------------------------
# Split specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """The parameter-axis definition of one dataset split (doc 8.1).

    A ``SplitSpec`` is the *distribution*; a :class:`ScenarioConfig` is one
    draw from it. Keeping them apart is what lets
    :func:`assert_parameter_axis_separation` reason about the splits without
    generating a single episode.

    Attributes:
        name: split name, one of :data:`PRESET_NAMES`.
        seed_start: first master seed [-] of the block (inclusive).
        seed_end: one past the last master seed [-] of the block (exclusive).
        density_min: lower edge of the traffic density band [aircraft/km^2].
        density_max: upper edge of the traffic density band [aircraft/km^2].
        wind_min_ms: lower edge of the steady wind speed band [m/s].
        wind_max_ms: upper edge of the steady wind speed band [m/s].
        platform_weights: platform mix weights [-] keyed by
            :class:`~core.types.PlatformType` integer.
        intent_weights: intent mix weights [-] keyed by
            :class:`~core.types.IntentType` integer.
        area_km2: density reference footprint [km^2]; see the module
            docstring. Identical across splits so that density, and not area,
            is the axis that moves.
        n_agents_min: hard floor on the traffic agent count [-].
        n_agents_max: hard ceiling on the traffic agent count [-].
        ownship_speed_ms: nominal ownship cruise airspeed [m/s].
        description: one-line human summary for the preset file and the GUI.
    """

    name: str
    seed_start: int
    seed_end: int
    density_min: float
    density_max: float
    wind_min_ms: float
    wind_max_ms: float
    platform_weights: Mapping[int, float]
    intent_weights: Mapping[int, float]
    area_km2: float = DENSITY_REFERENCE_AREA_KM2
    n_agents_min: int = 12
    n_agents_max: int = 26
    ownship_speed_ms: float = 55.0
    description: str = ""
    #: designed near-encounters per episode, upper bound [-] (rev.3: denser).
    n_designed_max: int = 5

    # -- derived views ----------------------------------------------------

    def platform_support(self) -> frozenset[int]:
        """Platform-type integers with strictly positive weight."""
        return frozenset(
            int(k) for k, v in self.platform_weights.items() if float(v) > 0.0
        )

    def intent_support(self) -> frozenset[int]:
        """Intent-type integers with strictly positive weight."""
        return frozenset(
            int(k) for k, v in self.intent_weights.items() if float(v) > 0.0
        )

    def density_band(self) -> tuple[float, float]:
        """``(min, max)`` traffic density [aircraft/km^2]."""
        return float(self.density_min), float(self.density_max)

    def wind_band(self) -> tuple[float, float]:
        """``(min, max)`` steady wind speed [m/s]."""
        return float(self.wind_min_ms), float(self.wind_max_ms)

    def agent_count_band(self) -> tuple[int, int]:
        """``(min, max)`` traffic agent count [-] implied by the density band.

        Applies exactly the arithmetic of
        :meth:`scenario.config.ScenarioConfig.n_background`, so this is what
        the simulator will really produce.
        """
        lo = int(round(self.density_min * self.area_km2))
        hi = int(round(self.density_max * self.area_km2))
        lo = max(self.n_agents_min, min(self.n_agents_max, lo))
        hi = max(self.n_agents_min, min(self.n_agents_max, hi))
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (weight keys become strings)."""
        return {
            "name": self.name,
            "seed_start": int(self.seed_start),
            "seed_end": int(self.seed_end),
            "density_min": float(self.density_min),
            "density_max": float(self.density_max),
            "wind_min_ms": float(self.wind_min_ms),
            "wind_max_ms": float(self.wind_max_ms),
            "platform_weights": {str(int(k)): float(v)
                                 for k, v in sorted(self.platform_weights.items())},
            "intent_weights": {str(int(k)): float(v)
                               for k, v in sorted(self.intent_weights.items())},
            "area_km2": float(self.area_km2),
            "n_agents_min": int(self.n_agents_min),
            "n_agents_max": int(self.n_agents_max),
            "ownship_speed_ms": float(self.ownship_speed_ms),
            "description": self.description,
        }


# -- the four splits -------------------------------------------------------

#: Training platform mix (doc 8.1): sUAS rotary / sUAS fixed-wing / UAM only.
_TRAIN_PLATFORMS: dict[int, float] = {
    int(PlatformType.BIRD): 0.0,
    int(PlatformType.MULTIROTOR): 0.40,
    int(PlatformType.FIXED_WING_UAS): 0.35,
    int(PlatformType.EVTOL_UAM): 0.25,
    int(PlatformType.GA): 0.0,
    int(PlatformType.DEBRIS): 0.0,
    int(PlatformType.UNKNOWN): 0.0,
}

#: Training intent mix (doc 8.1): corridor transit / delivery WP / loiter.
_TRAIN_INTENTS: dict[int, float] = {
    int(IntentType.CORRIDOR_TRANSIT): 0.40,
    int(IntentType.DELIVERY_WP): 0.35,
    int(IntentType.LOITER_HOVER): 0.25,
    int(IntentType.SEARCH_WANDER): 0.0,
    int(IntentType.REACTIVE_AVOID): 0.0,
    int(IntentType.PASSIVE_DRIFT): 0.0,
}

#: OOD platform mix: the trained three stay in (so the split is a SUPERSET
#: shift, not a disjoint world) and bird / GA / debris / unknown are added.
_OOD_PLATFORMS: dict[int, float] = {
    int(PlatformType.BIRD): 0.20,
    int(PlatformType.MULTIROTOR): 0.15,
    int(PlatformType.FIXED_WING_UAS): 0.15,
    int(PlatformType.EVTOL_UAM): 0.10,
    int(PlatformType.GA): 0.20,
    int(PlatformType.DEBRIS): 0.10,
    int(PlatformType.UNKNOWN): 0.10,
}

#: OOD intent mix: search-wander and reactive-avoid added (doc 8.1).
#: PASSIVE_DRIFT carries a small weight for completeness, but every DEBRIS
#: agent is forced onto it by :mod:`scenario.builder` regardless of the mix.
_OOD_INTENTS: dict[int, float] = {
    int(IntentType.CORRIDOR_TRANSIT): 0.15,
    int(IntentType.DELIVERY_WP): 0.15,
    int(IntentType.LOITER_HOVER): 0.10,
    int(IntentType.SEARCH_WANDER): 0.30,
    int(IntentType.REACTIVE_AVOID): 0.25,
    int(IntentType.PASSIVE_DRIFT): 0.05,
}


SPLIT_SPECS: dict[str, SplitSpec] = {
    "train": SplitSpec(
        name="train",
        seed_start=0,
        seed_end=SEED_BLOCK_SIZE,
        density_min=20.0,
        density_max=60.0,
        wind_min_ms=0.0,
        wind_max_ms=8.0,
        platform_weights=_TRAIN_PLATFORMS,
        intent_weights=_TRAIN_INTENTS,
        description=(
            "Training split (doc 8.1): density 20-60 /km^2, wind 0-8 m/s, "
            "intents {corridor transit, delivery WP, loiter/hover}, platforms "
            "{multirotor, fixed-wing UAS, eVTOL UAM}."
        ),
    ),
    "valid": SplitSpec(
        name="valid",
        seed_start=SEED_BLOCK_SIZE,
        seed_end=2 * SEED_BLOCK_SIZE,
        density_min=20.0,
        density_max=60.0,
        wind_min_ms=0.0,
        wind_max_ms=8.0,
        platform_weights=_TRAIN_PLATFORMS,
        intent_weights=_TRAIN_INTENTS,
        description=(
            "Validation split: identical distribution to train, disjoint seed "
            "block. In-distribution control, NOT a generalisation test."
        ),
    ),
    "test_id": SplitSpec(
        name="test_id",
        seed_start=2 * SEED_BLOCK_SIZE,
        seed_end=3 * SEED_BLOCK_SIZE,
        density_min=20.0,
        density_max=60.0,
        wind_min_ms=0.0,
        wind_max_ms=8.0,
        platform_weights=_TRAIN_PLATFORMS,
        intent_weights=_TRAIN_INTENTS,
        description=(
            "In-distribution test split: identical distribution to train, "
            "disjoint seed block. Reports the ceiling that test_ood is "
            "compared against."
        ),
    ),
    "test_ood": SplitSpec(
        name="test_ood",
        seed_start=3 * SEED_BLOCK_SIZE,
        seed_end=4 * SEED_BLOCK_SIZE,
        density_min=100.0,
        density_max=100.0,
        wind_min_ms=10.0,
        wind_max_ms=16.0,
        platform_weights=_OOD_PLATFORMS,
        intent_weights=_OOD_INTENTS,
        n_agents_min=24,
        n_agents_max=36,
        n_designed_max=6,
        description=(
            "Out-of-distribution test split (doc 8.1): density 100 /km^2, "
            "wind 10-16 m/s (band never seen in training), intents extended "
            "with {search wander, reactive avoid}, platforms extended with "
            "{bird, GA, debris, unknown}."
        ),
    ),
}


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------


def split_spec(name: str) -> SplitSpec:
    """Return the :class:`SplitSpec` named ``name``.

    Args:
        name: one of :data:`PRESET_NAMES`.

    Returns:
        The split specification.

    Raises:
        KeyError: if ``name`` is not a known split.
    """
    try:
        return SPLIT_SPECS[str(name)]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; expected one of {list(PRESET_NAMES)}"
        ) from None


def seed_block(name: str) -> tuple[int, int]:
    """Return the half-open master-seed block ``[start, end)`` of a split.

    Args:
        name: split name.

    Returns:
        ``(seed_start, seed_end)`` [-].
    """
    spec = split_spec(name)
    return int(spec.seed_start), int(spec.seed_end)


def episode_seed(name: str, index: int) -> int:
    """Master seed [-] of episode ``index`` within split ``name``.

    Args:
        name: split name.
        index: episode index [-] within the split, ``0 <= index < block size``.

    Returns:
        ``seed_start + index``.

    Raises:
        ValueError: if ``index`` is negative or would leave the seed block,
            which would silently collide with a neighbouring split.
    """
    spec = split_spec(name)
    i = int(index)
    if i < 0:
        raise ValueError(f"episode index must be >= 0, got {index!r}")
    size = spec.seed_end - spec.seed_start
    if i >= size:
        raise ValueError(
            f"episode index {i} overflows the {name!r} seed block of size "
            f"{size}; enlarge SEED_BLOCK_SIZE rather than reusing seeds"
        )
    return int(spec.seed_start) + i


def split_of_seed(seed: int) -> str | None:
    """Return the split a master seed belongs to, or ``None`` if unclaimed.

    Args:
        seed: master seed [-].

    Returns:
        The split name, or ``None`` when the seed falls outside every block.
    """
    s = int(seed)
    for name, spec in SPLIT_SPECS.items():
        if spec.seed_start <= s < spec.seed_end:
            return name
    return None


# --------------------------------------------------------------------------
# Config construction
# --------------------------------------------------------------------------


def make_preset(name: str, seed: int | None = None) -> ScenarioConfig:
    """Build the canonical :class:`ScenarioConfig` of a split.

    The density is set to the MIDPOINT of the split's band, so the stored
    preset file is a single representative configuration.
    :func:`config_for_episode` is what resamples the density per episode and is
    what dataset generation should use.

    Args:
        name: split name.
        seed: master seed [-]; defaults to the first seed of the split's block.

    Returns:
        A validated configuration.

    Raises:
        ValueError: if the assembled configuration fails
            :meth:`scenario.config.ScenarioConfig.validate`.
    """
    spec = split_spec(name)
    s = spec.seed_start if seed is None else int(seed)
    density = 0.5 * (spec.density_min + spec.density_max)
    cfg = ScenarioConfig(
        name=spec.name,
        seed=int(s),
        duration_s=60.0,
        area_km2=float(spec.area_km2),
        density_per_km2=float(density),
        n_agents_min=int(spec.n_agents_min),
        n_agents_max=int(spec.n_agents_max),
        ownship_speed_ms=float(spec.ownship_speed_ms),
        platform_mix=PlatformMixConfig(weights=dict(spec.platform_weights)),
        intent_mix=IntentMixConfig(weights=dict(spec.intent_weights)),
        encounter=EncounterConfig(n_designed_max=int(spec.n_designed_max)),
        wind=WindConfig(
            speed_min_ms=float(spec.wind_min_ms),
            speed_max_ms=float(spec.wind_max_ms),
            dir_min_rad=-math.pi,
            dir_max_rad=math.pi,
        ),
        corridor_enabled=True,
        nfz_count=2,
    )
    problems = cfg.validate()
    if problems:
        raise ValueError(
            f"preset {name!r} is not a valid ScenarioConfig: " + "; ".join(problems)
        )
    return cfg


def config_for_episode(name: str, index: int) -> ScenarioConfig:
    """Build the configuration of one specific episode of a split.

    The master seed comes from the split's block, and the traffic density is
    drawn uniformly from the split's band using a hub seeded by that same
    master seed. The draw therefore depends on nothing but ``(name, index)``,
    and it is taken on a THROWAWAY :class:`~core.rng.RNGHub` so that the
    simulator's own ``'scenario'`` stream starts unconsumed.

    Args:
        name: split name.
        index: episode index [-] within the split.

    Returns:
        A validated configuration whose ``name`` records the split and index.
    """
    spec = split_spec(name)
    seed = episode_seed(name, index)
    lo, hi = spec.density_band()
    if hi > lo:
        density = float(RNGHub(seed).stream("scenario").uniform(lo, hi))
    else:
        density = float(lo)
    cfg = make_preset(name, seed=seed)
    return cfg.copy_with(
        name=f"{spec.name}/{int(index):06d}",
        seed=int(seed),
        density_per_km2=density,
    )


# --------------------------------------------------------------------------
# Structural guarantees (doc 8.1 anti-leakage)
# --------------------------------------------------------------------------


def assert_no_seed_overlap() -> None:
    """Prove the four seed blocks are pairwise disjoint and non-empty.

    Doc 8.1 forbids seed-only splitting; this check does not by itself make
    the splits valid, it only removes the *second* failure mode -- a replay
    key shared between two splits, which would put literally the same episode
    in train and test.

    Raises:
        AssertionError: if any block is empty or two blocks intersect.
    """
    blocks = [(name, spec.seed_start, spec.seed_end)
              for name, spec in SPLIT_SPECS.items()]
    for name, start, end in blocks:
        if end <= start:
            raise AssertionError(
                f"seed block for {name!r} is empty: [{start}, {end})"
            )
    ordered = sorted(blocks, key=lambda b: b[1])
    for (n_a, _s_a, e_a), (n_b, s_b, _e_b) in zip(ordered, ordered[1:]):
        if s_b < e_a:
            raise AssertionError(
                f"seed blocks overlap: {n_a} ends at {e_a} but {n_b} starts "
                f"at {s_b}"
            )


def assert_parameter_axis_separation() -> None:
    """Prove ``test_ood`` is separated from ``train`` on parameter axes.

    Checks, in order:

    1. the density bands of train and test_ood do not intersect;
    2. the wind bands of train and test_ood do not intersect;
    3. test_ood's intent support strictly contains train's, i.e. it adds at
       least one intent that training never saw;
    4. test_ood's platform support strictly contains train's, likewise;
    5. valid and test_id are in-distribution controls -- their bands and
       supports match train exactly, so any difference they show against
       train is attributable to sampling noise alone.

    Raises:
        AssertionError: if any of the five properties fails.
    """
    train = split_spec("train")
    ood = split_spec("test_ood")

    t_lo, t_hi = train.density_band()
    o_lo, o_hi = ood.density_band()
    if not (o_lo > t_hi or t_lo > o_hi):
        raise AssertionError(
            f"density bands overlap: train [{t_lo}, {t_hi}] vs test_ood "
            f"[{o_lo}, {o_hi}] -- doc 8.1 requires an unseen density"
        )

    t_lo, t_hi = train.wind_band()
    o_lo, o_hi = ood.wind_band()
    if not (o_lo > t_hi or t_lo > o_hi):
        raise AssertionError(
            f"wind bands overlap: train [{t_lo}, {t_hi}] m/s vs test_ood "
            f"[{o_lo}, {o_hi}] m/s -- doc 8.1 requires an unseen wind band"
        )

    if not (train.intent_support() < ood.intent_support()):
        raise AssertionError(
            "test_ood intent support must strictly contain train's; got "
            f"train={sorted(train.intent_support())} "
            f"ood={sorted(ood.intent_support())}"
        )
    if not (train.platform_support() < ood.platform_support()):
        raise AssertionError(
            "test_ood platform support must strictly contain train's; got "
            f"train={sorted(train.platform_support())} "
            f"ood={sorted(ood.platform_support())}"
        )

    for control in ("valid", "test_id"):
        spec = split_spec(control)
        if spec.density_band() != train.density_band():
            raise AssertionError(
                f"{control!r} must share train's density band (it is an "
                "in-distribution control, doc 8.1)"
            )
        if spec.wind_band() != train.wind_band():
            raise AssertionError(
                f"{control!r} must share train's wind band"
            )
        if spec.intent_support() != train.intent_support():
            raise AssertionError(
                f"{control!r} must share train's intent support"
            )
        if spec.platform_support() != train.platform_support():
            raise AssertionError(
                f"{control!r} must share train's platform support"
            )


def parameter_axis_summary() -> dict[str, dict[str, Any]]:
    """Machine-readable table of the axes each split is cut on.

    Intended for the paper's experiment-design table and for the GUI.

    Returns:
        ``{split_name: {...}}`` with the seed block, density band [/km^2],
        wind band [m/s], implied agent-count band [-], and the sorted intent /
        platform supports, plus ``added_intents`` / ``added_platforms``
        relative to the training split.
    """
    train = split_spec("train")
    out: dict[str, dict[str, Any]] = {}
    for name in PRESET_NAMES:
        spec = split_spec(name)
        out[name] = {
            "seed_block": [int(spec.seed_start), int(spec.seed_end)],
            "density_per_km2": list(spec.density_band()),
            "wind_speed_ms": list(spec.wind_band()),
            "n_agents": list(spec.agent_count_band()),
            "intents": sorted(int(i) for i in spec.intent_support()),
            "platforms": sorted(int(p) for p in spec.platform_support()),
            "added_intents": sorted(
                int(i) for i in (spec.intent_support() - train.intent_support())
            ),
            "added_platforms": sorted(
                int(p) for p in (spec.platform_support() - train.platform_support())
            ),
            "description": spec.description,
        }
    return out


# --------------------------------------------------------------------------
# On-disk presets
# --------------------------------------------------------------------------


def preset_path(name: str, directory: str | None = None) -> str:
    """Path of the JSON file backing a preset.

    Args:
        name: split name.
        directory: override directory; defaults to :data:`PRESETS_DIR`.

    Returns:
        A forward-slash path, e.g. ``$PRISM_ROOT/dataGen/config/presets/train.json``.
    """
    d = PRESETS_DIR if directory is None else str(directory).replace("\\", "/")
    return f"{d}/{split_spec(name).name}.json"


def write_presets(directory: str | None = None) -> list[str]:
    """Write every preset to ``<directory>/<name>.json``.

    The structural guarantees are asserted BEFORE anything is written, so a
    broken split definition can never reach disk.

    Args:
        directory: destination; defaults to :data:`PRESETS_DIR`.

    Returns:
        The list of paths written, in :data:`PRESET_NAMES` order.
    """
    assert_no_seed_overlap()
    assert_parameter_axis_separation()
    paths: list[str] = []
    for name in PRESET_NAMES:
        path = preset_path(name, directory)
        save_scenario_config(make_preset(name), path)
        paths.append(path)
    return paths


def load_preset(name: str, directory: str | None = None) -> ScenarioConfig:
    """Load a preset from disk, falling back to the in-code definition.

    Args:
        name: split name.
        directory: source directory; defaults to :data:`PRESETS_DIR`.

    Returns:
        The configuration from ``<directory>/<name>.json`` if that file
        exists, otherwise :func:`make_preset`.
    """
    path = preset_path(name, directory)
    if os.path.isfile(path):
        return load_scenario_config(path)
    return make_preset(name)
