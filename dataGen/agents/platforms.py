"""Performance-envelope catalogue: one :class:`PerformanceEnvelope` per platform.

This module is the single place where "what a platform physically cannot do"
is defined. Everything else in dataGen proposes commands; the envelope clips
them (doc 3.3 (a): manoeuvres must stay physically possible).

Sampling philosophy
-------------------
Every value is drawn from a *family band* rather than being a fixed constant,
so two multirotors in the same episode are not clones. The bands are centred
on the doc 3.2 rev.2 baseline table and are typically +/-20 %. Draw order is
fixed per platform, so ``make_envelope`` is fully reproducible given the
generator state.

Units
-----
SI throughout on the returned object: metres, m/s, m/s^2, radians, rad/s,
rad/s^2, kilograms, m^2. :data:`PLATFORM_DEFAULTS` authors *angles* in degrees
(keys carry a ``_deg`` suffix) purely for human readability; the builder
converts them.

PARAMETER PROVENANCE (doc 12 requires a "parameter - value - source" table)
--------------------------------------------------------------------------
The honest status today is that *every number in this file is ``assumed``*
except gravity. What differs between rows is whether a published document
*bounds* the assumption. This is recorded machine-readably in
:data:`PARAMETER_PROVENANCE` and reproduced here:

  TRACED TO A STANDARD
    G = 9.80665 m/s^2                     ISO 80000-3 / CGPM (core.constants)
    standard rate turn = 3 deg/s          basic airmanship; used only as the
                                          sanity reference in the notes, never
                                          as a cap (doc 7.2 uses it for the RL
                                          action space, not for traffic)

  ASSUMED, BUT BOUNDED BY A CITEABLE LIMIT
    MULTIROTOR v_max 16-20 m/s            below the 14 CFR 107.51 ground-speed
                                          limit of 100 mph (44.7 m/s)
    FIXED_WING_UAS v 20-40 m/s            same Part 107 bound; small fixed-wing
                                          cruise sits well inside it
    EVTOL_UAM bank_max 22-28 deg          PASSENGER COMFORT limit, not a
                                          structural one. This is the number
                                          doc 7.2 leans on when a reviewer asks
                                          "why not just hover / turn harder?"

  ASSUMED, SOURCE STILL OUTSTANDING (doc 12.5 checklist)
    BIRD speed / turn agility             needs birdstrike + avoidance-behaviour
                                          literature
    EVTOL_UAM cruise speed 30-80 m/s      needs manufacturer spec or a NASA
                                          reference-vehicle model
    GA speed / bank                       needs a light-aircraft POH figure
    DEBRIS mass / drag area / wind        no standard exists; pure assumption
    all omega_dot_max (roll onset)        no public figure was used
    all wind_susceptibility               a modelling device, not a measurement
    all a_max / a_min                     plausible, unsourced
    UNKNOWN OOD multipliers               deliberately arbitrary; their job is
                                          to leave the trained envelope, not to
                                          model a real aircraft

Any row that is still ``assumed`` at publication time must appear in the paper
table marked as such and must enter the sensitivity analysis (doc 12 principle).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.constants import EPS_V, G, RAD
from core.state import DEFAULT_STABILITY, PerformanceEnvelope, StabilityProfile
from core.types import PLATFORM_COLORS, TYPE_NAMES, IntentType, PlatformType

__all__ = [
    "PLATFORM_DEFAULTS",
    "STABILITY_PROFILES",
    "INTENT_DWELL_MULTIPLIER",
    "stability_profile",
    "intent_dwell_multiplier",
    "PARAMETER_PROVENANCE",
    "UNKNOWN_BASE_CHOICES",
    "UNKNOWN_V_MAX_FACTOR",
    "UNKNOWN_OMEGA_MAX_FACTOR",
    "UNKNOWN_OMEGA_DOT_FACTOR",
    "UNKNOWN_BANK_CAP_RAD",
    "UNKNOWN_DWELL_SCALE",
    "TRAINING_PLATFORMS",
    "OOD_PLATFORMS",
    "make_envelope",
    "make_envelope_with_meta",
    "is_ood_platform",
    "mode_dwell_scale",
    "envelope_summary_table",
    "default_envelope",
]


# --------------------------------------------------------------------------
# Split used by doc 8.1 (dataset splitting). Kept here so scenario/presets and
# analysis can cite one definition instead of re-listing enum members.
# --------------------------------------------------------------------------

#: Platforms that may appear in the training / validation / Test-ID mix.
TRAINING_PLATFORMS: tuple[PlatformType, ...] = (
    PlatformType.MULTIROTOR,
    PlatformType.FIXED_WING_UAS,
    PlatformType.EVTOL_UAM,
)

#: Platforms reserved for the out-of-distribution split (doc 8.1).
OOD_PLATFORMS: tuple[PlatformType, ...] = (
    PlatformType.BIRD,
    PlatformType.GA,
    PlatformType.DEBRIS,
    PlatformType.UNKNOWN,
)


# --------------------------------------------------------------------------
# UNKNOWN (deliberate OOD class, doc 8.1)
# --------------------------------------------------------------------------

#: Base platforms an UNKNOWN agent may be built from (DEBRIS excluded: an
#: unpowered body pushed outside its envelope is not a meaningful OOD case).
UNKNOWN_BASE_CHOICES: tuple[PlatformType, ...] = (
    PlatformType.BIRD,
    PlatformType.MULTIROTOR,
    PlatformType.FIXED_WING_UAS,
    PlatformType.EVTOL_UAM,
    PlatformType.GA,
)

#: Multiplier band applied to the base ``v_max`` [-].
UNKNOWN_V_MAX_FACTOR: tuple[float, float] = (1.3, 1.6)

#: Multiplier band applied to every turn-rate cap of the base [-].
UNKNOWN_OMEGA_MAX_FACTOR: tuple[float, float] = (1.4, 1.8)

#: Multiplier band applied to the base ``omega_dot_max`` (roll onset) [-].
UNKNOWN_OMEGA_DOT_FACTOR: tuple[float, float] = (1.5, 2.0)

#: Hard ceiling [rad] on the boosted bank angle, so the boost cannot produce
#: ``tan(bank) -> inf``. 75 deg; every base bank is <= 50 deg, so this ceiling
#: never binds and the boosted envelope is always strictly wider than its base.
UNKNOWN_BANK_CAP_RAD: float = math.radians(75.0)

#: Legacy mode-dwell multiplier [-] for UNKNOWN agents. SUPERSEDED by
#: ``STABILITY_PROFILES[UNKNOWN]``, which states the OOD dwell directly
#: (1.0 s floor / 2.0 s mean) instead of scaling a trained platform's. It is
#: retained only because it is recorded in episode metadata and shown in the
#: GUI catalogue table; :class:`agents.maneuver.ModeMachine` does NOT apply it,
#: and applying it on top of the profile would double-count the shortening.
UNKNOWN_DWELL_SCALE: float = 0.45


# --------------------------------------------------------------------------
# Manoeuvre stability: how COMMITTED each platform is (doc 3.3 (b))
# --------------------------------------------------------------------------
#
# There is no global minimum dwell. A 1.5 s floor is right for a bird and
# absurd for a passenger eVTOL: a UAM aircraft that re-decides its manoeuvre
# every 1.5 s produces a mushy wandering trace rather than an aircraft
# trajectory, and doc 7.2 warns that reviewers notice exactly this. So dwell,
# turn aggressiveness, how hard the mode overrides the intent, and how sticky
# the transition matrix is are all PER PLATFORM.
#
# The ordering below is the whole design in one line -- BIRD and UNKNOWN are
# twitchy and hard to predict, EVTOL_UAM and GA are committed and easy, and
# that spread is a FEATURE: it is what lets the paper show a calibrated model
# modulating its sigma by class (doc 1, doc 5.2). See
# analysis.cv_baseline.EXPECTED_DIFFICULTY, which judges each class against
# its own band rather than one global band.
#
#   PLATFORM        dwell_min  dwell_mean  turn_frac   authority  diag_bias
#   BIRD               1.0 s      2.5 s    0.40-1.00     0.90       0.00
#   MULTIROTOR         2.0 s      4.5 s    0.30-0.90     0.80       0.05
#   FIXED_WING_UAS     4.0 s      9.0 s    0.30-0.80     0.65       0.15
#   EVTOL_UAM         12.0 s     25.0 s    0.15-0.45     0.35       0.35
#   GA                 8.0 s     18.0 s    0.20-0.60     0.50       0.25
#   DEBRIS               n/a        n/a          n/a       n/a        n/a
#   UNKNOWN            1.0 s      2.0 s    0.50-1.00     0.95      -0.10

STABILITY_PROFILES: dict[int, StabilityProfile] = {
    # Flapping flight: decisive, fully committed, re-decided every couple of
    # seconds. The hardest class to predict, on purpose.
    int(PlatformType.BIRD): StabilityProfile(
        dwell_min_s=1.0, dwell_mean_s=2.5,
        turn_frac_lo=0.40, turn_frac_hi=1.00,
        mode_authority=0.90, diag_bias=0.00,
    ),
    # sUAS rotary: agile but flown to a task, so a few seconds of commitment.
    int(PlatformType.MULTIROTOR): StabilityProfile(
        dwell_min_s=2.0, dwell_mean_s=4.5,
        turn_frac_lo=0.30, turn_frac_hi=0.90,
        mode_authority=0.80, diag_bias=0.05,
    ),
    # Fixed-wing cannot stop and rolls slowly: legs, not twitches.
    int(PlatformType.FIXED_WING_UAS): StabilityProfile(
        dwell_min_s=4.0, dwell_mean_s=9.0,
        turn_frac_lo=0.30, turn_frac_hi=0.80,
        mode_authority=0.65, diag_bias=0.15,
    ),
    # Passenger UAM in corridor cruise. The 12 s floor and 0.35 authority are
    # what stop it wandering off its centreline; the 0.15-0.45 turn band keeps
    # every manoeuvre well inside the ride-comfort bank limit.
    int(PlatformType.EVTOL_UAM): StabilityProfile(
        dwell_min_s=12.0, dwell_mean_s=25.0,
        turn_frac_lo=0.15, turn_frac_hi=0.45,
        mode_authority=0.35, diag_bias=0.35,
    ),
    # Manned light aircraft: gentle, deliberate, hand-flown.
    int(PlatformType.GA): StabilityProfile(
        dwell_min_s=8.0, dwell_mean_s=18.0,
        turn_frac_lo=0.20, turn_frac_hi=0.60,
        mode_authority=0.50, diag_bias=0.25,
    ),
    # Unpowered: NEVER CONSULTED for the mode command, because the mode machine
    # pins an ``is_powered == False`` body to CV and agents.base routes it to
    # the passive integrator. Present (inert: zero authority, long dwell) so
    # that every platform key resolves and no caller needs a special case.
    int(PlatformType.DEBRIS): StabilityProfile(
        dwell_min_s=5.0, dwell_mean_s=30.0,
        turn_frac_lo=0.00, turn_frac_hi=0.00,
        mode_authority=0.00, diag_bias=0.00,
    ),
    # The deliberate OOD class: erratic BY DESIGN. The NEGATIVE diag_bias makes
    # it switch mode more often than any trained platform ever does, which is
    # the point -- the predictor must answer with a larger sigma instead of
    # confidently assigning it to a trained class.
    int(PlatformType.UNKNOWN): StabilityProfile(
        dwell_min_s=1.0, dwell_mean_s=2.0,
        turn_frac_lo=0.50, turn_frac_hi=1.00,
        mode_authority=0.95, diag_bias=-0.10,
    ),
}


#: Multiplier [-] applied to a platform's ``dwell_mean_s`` by the agent's
#: intent: the same airframe holds a manoeuvre far longer in corridor cruise
#: than while searching. The PLATFORM floor ``dwell_min_s`` is applied AFTER
#: this multiplication (in :func:`agents.behavior.sample_dwell`), so a
#: wandering eVTOL is still far steadier than a wandering bird.
#: ``PASSIVE_DRIFT`` is n/a -- an unpowered body has no manoeuvre modes -- and
#: is present as 1.0 only so the table is total.
INTENT_DWELL_MULTIPLIER: dict[int, float] = {
    int(IntentType.CORRIDOR_TRANSIT): 2.00,
    int(IntentType.DELIVERY_WP): 1.00,
    int(IntentType.LOITER_HOVER): 1.20,
    int(IntentType.SEARCH_WANDER): 0.45,
    int(IntentType.REACTIVE_AVOID): 0.70,
    int(IntentType.PASSIVE_DRIFT): 1.00,
}


def stability_profile(platform: PlatformType) -> StabilityProfile:
    """Manoeuvre-commitment profile of ``platform``.

    Args:
        platform: platform class.

    Returns:
        Its :class:`~core.state.StabilityProfile`, or
        :data:`~core.state.DEFAULT_STABILITY` for a platform with no entry
        (which cannot happen for a real :class:`~core.types.PlatformType`;
        the fallback exists so a future enum member degrades to the old
        behaviour instead of raising inside the simulation loop).
    """
    return STABILITY_PROFILES.get(int(PlatformType(int(platform))), DEFAULT_STABILITY)


def intent_dwell_multiplier(intent: IntentType | int) -> float:
    """Dwell multiplier [-] of an intent (see :data:`INTENT_DWELL_MULTIPLIER`).

    Args:
        intent: the agent's effective intent.

    Returns:
        The multiplier; 1.0 for an intent with no entry.
    """
    return float(INTENT_DWELL_MULTIPLIER.get(int(intent), 1.0))


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------
#
# Each ``sample`` value is either a scalar (fixed) or a ``(lo, hi)`` band drawn
# uniformly. ``omega_max_deg = None``     -> derive from the binding turn model
#                                            at ``v_min`` (see _build_envelope)
#            ``lateral_accel_max = None`` -> derive as G*tan(bank_max)
#
# ``bank_max_deg`` is stored for every platform. For platforms with
# ``bank_limited = False`` it is INFORMATIONAL ONLY (an equivalent bank angle
# for display); :meth:`PerformanceEnvelope.max_turn_rate` ignores it there.

PLATFORM_DEFAULTS: dict[int, dict[str, Any]] = {
    int(PlatformType.BIRD): {
        "label": TYPE_NAMES[int(PlatformType.BIRD)],
        "family": "biological",
        "split": "ood",
        "is_powered": True,
        "can_hover": False,
        "bank_limited": False,
        "doc_speed_range_ms": (5.0, 25.0),
        "sample": {
            "v_min": (5.0, 8.0),
            "v_max": (20.0, 25.0),
            "a_max": (2.4, 3.6),
            "a_min": (-3.6, -2.4),
            "omega_max_deg": (48.0, 72.0),
            "omega_dot_deg": (100.0, 140.0),
            "bank_max_deg": (55.0, 65.0),
            "lateral_accel_max": (10.0, 14.0),
            "wind_susceptibility": (0.85, 0.95),
            "mass": (0.5, 2.0),
            "drag_area": (0.005, 0.02),
        },
        "notes": (
            "Hardest class to predict: 60 deg/s nominal turn rate with a "
            "120 deg/s^2 roll onset means a bird can reverse course inside one "
            "logged step. Not bank limited (flapping flight is not a "
            "coordinated turn), so the absolute omega_max governs and the "
            "stored bank/lateral values are display-only. Doc 3.2 rev.2 quotes "
            "10-25 m/s; the band is widened down to 5 m/s so slow soaring and "
            "loitering birds exist in the data."
        ),
    },
    int(PlatformType.MULTIROTOR): {
        "label": TYPE_NAMES[int(PlatformType.MULTIROTOR)],
        "family": "sUAS rotary",
        "split": "train",
        "is_powered": True,
        "can_hover": True,
        "bank_limited": False,
        "doc_speed_range_ms": (0.0, 20.0),
        "sample": {
            "v_min": 0.0,
            "v_max": (16.0, 20.0),
            "a_max": (3.2, 4.8),
            "a_min": (-4.8, -3.2),
            "omega_max_deg": (72.0, 108.0),
            "omega_dot_deg": (75.0, 105.0),
            "bank_max_deg": 30.0,
            "lateral_accel_max": (4.5, 6.8),
            "wind_susceptibility": (0.50, 0.70),
            "mass": (1.5, 6.0),
            "drag_area": (0.03, 0.12),
        },
        "notes": (
            "Tilt limited, not bank limited: the turn-rate cap is "
            "lateral_accel_max / v, with the baseline 5.66 m/s^2 = G*tan(30 "
            "deg). Can stop (v_min = 0), which is what makes HOVER a legal "
            "mode. Speed band sits far inside the 14 CFR 107.51 ground-speed "
            "limit of 100 mph."
        ),
    },
    int(PlatformType.FIXED_WING_UAS): {
        "label": TYPE_NAMES[int(PlatformType.FIXED_WING_UAS)],
        "family": "sUAS fixed-wing",
        "split": "train",
        "is_powered": True,
        "can_hover": False,
        "bank_limited": True,
        "doc_speed_range_ms": (20.0, 40.0),
        "sample": {
            "v_min": (20.0, 23.0),
            "v_max": (36.0, 40.0),
            "a_max": (1.6, 2.4),
            "a_min": (-2.4, -1.6),
            "omega_max_deg": None,
            "omega_dot_deg": (25.0, 35.0),
            "bank_max_deg": (40.0, 50.0),
            "lateral_accel_max": None,
            "wind_susceptibility": (0.45, 0.55),
            "mass": (8.0, 25.0),
            "drag_area": (0.08, 0.25),
        },
        "notes": (
            "Coordinated-turn platform: omega_cap = G*tan(bank)/v, so the "
            "achievable turn rate FALLS as it speeds up. Cannot stop. This is "
            "the class that produces the clean minimum-turn-radius structure "
            "a CV extrapolator misses."
        ),
    },
    int(PlatformType.EVTOL_UAM): {
        "label": TYPE_NAMES[int(PlatformType.EVTOL_UAM)],
        "family": "UAM",
        "split": "train",
        "is_powered": True,
        "can_hover": True,
        "bank_limited": True,
        "doc_speed_range_ms": (30.0, 80.0),
        "sample": {
            "v_min": (30.0, 33.0),
            "v_max": (72.0, 80.0),
            "a_max": (1.2, 1.8),
            "a_min": (-1.8, -1.2),
            "omega_max_deg": None,
            "omega_dot_deg": (4.0, 6.0),
            "bank_max_deg": (22.0, 28.0),
            "lateral_accel_max": None,
            "wind_susceptibility": (0.20, 0.30),
            "mass": (1500.0, 2500.0),
            "drag_area": (1.5, 3.5),
        },
        "notes": (
            "The 25 deg nominal bank is a PASSENGER COMFORT limit, not a "
            "structural or aerodynamic one -- this is the reviewer defence of "
            "doc 7.2: an eVTOL in cruise does not simply hover to avoid a "
            "conflict, because (a) ride quality, (b) energy, (c) corridor "
            "spacing behind it. can_hover is True (the airframe can), but the "
            "scope of this paper is the cruise phase only, so v_min is held at "
            "~30 m/s and a HOVER command cannot actually stop the vehicle. "
            "The 5 deg/s^2 roll onset makes it the most sluggish class."
        ),
    },
    int(PlatformType.GA): {
        "label": TYPE_NAMES[int(PlatformType.GA)],
        "family": "manned",
        "split": "ood",
        "is_powered": True,
        "can_hover": False,
        "bank_limited": True,
        "doc_speed_range_ms": (40.0, 70.0),
        "sample": {
            "v_min": (40.0, 44.0),
            "v_max": (64.0, 70.0),
            "a_max": (1.2, 1.8),
            "a_min": (-1.8, -1.2),
            "omega_max_deg": None,
            "omega_dot_deg": (12.0, 18.0),
            "bank_max_deg": (27.0, 33.0),
            "lateral_accel_max": None,
            "wind_susceptibility": (0.30, 0.40),
            "mass": (900.0, 1500.0),
            "drag_area": (1.0, 2.5),
        },
        "notes": (
            "Light aircraft with no ADS-B out: fast, gentle, bank limited at "
            "about 30 deg. Its cap at cruise (~8 deg/s) is comfortably above "
            "the 3 deg/s standard rate turn, so standard-rate manoeuvring is "
            "always inside the envelope."
        ),
    },
    int(PlatformType.DEBRIS): {
        "label": TYPE_NAMES[int(PlatformType.DEBRIS)],
        "family": "unpowered",
        "split": "ood",
        "is_powered": False,
        "can_hover": False,
        "bank_limited": False,
        "doc_speed_range_ms": (0.0, 25.0),
        "sample": {
            "v_min": 0.0,
            "v_max": (15.0, 25.0),
            "a_max": (0.3, 0.8),
            "a_min": (-4.0, -2.0),
            "omega_max_deg": (10.0, 30.0),
            "omega_dot_deg": (30.0, 90.0),
            "bank_max_deg": 0.0,
            "lateral_accel_max": (1.0, 3.0),
            "wind_susceptibility": 1.0,
            "mass": (0.2, 2.0),
            "drag_area": (0.01, 0.08),
        },
        "notes": (
            "is_powered = False: agents.base routes this class to the passive "
            "integrator, where mass and drag_area (Cd*A) drive the ballistic "
            "response and wind_susceptibility = 1.0 advects it fully with the "
            "air mass. Its v is a GROUND speed, not an airspeed (core.state). "
            "The a_max / a_min / omega caps are bookkeeping bounds so that "
            "clamping code never sees a degenerate envelope; they are not "
            "control authority."
        ),
    },
    int(PlatformType.UNKNOWN): {
        "label": TYPE_NAMES[int(PlatformType.UNKNOWN)],
        "family": "out-of-distribution",
        "split": "ood",
        "is_powered": True,
        "can_hover": False,
        "bank_limited": False,
        "doc_speed_range_ms": (0.0, 128.0),
        "sample": {},  # synthesised from a base platform, see _build_unknown
        "notes": (
            "The deliberate OOD class (doc 8.1). A random base platform is "
            "drawn and then pushed OUTSIDE the trained envelope: v_max x "
            "1.3-1.6, every turn-rate cap x 1.4-1.8, roll onset x 1.5-2.0, and "
            "a shorter mode dwell. The point is NOT realism. The point is that "
            "the predictor must answer with a LARGER SIGMA instead of "
            "confidently assigning it to a trained class (doc 5.2, doc 9.5)."
        ),
    },
}


#: Machine-readable provenance for doc 12's parameter table.
#: value: (status, source-or-bound). status is 'traced' or 'assumed'.
PARAMETER_PROVENANCE: dict[str, tuple[str, str]] = {
    "G": ("traced", "ISO 80000-3 / CGPM standard gravity (core.constants.G)"),
    "standard_rate_turn": ("traced", "3 deg/s, basic airmanship; reference only"),
    "MULTIROTOR.v_max": (
        "assumed",
        "bounded by 14 CFR 107.51 ground-speed limit (100 mph = 44.7 m/s)",
    ),
    "FIXED_WING_UAS.v_range": (
        "assumed",
        "bounded by 14 CFR 107.51; typical small fixed-wing cruise",
    ),
    "EVTOL_UAM.bank_max": (
        "assumed",
        "passenger ride-comfort limit; doc 7.2 defence paragraph",
    ),
    "EVTOL_UAM.v_range": (
        "assumed",
        "OUTSTANDING: manufacturer spec or NASA reference vehicle (doc 12.4)",
    ),
    "BIRD.v_range": ("assumed", "OUTSTANDING: birdstrike literature (doc 12.4)"),
    "BIRD.omega_max": (
        "assumed",
        "OUTSTANDING: avian avoidance-behaviour observation study (doc 12.4)",
    ),
    "GA.v_range": ("assumed", "OUTSTANDING: light-aircraft POH cruise figures"),
    "GA.bank_max": ("assumed", "typical enroute manoeuvring bank"),
    "DEBRIS.mass": ("assumed", "no standard exists"),
    "DEBRIS.drag_area": ("assumed", "no standard exists"),
    "omega_dot_max": ("assumed", "roll-onset rate; no public figure used"),
    "wind_susceptibility": ("assumed", "modelling device, not a measurement"),
    "a_max": ("assumed", "plausible longitudinal authority, unsourced"),
    "a_min": ("assumed", "plausible longitudinal authority, unsourced"),
    "UNKNOWN.factors": (
        "assumed",
        "deliberately arbitrary; purpose is to leave the trained envelope",
    ),
    "stability.dwell": (
        "assumed",
        "manoeuvre commitment per platform; no public figure. Ordering (eVTOL "
        ">> GA > fixed-wing > multirotor > bird) is the defensible claim, not "
        "the absolute seconds; enters the sensitivity analysis (doc 12)",
    ),
    "stability.mode_authority": (
        "assumed",
        "modelling device: how far a discrete manoeuvre overrides guidance",
    ),
    "stability.diag_bias": (
        "assumed",
        "modelling device: stickiness of the mode transition matrix",
    ),
}


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------


def _draw(rng: np.random.Generator, spec: Any) -> float:
    """Draw one value from a ``(lo, hi)`` band or a fixed scalar.

    A draw is always consumed, even for a fixed scalar, so that the number of
    random numbers used by :func:`make_envelope` depends only on the platform
    and never on the particular values in the table.

    Args:
        rng: generator (typically ``RNGHub.stream('platform')``).
        spec: either a scalar or a ``(lo, hi)`` sequence, in the unit of the
            parameter being sampled.

    Returns:
        The sampled value [same unit as ``spec``].
    """
    if isinstance(spec, (tuple, list)):
        lo = float(spec[0])
        hi = float(spec[1])
    else:
        lo = hi = float(spec)
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def _build_envelope(
    entry: Mapping[str, Any],
    rng: np.random.Generator,
    platform: PlatformType,
) -> PerformanceEnvelope:
    """Sample one envelope from a :data:`PLATFORM_DEFAULTS` entry.

    Args:
        entry: a catalogue entry (must carry a non-empty ``sample`` dict).
        rng: generator.
        platform: the platform the entry belongs to; selects the
            :class:`~core.state.StabilityProfile` attached to the result. It is
            passed explicitly rather than read from ``entry`` because
            ``_build_unknown`` builds a BASE platform's envelope and then
            overrides the profile with the OOD one.

    Returns:
        A fully populated :class:`PerformanceEnvelope` in SI units. No random
        draw is consumed by the stability profile: commitment is a fixed
        property of the class, so two multirotors differ in their limits but
        agree on how long they hold a manoeuvre.
    """
    s: Mapping[str, Any] = entry["sample"]
    bank_limited = bool(entry["bank_limited"])
    can_hover = bool(entry["can_hover"])
    is_powered = bool(entry["is_powered"])

    # Draw order is fixed -- do not reorder, it changes every seeded dataset.
    v_min = _draw(rng, s["v_min"])
    v_max = _draw(rng, s["v_max"])
    a_max = _draw(rng, s["a_max"])
    a_min = _draw(rng, s["a_min"])
    omega_dot_max = math.radians(_draw(rng, s["omega_dot_deg"]))
    bank_max = math.radians(_draw(rng, s["bank_max_deg"]))
    lateral_spec = s["lateral_accel_max"]
    lateral_accel_max = (
        G * math.tan(bank_max) if lateral_spec is None else _draw(rng, lateral_spec)
    )
    omega_spec = s["omega_max_deg"]
    wind_susceptibility = _draw(rng, s["wind_susceptibility"])
    mass = _draw(rng, s["mass"])
    drag_area = _draw(rng, s["drag_area"])

    if v_max <= v_min:  # numerical guard; bands are authored so this cannot fire
        v_max = v_min + 1.0

    if omega_spec is None:
        # Derive the absolute cap so that it is exactly non-binding at v_min:
        # the speed-dependent model governs across the whole speed band.
        v_ref = v_min if v_min > EPS_V else EPS_V
        if bank_limited:
            omega_max = G * math.tan(bank_max) / v_ref
        else:
            omega_max = lateral_accel_max / v_ref
    else:
        omega_max = math.radians(_draw(rng, omega_spec))

    return PerformanceEnvelope(
        v_min=v_min,
        v_max=v_max,
        a_max=a_max,
        a_min=a_min,
        omega_max=omega_max,
        omega_dot_max=omega_dot_max,
        bank_max=bank_max,
        bank_limited=bank_limited,
        lateral_accel_max=lateral_accel_max,
        can_hover=can_hover,
        is_powered=is_powered,
        wind_susceptibility=min(max(wind_susceptibility, 0.0), 1.0),
        mass=mass,
        drag_area=drag_area,
        stability=stability_profile(platform),
    )


def _build_unknown(
    rng: np.random.Generator,
) -> tuple[PerformanceEnvelope, PlatformType, PerformanceEnvelope, dict[str, float]]:
    """Build an out-of-distribution envelope by inflating a base platform.

    Every cap that feeds :meth:`PerformanceEnvelope.max_turn_rate` is scaled by
    the SAME turn factor -- the absolute ``omega_max``, the tilt-derived
    ``lateral_accel_max`` and ``tan(bank_max)`` -- so the boosted envelope is
    strictly wider than its base at every airspeed, whichever regime binds.

    Args:
        rng: generator.

    Returns:
        ``(boosted, base_platform, base_envelope, factors)`` where ``factors``
        maps ``'v_max' / 'omega_max' / 'omega_dot'`` to the multipliers used [-].
    """
    idx = int(rng.integers(0, len(UNKNOWN_BASE_CHOICES)))
    base_platform = UNKNOWN_BASE_CHOICES[idx]
    base = _build_envelope(PLATFORM_DEFAULTS[int(base_platform)], rng, base_platform)

    f_v = float(rng.uniform(*UNKNOWN_V_MAX_FACTOR))
    f_w = float(rng.uniform(*UNKNOWN_OMEGA_MAX_FACTOR))
    f_d = float(rng.uniform(*UNKNOWN_OMEGA_DOT_FACTOR))
    f_wind = float(rng.uniform(0.8, 1.2))

    if base.bank_max > 0.0:
        bank_max = min(math.atan(math.tan(base.bank_max) * f_w), UNKNOWN_BANK_CAP_RAD)
    else:
        bank_max = base.bank_max

    boosted = PerformanceEnvelope(
        v_min=base.v_min,
        v_max=base.v_max * f_v,
        a_max=base.a_max * f_v,
        a_min=base.a_min * f_v,
        omega_max=base.omega_max * f_w,
        omega_dot_max=base.omega_dot_max * f_d,
        bank_max=bank_max,
        bank_limited=base.bank_limited,
        lateral_accel_max=base.lateral_accel_max * f_w,
        can_hover=base.can_hover,
        is_powered=base.is_powered,
        wind_susceptibility=min(max(base.wind_susceptibility * f_wind, 0.0), 1.0),
        mass=base.mass,
        drag_area=base.drag_area,
        # The OOD profile REPLACES the base platform's: an UNKNOWN built from
        # an eVTOL must not inherit a 12 s commitment floor, or it would be the
        # most predictable object in the dataset.
        stability=stability_profile(PlatformType.UNKNOWN),
    )
    factors = {
        "v_max": f_v,
        "a_max": f_v,
        "omega_max": f_w,
        "lateral_accel_max": f_w,
        "bank_max_tan": f_w,
        "omega_dot": f_d,
        "wind_susceptibility": f_wind,
    }
    return boosted, base_platform, base, factors


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def make_envelope(
    platform: PlatformType,
    rng: np.random.Generator,
) -> PerformanceEnvelope:
    """Sample a performance envelope for ``platform``.

    Args:
        platform: the platform class to build.
        rng: generator, normally ``RNGHub.stream('platform')``. The number of
            draws consumed depends only on ``platform``.

    Returns:
        A :class:`PerformanceEnvelope` in SI units. Family bands guarantee
        ``v_min < v_max``, ``a_min < 0 < a_max`` and strictly positive caps.

    Raises:
        KeyError: if ``platform`` has no catalogue entry.
    """
    return make_envelope_with_meta(platform, rng)[0]


def make_envelope_with_meta(
    platform: PlatformType,
    rng: np.random.Generator,
) -> tuple[PerformanceEnvelope, dict[str, Any]]:
    """Sample an envelope and return the provenance metadata alongside it.

    :class:`PerformanceEnvelope` has no field in which to "mark" an
    out-of-distribution agent, so the mark is returned here instead of being
    smuggled into the envelope. Callers that need to record OOD-ness in episode
    metadata, or that need the shortened mode dwell of doc 8.1, should use this
    function; everything else can use :func:`make_envelope`.

    Args:
        platform: the platform class to build.
        rng: generator.

    Returns:
        ``(envelope, meta)``. ``meta`` is JSON-safe and contains:
            ``platform``      int PlatformType value
            ``label``         display name
            ``is_ood``        True only for UNKNOWN
            ``base_platform`` int base PlatformType for UNKNOWN, else None
            ``base_envelope`` dict of the pre-boost envelope for UNKNOWN, else
                              None
            ``factors``       multipliers applied for UNKNOWN, else ``{}``
            ``dwell_scale``   mode-dwell multiplier [-] for agents.maneuver

    Raises:
        KeyError: if ``platform`` has no catalogue entry.
    """
    ptype = PlatformType(int(platform))
    entry = PLATFORM_DEFAULTS[int(ptype)]

    if ptype is PlatformType.UNKNOWN:
        env, base_platform, base_env, factors = _build_unknown(rng)
        meta: dict[str, Any] = {
            "platform": int(ptype),
            "label": entry["label"],
            "is_ood": True,
            "base_platform": int(base_platform),
            "base_envelope": base_env.to_dict(),
            "factors": factors,
            "dwell_scale": UNKNOWN_DWELL_SCALE,
            "stability": env.stability.to_dict(),
        }
        return env, meta

    env = _build_envelope(entry, rng, ptype)
    meta = {
        "platform": int(ptype),
        "label": entry["label"],
        "is_ood": False,
        "base_platform": None,
        "base_envelope": None,
        "factors": {},
        "dwell_scale": 1.0,
        "stability": env.stability.to_dict(),
    }
    return env, meta


def is_ood_platform(platform: PlatformType) -> bool:
    """True if ``platform`` belongs to the doc 8.1 out-of-distribution split.

    Args:
        platform: platform class.

    Returns:
        True for BIRD / GA / DEBRIS / UNKNOWN, False for the trained mix.
    """
    return PlatformType(int(platform)) in OOD_PLATFORMS


def mode_dwell_scale(platform: PlatformType) -> float:
    """LEGACY dwell multiplier [-]; NOT applied by the simulator any more.

    Superseded by :data:`STABILITY_PROFILES`, which gives every platform --
    including UNKNOWN -- its own ``dwell_min_s`` / ``dwell_mean_s`` outright.
    :class:`agents.maneuver.ModeMachine` reads the profile and ignores this
    function; multiplying by it as well would shorten the OOD dwell twice.
    Kept because the value is recorded in episode metadata and displayed in
    the GUI catalogue, where it still describes the intent of the OOD class.

    Args:
        platform: platform class.

    Returns:
        1.0 for every trained platform, :data:`UNKNOWN_DWELL_SCALE` for UNKNOWN.
    """
    if PlatformType(int(platform)) is PlatformType.UNKNOWN:
        return UNKNOWN_DWELL_SCALE
    return 1.0


def default_envelope(platform: PlatformType) -> PerformanceEnvelope:
    """Deterministic mid-band envelope, for GUI previews and unit tests.

    No randomness is involved: every band collapses to its midpoint. This is a
    display / sanity helper, NOT the function used to build episode agents --
    using it for data generation would make every agent of a class a clone,
    which is exactly what the sampling exists to prevent.

    Args:
        platform: platform class.

    Returns:
        The mid-band :class:`PerformanceEnvelope` in SI units.
    """
    ptype = PlatformType(int(platform))
    if ptype is PlatformType.UNKNOWN:
        # Mid-band of the mid-band base (first choice) with mid multipliers.
        base = default_envelope(UNKNOWN_BASE_CHOICES[0])
        f_v = 0.5 * (UNKNOWN_V_MAX_FACTOR[0] + UNKNOWN_V_MAX_FACTOR[1])
        f_w = 0.5 * (UNKNOWN_OMEGA_MAX_FACTOR[0] + UNKNOWN_OMEGA_MAX_FACTOR[1])
        f_d = 0.5 * (UNKNOWN_OMEGA_DOT_FACTOR[0] + UNKNOWN_OMEGA_DOT_FACTOR[1])
        bank = (
            min(math.atan(math.tan(base.bank_max) * f_w), UNKNOWN_BANK_CAP_RAD)
            if base.bank_max > 0.0
            else base.bank_max
        )
        return PerformanceEnvelope(
            v_min=base.v_min,
            v_max=base.v_max * f_v,
            a_max=base.a_max * f_v,
            a_min=base.a_min * f_v,
            omega_max=base.omega_max * f_w,
            omega_dot_max=base.omega_dot_max * f_d,
            bank_max=bank,
            bank_limited=base.bank_limited,
            lateral_accel_max=base.lateral_accel_max * f_w,
            can_hover=base.can_hover,
            is_powered=base.is_powered,
            wind_susceptibility=base.wind_susceptibility,
            mass=base.mass,
            drag_area=base.drag_area,
            stability=stability_profile(PlatformType.UNKNOWN),
        )
    return _build_envelope(PLATFORM_DEFAULTS[int(ptype)], _MidpointRNG(), ptype)


def envelope_summary_table() -> list[dict[str, Any]]:
    """JSON-safe catalogue table for the GUI and for the paper's parameter table.

    Returns:
        One row per :class:`~core.types.PlatformType`, ordered by enum value.
        Angular quantities are reported in DEGREES (``*_deg``) because the row
        is meant for human display; the simulator itself never reads this.
        Each row carries the mid-band envelope, the authored bands, the split
        label, the display colour and the free-text notes.
    """
    rows: list[dict[str, Any]] = []
    for ptype in PlatformType:
        entry = PLATFORM_DEFAULTS[int(ptype)]
        env = default_envelope(ptype)
        rows.append(
            {
                "platform": int(ptype),
                "label": entry["label"],
                "family": entry["family"],
                "split": entry["split"],
                "color": PLATFORM_COLORS[ptype],
                "doc_speed_range_ms": list(entry["doc_speed_range_ms"]),
                "v_min_ms": env.v_min,
                "v_max_ms": env.v_max,
                "a_max_ms2": env.a_max,
                "a_min_ms2": env.a_min,
                "omega_max_deg_s": env.omega_max * RAD,
                "omega_dot_max_deg_s2": env.omega_dot_max * RAD,
                "bank_max_deg": env.bank_max * RAD,
                "bank_limited": env.bank_limited,
                "lateral_accel_max_ms2": env.lateral_accel_max,
                "can_hover": env.can_hover,
                "is_powered": env.is_powered,
                "wind_susceptibility": env.wind_susceptibility,
                "mass_kg": env.mass,
                "drag_area_m2": env.drag_area,
                "turn_rate_at_cruise_deg_s": env.max_turn_rate(
                    0.5 * (env.v_min + env.v_max)
                )
                * RAD,
                "min_turn_radius_at_cruise_m": env.turn_radius(
                    0.5 * (env.v_min + env.v_max)
                ),
                "dwell_scale": mode_dwell_scale(ptype),
                "is_ood": is_ood_platform(ptype),
                "dwell_min_s": env.stability.dwell_min_s,
                # 0.0 would mean "no platform mean, use the legacy per-(intent,
                # mode) table"; no catalogue entry is in that state.
                "dwell_mean_s": float(env.stability.dwell_mean_s or 0.0),
                "turn_frac_lo": env.stability.turn_frac_lo,
                "turn_frac_hi": env.stability.turn_frac_hi,
                "mode_authority": env.stability.mode_authority,
                "diag_bias": env.stability.diag_bias,
                "notes": entry["notes"],
            }
        )
    return rows


class _MidpointRNG:
    """Deterministic stand-in for a Generator that returns band midpoints.

    Only the two methods :func:`_build_envelope` uses are implemented. It
    exists so that :func:`default_envelope` can reuse the exact construction
    logic of the sampled path instead of duplicating it.
    """

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return the midpoint of ``[low, high]`` [same unit as the arguments]."""
        return 0.5 * (float(low) + float(high))

    def integers(self, low: int, high: int | None = None) -> int:
        """Return the lowest value of the requested integer range [-]."""
        return int(low) if high is not None else 0


def _self_check() -> list[str]:
    """Internal consistency check of the catalogue (used by the test module).

    Returns:
        A list of human-readable problems; empty means the catalogue is sane.
    """
    problems: list[str] = []
    for ptype in PlatformType:
        key = int(ptype)
        if key not in PLATFORM_DEFAULTS:
            problems.append(f"{ptype.name}: missing from PLATFORM_DEFAULTS")
            continue
        entry = PLATFORM_DEFAULTS[key]
        for field_name in ("label", "family", "split", "is_powered", "can_hover",
                           "bank_limited", "doc_speed_range_ms", "sample", "notes"):
            if field_name not in entry:
                problems.append(f"{ptype.name}: entry missing '{field_name}'")
        sample = entry.get("sample", {})
        for name, spec in sample.items():
            if isinstance(spec, (tuple, list)):
                if len(spec) != 2:
                    problems.append(f"{ptype.name}.{name}: band must be (lo, hi)")
                elif float(spec[0]) > float(spec[1]):
                    problems.append(f"{ptype.name}.{name}: band lo > hi")
        if key not in STABILITY_PROFILES:
            problems.append(f"{ptype.name}: missing from STABILITY_PROFILES")
        else:
            for problem in STABILITY_PROFILES[key].validate():
                problems.append(f"{ptype.name}.stability: {problem}")
    for intent in IntentType:
        if int(intent) not in INTENT_DWELL_MULTIPLIER:
            problems.append(f"{intent.name}: missing from INTENT_DWELL_MULTIPLIER")
        elif not (INTENT_DWELL_MULTIPLIER[int(intent)] > 0.0):
            problems.append(f"{intent.name}: dwell multiplier must be > 0")
    return problems


def _band_names() -> Sequence[str]:
    """Names of the sampled parameters, in draw order (documentation helper)."""
    return (
        "v_min",
        "v_max",
        "a_max",
        "a_min",
        "omega_dot_deg",
        "bank_max_deg",
        "lateral_accel_max",
        "wind_susceptibility",
        "mass",
        "drag_area",
        "omega_max_deg",
    )
