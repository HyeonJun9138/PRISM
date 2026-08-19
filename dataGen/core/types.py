"""Enumerations and display tables shared by every PRISM dataGen module.

This module contains NO logic beyond the enum definitions and a few constant
lookup tables. It must stay import-cheap and dependency-free (stdlib only) so
that every other package can import it without creating cycles.

All enum values are stable integers: they are written to disk as ``int8`` in
:mod:`storage.schema`, so their numeric values are part of the dataset
contract and MUST NOT be renumbered once data has been generated.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "PlatformType",
    "ManeuverMode",
    "IntentType",
    "GeometryType",
    "RiskLevel",
    "PLATFORM_COLORS",
    "TYPE_NAMES",
    "MODE_NAMES",
    "INTENT_NAMES",
    "GEOMETRY_NAMES",
    "RISK_NAMES",
]


class PlatformType(IntEnum):
    """Airframe class of an agent (doc 3.2 (3), auxiliary label of doc 5.1).

    BIRD and GA are reserved for the out-of-distribution test split (doc 8.1);
    the training mix uses MULTIROTOR / FIXED_WING_UAS / EVTOL_UAM only.
    UNKNOWN exists so that observation-layer consumers can carry a track with
    no ground-truth type without inventing a class.
    """

    BIRD = 0
    MULTIROTOR = 1
    FIXED_WING_UAS = 2
    EVTOL_UAM = 3
    GA = 4
    DEBRIS = 5
    UNKNOWN = 6


class ManeuverMode(IntEnum):
    """Low-level kinematic mode currently executed by an agent (doc 3.7).

    CV       -- constant velocity, zero commanded turn rate and acceleration.
    CT_LEFT  -- coordinated turn, positive (CCW, increasing psi) turn rate.
    CT_RIGHT -- coordinated turn, negative (CW, decreasing psi) turn rate.
    CA       -- constant longitudinal acceleration, zero commanded turn rate.
    HOVER    -- speed commanded toward zero; only valid for hover-capable
                platforms (``PerformanceEnvelope.can_hover``).
    """

    CV = 0
    CT_LEFT = 1
    CT_RIGHT = 2
    CA = 3
    HOVER = 4


class IntentType(IntEnum):
    """High-level purpose driving an agent's mode decisions (doc 3.2 (1)).

    SEARCH_WANDER, REACTIVE_AVOID and PASSIVE_DRIFT are reserved for the
    out-of-distribution split (doc 8.1); the training mix uses
    CORRIDOR_TRANSIT / DELIVERY_WP / LOITER_HOVER.
    """

    CORRIDOR_TRANSIT = 0
    DELIVERY_WP = 1
    LOITER_HOVER = 2
    SEARCH_WANDER = 3
    REACTIVE_AVOID = 4
    PASSIVE_DRIFT = 5


class GeometryType(IntEnum):
    """Encounter geometry class of a designed encounter (doc 3.4).

    Approach-angle convention, measured as the absolute relative bearing
    between the ownship velocity and the intruder velocity:
        HEAD_ON     150 deg .. 180 deg
        CROSSING     30 deg .. 150 deg
        OVERTAKING    0 deg ..  30 deg
    BACKGROUND marks traffic that was not geometrically designed at all.
    """

    HEAD_ON = 0
    CROSSING = 1
    OVERTAKING = 2
    BACKGROUND = 3


class RiskLevel(IntEnum):
    """Stratification label of an episode, derived from minimum separation.

    Thresholds are applied by :mod:`analysis.metrics` against the standard
    separation criteria in :mod:`core.constants` (doc 9.4):
        NONE   -- no loss of well clear
        LOW    -- well-clear margin eroded but no DWC violation
        MEDIUM -- DWC violated, NMAC not reached
        HIGH   -- NMAC (horizontal 500 ft) reached
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


#: Per-platform display colour (hex RGB string) used by the Canvas 2D GUI.
PLATFORM_COLORS: dict[PlatformType, str] = {
    PlatformType.BIRD: "#e8c547",
    PlatformType.MULTIROTOR: "#4fc3f7",
    PlatformType.FIXED_WING_UAS: "#66bb6a",
    PlatformType.EVTOL_UAM: "#ab47bc",
    PlatformType.GA: "#ef5350",
    PlatformType.DEBRIS: "#90a4ae",
    PlatformType.UNKNOWN: "#eceff1",
}

#: Human-readable platform names, keyed by the raw integer stored on disk.
TYPE_NAMES: dict[int, str] = {
    int(PlatformType.BIRD): "Bird",
    int(PlatformType.MULTIROTOR): "Multirotor",
    int(PlatformType.FIXED_WING_UAS): "Fixed-wing UAS",
    int(PlatformType.EVTOL_UAM): "eVTOL UAM",
    int(PlatformType.GA): "General aviation",
    int(PlatformType.DEBRIS): "Debris",
    int(PlatformType.UNKNOWN): "Unknown",
}

#: Human-readable maneuver-mode names, keyed by the raw integer stored on disk.
MODE_NAMES: dict[int, str] = {
    int(ManeuverMode.CV): "CV",
    int(ManeuverMode.CT_LEFT): "CT-left",
    int(ManeuverMode.CT_RIGHT): "CT-right",
    int(ManeuverMode.CA): "CA",
    int(ManeuverMode.HOVER): "Hover",
}

#: Human-readable intent names, keyed by the raw integer stored on disk.
INTENT_NAMES: dict[int, str] = {
    int(IntentType.CORRIDOR_TRANSIT): "Corridor transit",
    int(IntentType.DELIVERY_WP): "Delivery waypoints",
    int(IntentType.LOITER_HOVER): "Loiter / hover",
    int(IntentType.SEARCH_WANDER): "Search wander",
    int(IntentType.REACTIVE_AVOID): "Reactive avoid",
    int(IntentType.PASSIVE_DRIFT): "Passive drift",
}

#: Human-readable encounter-geometry names, keyed by the raw integer.
GEOMETRY_NAMES: dict[int, str] = {
    int(GeometryType.HEAD_ON): "Head-on",
    int(GeometryType.CROSSING): "Crossing",
    int(GeometryType.OVERTAKING): "Overtaking",
    int(GeometryType.BACKGROUND): "Background",
}

#: Human-readable risk-level names, keyed by the raw integer.
RISK_NAMES: dict[int, str] = {
    int(RiskLevel.NONE): "None",
    int(RiskLevel.LOW): "Low",
    int(RiskLevel.MEDIUM): "Medium",
    int(RiskLevel.HIGH): "High",
}
