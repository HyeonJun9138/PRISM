"""Per-platform reaction layer -- 논문관련.txt [3.2](4), rev.3.

Every POWERED platform carries a reaction layer: when developing geometry
against another aircraft (or the ownship) projects a loss of separation, the
platform may notice it (Bernoulli, per-platform probability), waits a
reaction delay, then commits to an evasive turn for a latch duration before
relaxing back to its mission.

Why this exists (paper argument, not a code nicety):
  * avoidance is a property of every pilot/autopilot, not a mission of one
    intent class -- so interaction structure exists in the TRAIN split too;
  * "did it notice?" is a Bernoulli draw and "which way?" is a (possibly
    biased) coin: two more stochastic branch points that constant-velocity
    extrapolation can never capture (doc 3.3);
  * regulated aircraft prefer a RIGHT turn (rules of the air, doc 12.5),
    birds pick a side at random -- the asymmetry of the branch probabilities
    is itself a platform signature the predictor's mode weights can learn.

Architecture:  intent -> [reaction layer] -> mode machine -> physics.
The layer is passive until latched; while latched it blends its evasive turn
into the command stream at a per-platform authority.

Determinism contract: `perceive()` draws from the shared 'perception' stream
(notice + delay) and 'branch_choice' stream (turn side); both are in
simulation.rollout.BRANCH_STREAMS, so branch rollouts re-randomise reactions
exactly like every other future decision. State is a plain dict fully
captured by snapshot()/restore() -- an incomplete capture here would silently
corrupt the doc 3.3 calibration ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from core.constants import DWC_HMD_M, DWC_TAU_MOD_S
from core.types import PlatformType

__all__ = [
    "ReactionProfile",
    "REACTION_PROFILES",
    "ReactionLayer",
    "REACTION_STATE_KEY",
    "reaction_profile",
]

#: Key under which the layer's state travels inside ``AgentSnapshot.intent_state``.
REACTION_STATE_KEY: str = "_reaction"


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReactionProfile:
    """How one platform class notices and evades a conflict.

    Attributes:
        detect_range_m: beyond this range the threat is invisible [m].
        notice_prob: P(notice) per qualifying scan while idle [-]. The scan
            runs at 2 Hz, so the effective per-second probability is higher.
        delay_mean_s / delay_jitter_s: reaction delay after noticing [s].
        latch_s: how long the evasive turn is held [s].
        turn_frac: evasive turn rate as a fraction of max_turn_rate(v) [-].
        right_bias: P(evade to the RIGHT) [-]. 0.5 = symmetric (bird);
            regulated aircraft sit near 0.8 (rules of the air, doc 12.5).
        wrong_way_prob: P(the chosen side is flipped TOWARD the threat) [-].
            Zero for everything except UNKNOWN -- its OOD scare factor.
        authority: blend weight of the evasive command while latched [-].
        cooldown_s: refractory period after the latch releases [s].
        cpa_threshold_m: projected miss distance that qualifies as a
            conflict [m].
        tcpa_max_s: only conflicts developing within this horizon count [s].
        decel_frac: fraction of a_min commanded while latched [-] (a small
            defensive deceleration; 0 for platforms that evade by turn only).
    """

    detect_range_m: float
    notice_prob: float
    delay_mean_s: float
    delay_jitter_s: float
    latch_s: float
    turn_frac: float
    right_bias: float
    authority: float
    cooldown_s: float
    cpa_threshold_m: float
    tcpa_max_s: float
    wrong_way_prob: float = 0.0
    decel_frac: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}

    def validate(self) -> list[str]:
        problems = []
        for name, lo, hi in (
            ("notice_prob", 0.0, 1.0), ("right_bias", 0.0, 1.0),
            ("wrong_way_prob", 0.0, 1.0), ("authority", 0.0, 1.0),
            ("turn_frac", 0.0, 1.0), ("decel_frac", 0.0, 1.0),
        ):
            v = float(getattr(self, name))
            if not lo <= v <= hi:
                problems.append(f"{name}={v} outside [{lo}, {hi}]")
        for name in ("detect_range_m", "latch_s", "cpa_threshold_m", "tcpa_max_s"):
            if float(getattr(self, name)) <= 0:
                problems.append(f"{name} must be positive")
        return problems


#: Doc [3.2](4) table, rev.3. DEBRIS carries None: an unpowered body cannot
#: react. Values marked 'assumed' pending the doc 12.5 citations
#: (see-and-avoid visual-acquisition studies, ICAO Annex 2 / 14 CFR 91.113).
REACTION_PROFILES: dict[PlatformType, ReactionProfile | None] = {
    # ~0.95 (certified DAA sensor), ~1 s, gentle RIGHT-biased turn that keeps
    # the corridor reachable; passenger-comfort turn fraction.
    PlatformType.EVTOL_UAM: ReactionProfile(
        detect_range_m=2500.0, notice_prob=0.95, delay_mean_s=1.0,
        delay_jitter_s=0.2, latch_s=6.0, turn_frac=0.35, right_bias=0.85,
        authority=0.65, cooldown_s=8.0, cpa_threshold_m=500.0,
        tcpa_max_s=DWC_TAU_MOD_S, decel_frac=0.15),
    # autopilot-grade DAA, slower to commit, mild right preference.
    PlatformType.FIXED_WING_UAS: ReactionProfile(
        detect_range_m=1500.0, notice_prob=0.55, delay_mean_s=1.5,
        delay_jitter_s=0.4, latch_s=5.0, turn_frac=0.5, right_bias=0.75,
        authority=0.7, cooldown_s=6.0, cpa_threshold_m=350.0,
        tcpa_max_s=25.0),
    # operator on a screen: misses a lot, then side-steps decisively.
    PlatformType.MULTIROTOR: ReactionProfile(
        detect_range_m=800.0, notice_prob=0.65, delay_mean_s=1.0,
        delay_jitter_s=0.4, latch_s=4.0, turn_frac=0.6, right_bias=0.65,
        authority=0.85, cooldown_s=4.0, cpa_threshold_m=200.0,
        tcpa_max_s=20.0, decel_frac=0.3),
    # see-and-avoid only; poor small-target acquisition (doc 12.5), late,
    # then a regulation right turn.
    PlatformType.GA: ReactionProfile(
        detect_range_m=2000.0, notice_prob=0.35, delay_mean_s=2.0,
        delay_jitter_s=0.6, latch_s=6.0, turn_frac=0.4, right_bias=0.8,
        authority=0.7, cooldown_s=8.0, cpa_threshold_m=450.0,
        tcpa_max_s=30.0),
    # last-second, violent, side chosen by coin -- the classic bird escape.
    PlatformType.BIRD: ReactionProfile(
        detect_range_m=300.0, notice_prob=0.9, delay_mean_s=0.3,
        delay_jitter_s=0.1, latch_s=2.5, turn_frac=0.9, right_bias=0.5,
        authority=0.95, cooldown_s=2.0, cpa_threshold_m=100.0,
        tcpa_max_s=8.0),
    # erratic by design (the OOD class): unpredictable notice, occasional
    # WRONG-WAY turns.
    PlatformType.UNKNOWN: ReactionProfile(
        detect_range_m=1200.0, notice_prob=0.5, delay_mean_s=0.8,
        delay_jitter_s=0.7, latch_s=3.0, turn_frac=0.8, right_bias=0.5,
        authority=0.9, cooldown_s=3.0, cpa_threshold_m=300.0,
        tcpa_max_s=15.0, wrong_way_prob=0.15),
    PlatformType.DEBRIS: None,
}


def reaction_profile(platform: PlatformType | int) -> ReactionProfile | None:
    """Profile for a platform; None when the platform cannot react."""
    return REACTION_PROFILES.get(PlatformType(int(platform)))


# import-time self check: every profile must be internally consistent
for _p, _prof in REACTION_PROFILES.items():
    if _prof is not None:
        _problems = _prof.validate()
        assert not _problems, f"REACTION_PROFILES[{_p.name}]: {_problems}"


# --------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------

_IDLE, _DELAY, _LATCHED, _COOLDOWN = "idle", "delay", "latched", "cooldown"


class ReactionLayer:
    """Per-agent evasion state machine: idle -> delay -> latched -> cooldown.

    The ENGINE drives :meth:`perceive` at the logging cadence (2 Hz) with a
    threat table; :meth:`tick`/:meth:`apply` run every integration step so
    the latched turn is smooth at 20 Hz. This keeps the O(N^2) geometry scan
    off the 20 Hz path.

    Args:
        profile: the platform's :class:`ReactionProfile`.
        exclude_ownship: True for REACTIVE_AVOID agents, whose intent already
            evades the ownship -- the layer then only reacts to OTHER
            traffic, so the two mechanisms never double-command a turn.
    """

    __slots__ = ("profile", "exclude_ownship", "state", "events")

    def __init__(self, profile: ReactionProfile,
                 exclude_ownship: bool = False) -> None:
        self.profile = profile
        self.exclude_ownship = bool(exclude_ownship)
        self.events = 0     # latches fired over the layer's lifetime [-]
        self.state: dict[str, Any] = {
            "phase": _IDLE, "timer": 0.0, "sign": 0.0, "threat_id": -1,
        }

    # ------------------------------------------------------------ perception
    def perceive(
        self,
        agent: Any,
        threats: Sequence[tuple[int, float, float, float, float]],
        rng_perception: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> bool:
        """Scan the threat table; possibly arm a reaction. Returns True when
        a new reaction was armed on this call.

        Args:
            agent: the reacting agent (position/velocity read from state).
            threats: ``(id, x, y, vx, vy)`` of every OTHER live object, the
                ownship carried with id ``-1``.
            rng_perception: the shared 'perception' stream (notice + delay).
            rng_branch: the shared 'branch_choice' stream (turn side).
        """
        if self.state["phase"] != _IDLE:
            return False
        p = self.profile
        ax, ay = agent.state.x, agent.state.y
        avx = agent.state.v * math.cos(agent.state.psi)
        avy = agent.state.v * math.sin(agent.state.psi)

        worst: tuple[float, int, float, float] | None = None  # (tcpa, id, rx, ry)
        for tid, tx, ty, tvx, tvy in threats:
            if tid == agent.agent_id:
                continue
            if self.exclude_ownship and tid == -1:
                continue
            rx, ry = tx - ax, ty - ay
            r = math.hypot(rx, ry)
            if r > p.detect_range_m or r < 1e-6:
                continue
            vx_rel, vy_rel = tvx - avx, tvy - avy
            v2 = vx_rel * vx_rel + vy_rel * vy_rel
            if v2 < 1e-9:
                continue
            tcpa = -(rx * vx_rel + ry * vy_rel) / v2
            if not (0.0 < tcpa <= p.tcpa_max_s):
                continue
            cx, cy = rx + vx_rel * tcpa, ry + vy_rel * tcpa
            cpa = math.hypot(cx, cy)
            if cpa >= p.cpa_threshold_m:
                continue
            if worst is None or tcpa < worst[0]:
                worst = (tcpa, tid, rx, ry)

        if worst is None:
            return False
        # one Bernoulli per 2 Hz scan while a conflict is developing
        if float(rng_perception.random()) >= p.notice_prob:
            return False

        # side pick: right_bias toward a RIGHT turn (negative omega, CCW+
        # convention); UNKNOWN may flip toward the threat (wrong way).
        sign = -1.0 if float(rng_branch.random()) < p.right_bias else 1.0
        if p.wrong_way_prob > 0.0 and float(rng_branch.random()) < p.wrong_way_prob:
            sign = -sign
        delay = max(0.05, p.delay_mean_s
                    + p.delay_jitter_s * float(rng_perception.standard_normal()))
        self.state.update(phase=_DELAY, timer=delay, sign=sign,
                          threat_id=int(worst[1]))
        return True

    # --------------------------------------------------------------- driving
    def tick(self, dt: float) -> None:
        """Advance the phase timers by one integration step [s]."""
        ph = self.state["phase"]
        if ph == _IDLE:
            return
        self.state["timer"] = float(self.state["timer"]) - float(dt)
        if self.state["timer"] > 0.0:
            return
        if ph == _DELAY:
            self.state.update(phase=_LATCHED, timer=self.profile.latch_s)
            self.events += 1
        elif ph == _LATCHED:
            self.state.update(phase=_COOLDOWN, timer=self.profile.cooldown_s)
        else:  # cooldown expired
            self.state.update(phase=_IDLE, timer=0.0, sign=0.0, threat_id=-1)

    def apply(self, omega_cmd: float, a_cmd: float, agent: Any) -> tuple[float, float]:
        """Blend the evasive command into the stream while latched."""
        if self.state["phase"] != _LATCHED:
            return omega_cmd, a_cmd
        p = self.profile
        cap = agent.envelope.max_turn_rate(agent.state.v)
        omega_evade = float(self.state["sign"]) * p.turn_frac * cap
        omega = (1.0 - p.authority) * omega_cmd + p.authority * omega_evade
        a_out = a_cmd
        if p.decel_frac > 0.0:
            a_evade = p.decel_frac * agent.envelope.a_min   # a_min is negative
            a_out = (1.0 - p.authority) * a_cmd + p.authority * a_evade
        return omega, a_out

    @property
    def is_latched(self) -> bool:
        return self.state["phase"] == _LATCHED

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> dict[str, Any]:
        return {"phase": str(self.state["phase"]),
                "timer": float(self.state["timer"]),
                "sign": float(self.state["sign"]),
                "threat_id": int(self.state["threat_id"]),
                "events": int(self.events),
                "exclude_ownship": bool(self.exclude_ownship)}

    def restore(self, snap: Mapping[str, Any]) -> None:
        self.state = {
            "phase": str(snap.get("phase", _IDLE)),
            "timer": float(snap.get("timer", 0.0)),
            "sign": float(snap.get("sign", 0.0)),
            "threat_id": int(snap.get("threat_id", -1)),
        }
        self.events = int(snap.get("events", 0))
        self.exclude_ownship = bool(snap.get("exclude_ownship", self.exclude_ownship))
