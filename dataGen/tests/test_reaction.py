# -*- coding: utf-8 -*-
"""Tests for the rev.3 reaction layer (agents/reaction.py).

What is protected here:
  1. profile table sanity (DEBRIS cannot react, probabilities in range);
  2. a forced conflict geometry actually arms and latches a reaction;
  3. the right-turn bias is statistically real (regulated aircraft) and
     symmetric for birds -- the asymmetry the predictor's mode weights learn;
  4. same seed => bit-identical episodes WITH reactions enabled;
  5. reseeding the 'perception' stream changes reaction outcomes while the
     situation stays identical (the doc 3.3 branch-rollout requirement).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.reaction import (REACTION_PROFILES, ReactionLayer,
                             reaction_profile)
from core.state import AgentState, PerformanceEnvelope
from core.types import PlatformType

PASS = 0
FAIL = 0


def check(cond: bool, label: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


class _StubAgent:
    """Minimal agent double: state + envelope + id."""

    def __init__(self, x, y, v, psi, agent_id=7):
        self.agent_id = agent_id
        self.state = AgentState(x=x, y=y, v=v, psi=psi)
        from agents.platforms import make_envelope
        self.envelope = make_envelope(PlatformType.EVTOL_UAM,
                                      np.random.default_rng(0))


def test_profiles() -> None:
    check(REACTION_PROFILES[PlatformType.DEBRIS] is None, "debris must not react")
    for p, prof in REACTION_PROFILES.items():
        if prof is None:
            continue
        check(not prof.validate(), f"{p.name} profile valid")
    check(reaction_profile(PlatformType.BIRD).right_bias == 0.5,
          "bird side pick is symmetric")
    check(reaction_profile(PlatformType.EVTOL_UAM).right_bias > 0.7,
          "eVTOL prefers the right turn")
    check(reaction_profile(PlatformType.UNKNOWN).wrong_way_prob > 0,
          "UNKNOWN sometimes turns the wrong way")


def _head_on_threat(agent, rng_m: float = 800.0
                    ) -> list[tuple[int, float, float, float, float]]:
    """A target dead ahead at ``rng_m``, closing head-on: guaranteed conflict."""
    x = agent.state.x + rng_m
    return [(99, x, agent.state.y, -40.0, 0.0)]


def test_trigger_and_latch() -> None:
    prof = reaction_profile(PlatformType.EVTOL_UAM)
    agent = _StubAgent(0.0, 0.0, 55.0, 0.0)
    layer = ReactionLayer(prof)
    rng_p = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)

    armed = False
    for _ in range(20):     # 10 s of 2 Hz scans; notice_prob 0.95 -> certain
        if layer.perceive(agent, _head_on_threat(agent), rng_p, rng_b):
            armed = True
            break
    check(armed, "head-on conflict arms the eVTOL layer")

    # run the delay out -> must latch, and the blend must actually turn
    for _ in range(400):    # 20 s of 0.05 s ticks
        layer.tick(0.05)
        if layer.is_latched:
            break
    check(layer.is_latched, "delay expires into a latch")
    om, a = layer.apply(0.0, 0.0, agent)
    check(abs(om) > 1e-4, "latched layer commands a turn")
    check(a <= 0.0, "eVTOL latch commands a defensive deceleration")

    # non-conflicting geometry must NOT arm: target moving away
    layer2 = ReactionLayer(prof)
    away = [(98, agent.state.x + 800.0, 0.0, +40.0, 0.0)]
    armed2 = any(layer2.perceive(agent, away, rng_p, rng_b) for _ in range(20))
    check(not armed2, "receding target never arms the layer")


def test_right_bias_statistics() -> None:
    """Measured side choice must match the profile's bias."""
    for ptype, tol in ((PlatformType.EVTOL_UAM, 0.06), (PlatformType.BIRD, 0.06)):
        prof = reaction_profile(ptype)
        rng_p = np.random.default_rng(11)
        rng_b = np.random.default_rng(12)
        rights = 0
        n = 0
        agent = _StubAgent(0.0, 0.0, 40.0, 0.0)
        # keep the threat inside the platform's own detection range: a bird
        # only sees ~300 m (that is the point of its profile)
        rng_m = min(prof.detect_range_m * 0.8, 800.0)
        for _ in range(4000):
            layer = ReactionLayer(prof)
            if layer.perceive(agent, _head_on_threat(agent, rng_m), rng_p, rng_b):
                n += 1
                if layer.state["sign"] < 0:      # negative omega = right turn
                    rights += 1
        frac = rights / max(1, n)
        expected = prof.right_bias * (1 - prof.wrong_way_prob) \
            + (1 - prof.right_bias) * prof.wrong_way_prob
        check(n > 500, f"{ptype.name}: enough arms to measure ({n})")
        check(abs(frac - expected) < tol,
              f"{ptype.name}: right fraction {frac:.3f} ~ {expected:.3f}")


def test_episode_determinism() -> None:
    """Same seed => bit-identical arrays, reactions and all."""
    from scenario.config import load_scenario_config
    from simulation.episode import run_episode
    cfg = load_scenario_config(
        os.path.join(os.path.dirname(__file__), "..", "config", "presets", "train.json"))
    a1, m1 = run_episode(cfg, seed=4242)
    a2, m2 = run_episode(cfg, seed=4242)
    same = all(np.array_equal(getattr(a1, k), getattr(a2, k), equal_nan=True)
               for k in ("gt_pos", "gt_psi", "gt_mode", "obs_pos", "obs_detected"))
    check(same, "same-seed episodes bit-identical with reactions on")
    check(m1.reaction_events == m2.reaction_events,
          f"reaction_events deterministic ({m1.reaction_events})")
    check(m1.reaction_events > 0, "dense preset produces reaction events")


def test_perception_branching() -> None:
    """Reseeding 'perception' (+ the other branch streams) must change the
    future while the past stays identical -- doc 3.3 for reactions."""
    from scenario.config import load_scenario_config
    from simulation.engine import SimulationEngine
    from simulation.rollout import BRANCH_STREAMS
    cfg = load_scenario_config(
        os.path.join(os.path.dirname(__file__), "..", "config", "presets", "train.json"))
    check("perception" in BRANCH_STREAMS, "'perception' is reseeded per branch")

    eng = SimulationEngine(cfg.copy_with(seed=777))
    for _ in range(600):        # 30 s in
        eng.step(log=False)
    snap = eng.snapshot()

    outcomes = []
    for branch_seed in (1, 2, 3, 4, 5, 6):
        eng.restore(snap)
        eng.reseed_branch(branch_seed, BRANCH_STREAMS)
        for _ in range(300):    # 15 s forward
            eng.step(log=False)
        pos = np.array([[a.state.x, a.state.y] for a in eng.scenario.agents])
        outcomes.append(pos)
    spreads = [float(np.max(np.abs(outcomes[i] - outcomes[0])))
               for i in range(1, len(outcomes))]
    check(max(spreads) > 5.0,
          f"branch futures diverge with perception reseeded (max {max(spreads):.1f} m)")


def main() -> int:
    for fn in (test_profiles, test_trigger_and_latch, test_right_bias_statistics,
               test_episode_determinism, test_perception_branching):
        fn()
    print(f"test_reaction: {PASS} checks passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
