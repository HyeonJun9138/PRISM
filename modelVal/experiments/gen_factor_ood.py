"""Factor-wise OOD dataset generation (reviewer 2, comment III).

The submitted paper's ``test_ood`` split moves FOUR parameter axes at once
relative to ``train`` -- traffic-load, wind band, platform support and intent
support -- so a coverage drop cannot be attributed to any one of them.  This
script builds one dataset run per axis, each identical to ``train`` except for
that single axis, plus an ``all`` run that reproduces the paper's combination
on the same seed block.  The trained model is then evaluated on each without
retraining, which turns one aggregate OOD number into a per-axis breakdown.

What this script is careful about
---------------------------------
* ``dataGen/`` is FROZEN (CLAUDE.md rule 6 / the reproducibility fingerprint).
  Nothing here edits it; the runs are assembled from the public
  ``ScenarioConfig`` API exactly the way ``main.py generate`` does.
* Seeds come from a block ABOVE the four canonical splits
  (``4 * SEED_BLOCK_SIZE`` and up, one sub-block per factor), so no episode
  here can collide with train / valid / test_id / test_ood.  Each factor gets
  its own sub-block, so factor runs do not collide with each other either.
* Per-episode density is drawn from the factor's own band with a throwaway
  RNG hub seeded by the master seed, mirroring ``config_for_episode`` so the
  draw depends on nothing but (factor, index).
* The reviewer's fifth phrase, "avoidance parameters", is NOT a fifth axis in
  the generator.  Reaction behaviour is a per-platform property
  (``agents.reaction.ReactionProfile``) plus the OOD-only ``REACTIVE_AVOID``
  intent, so it enters through the platform and intent axes.  The ``platform``
  and ``intent`` runs therefore carry it, and the answer to the reviewer says
  so explicitly rather than inventing an axis the simulator does not have.

Usage
-----
    python -m experiments.gen_factor_ood --factor density --episodes 300
    python -m experiments.gen_factor_ood --factor all --episodes 300
    python -m experiments.gen_factor_ood --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

from core.rng import RNGHub  # noqa: E402
from scenario.config import (EncounterConfig, IntentMixConfig,  # noqa: E402
                             PlatformMixConfig, WindConfig)
from scenario.presets import (SEED_BLOCK_SIZE, make_preset,  # noqa: E402
                              split_spec)
from simulation.engine import SimulationEngine  # noqa: E402
from storage.writer import DatasetWriter  # noqa: E402


#: One entry per factor.  ``seed_sub`` indexes a private sub-block starting at
#: ``4 * SEED_BLOCK_SIZE``; ``fields`` names the ``SplitSpec`` attributes the
#: factor copies from ``test_ood`` instead of ``train``.
FACTORS: dict[str, dict] = {
    "density": {
        "seed_sub": 0,
        "fields": ("density_min", "density_max", "n_agents_min",
                   "n_agents_max", "n_designed_max"),
        "description": "traffic-load axis only: test_ood load, train wind / "
                       "platforms / intents",
    },
    "wind": {
        "seed_sub": 1,
        "fields": ("wind_min_ms", "wind_max_ms"),
        "description": "wind axis only: 10-16 m/s band, train load / "
                       "platforms / intents",
    },
    "platform": {
        "seed_sub": 2,
        "fields": ("platform_weights",),
        "description": "platform-support axis only: bird / GA / debris / "
                       "unknown added (their ReactionProfiles come with "
                       "them), train load / wind / intents",
    },
    "intent": {
        "seed_sub": 3,
        "fields": ("intent_weights",),
        "description": "intent-support axis only: search-wander and "
                       "reactive-avoid added, train load / wind / platforms",
    },
    "all": {
        "seed_sub": 4,
        "fields": ("density_min", "density_max", "n_agents_min",
                   "n_agents_max", "n_designed_max", "wind_min_ms",
                   "wind_max_ms", "platform_weights", "intent_weights"),
        "description": "every axis at once -- the paper's test_ood "
                       "combination, regenerated on this seed block so it is "
                       "comparable to the single-axis runs",
    },
    "control": {
        "seed_sub": 5,
        "fields": (),
        "description": "no axis moved -- an in-distribution control on the "
                       "same fresh seed block, so 'unseen seeds' is ruled "
                       "out as an explanation for any factor's drop",
    },
}


def factor_spec(factor: str):
    """Return a ``SplitSpec`` for ``factor``: train with one axis replaced."""
    if factor not in FACTORS:
        raise KeyError(f"unknown factor {factor!r}; "
                       f"choose from {sorted(FACTORS)}")
    from dataclasses import replace

    train = split_spec("train")
    ood = split_spec("test_ood")
    meta = FACTORS[factor]
    over = {f: getattr(ood, f) for f in meta["fields"]}
    base = 4 * SEED_BLOCK_SIZE + meta["seed_sub"] * SEED_BLOCK_SIZE
    return replace(train, name=f"ood_{factor}", seed_start=base,
                   seed_end=base + SEED_BLOCK_SIZE,
                   description=meta["description"], **over)


def config_for(factor: str, index: int):
    """Build the ``ScenarioConfig`` of episode ``index`` of ``factor``.

    Mirrors ``scenario.presets.config_for_episode``: the density is drawn on a
    throwaway RNG hub seeded by the master seed, so the draw is a pure
    function of (factor, index) and the simulator's own scenario stream starts
    unconsumed.
    """
    spec = factor_spec(factor)
    seed = int(spec.seed_start) + int(index)
    lo, hi = spec.density_band()
    density = (float(RNGHub(seed).stream("scenario").uniform(lo, hi))
               if hi > lo else float(lo))
    cfg = make_preset("train", seed=seed)
    return cfg.copy_with(
        name=f"ood_{factor}/{int(index):06d}",
        seed=seed,
        density_per_km2=density,
        n_agents_min=int(spec.n_agents_min),
        n_agents_max=int(spec.n_agents_max),
        platform_mix=PlatformMixConfig(weights=dict(spec.platform_weights)),
        intent_mix=IntentMixConfig(weights=dict(spec.intent_weights)),
        encounter=EncounterConfig(n_designed_max=int(spec.n_designed_max)),
        wind=WindConfig(speed_min_ms=float(spec.wind_min_ms),
                        speed_max_ms=float(spec.wind_max_ms),
                        dir_min_rad=-3.141592653589793,
                        dir_max_rad=3.141592653589793),
    )


def generate(factor: str, n: int, tag: str, rollouts: int,
             branch_t: float) -> str:
    """Generate ``n`` episodes of ``factor`` and return the run directory."""
    spec = factor_spec(factor)
    writer = DatasetWriter()
    run_dir = writer.open_run(config_for(factor, 0),
                              preset=f"ood_{factor}", tag=tag)
    print(f"factor    : {factor}  ({spec.description})")
    print(f"run       : {run_dir}")
    print(f"seed block: [{spec.seed_start}, {spec.seed_end})   episodes: {n}")
    nmac = 0
    try:
        for i in range(n):
            cfg = config_for(factor, i)
            engine = SimulationEngine(cfg, episode_index=i)
            ep, meta = engine.run_episode()
            writer.write_episode(ep, meta)
            nmac += int(bool(meta.nmac_occurred))
            if rollouts:
                _rollouts(writer, engine, i, rollouts, branch_t)
            if (i + 1) % 25 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}  nmac {nmac}", flush=True)
        writer.add_split(
            f"ood_{factor}", n_episodes=n, seed_base=int(spec.seed_start),
            seed_block=[int(spec.seed_start), int(spec.seed_end)],
            episode_range=[0, n - 1],
            notes=(f"factor-wise OOD run for reviewer 2 comment III; "
                   f"axis={factor}; {spec.description}"))
    except BaseException:
        writer.close(status="aborted")
        raise
    writer.close()
    print(f"done: {n} episodes ({nmac} with an NMAC) -> {run_dir}")
    return run_dir


def _rollouts(writer, engine, index: int, k: int, branch_t: float) -> int:
    """Delegate to dataGen's own rollout writer so the format stays identical."""
    from main import _write_branch_rollouts  # dataGen/main.py, on sys.path
    return _write_branch_rollouts(writer, engine, index, k, float(branch_t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", choices=sorted(FACTORS))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--tag", default="factor")
    ap.add_argument("--rollouts", type=int, default=0)
    ap.add_argument("--branch-t", type=float, default=30.0)
    ap.add_argument("--list", action="store_true",
                    help="print each factor's axes and seed block, generate "
                         "nothing")
    args = ap.parse_args()

    if args.list:
        train = split_spec("train")
        for name in sorted(FACTORS):
            s = factor_spec(name)
            print(f"\n{name}: {FACTORS[name]['description']}")
            print(f"  seeds     [{s.seed_start}, {s.seed_end})")
            print(f"  density   {s.density_band()}  (train "
                  f"{train.density_band()})")
            print(f"  wind      {s.wind_band()}  (train {train.wind_band()})")
            print(f"  platforms {sorted(s.platform_support())}  (train "
                  f"{sorted(train.platform_support())})")
            print(f"  intents   {sorted(s.intent_support())}  (train "
                  f"{sorted(train.intent_support())})")
            print(f"  agents    {s.agent_count_band()}")
        return

    if not args.factor:
        ap.error("--factor is required unless --list is given")
    generate(args.factor, int(args.episodes), args.tag, int(args.rollouts),
             float(args.branch_t))


if __name__ == "__main__":
    main()
