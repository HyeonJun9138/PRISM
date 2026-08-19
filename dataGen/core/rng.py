"""Deterministic, named random-number streams.

Reproducibility is a paper requirement (doc 10: "each stage's output is stored
together with its seed and configuration"). Two properties matter:

1. **Cross-process stability.** The mapping from a stream name to its seed is
   built with :func:`zlib.crc32`, never Python's builtin ``hash()``, which is
   salted per process (PYTHONHASHSEED) and would silently break replay.

2. **Stream independence.** Each named stream is spawned from the master seed
   with its own ``spawn_key``, so adding, removing or re-ordering draws in one
   subsystem cannot perturb another. This is what makes branch rollouts
   (doc 3.3) correct: :meth:`RNGHub.reseed_branch` re-randomises ONLY the
   decision streams while wind, sensor noise and placement replay identically.
"""

from __future__ import annotations

import zlib
from typing import Any, Iterable

import numpy as np

__all__ = ["stable_hash", "STREAM_NAMES", "BRANCH_STREAMS", "RNGHub"]


#: Canonical stream names. Every module MUST use one of these exact strings.
STREAM_NAMES: tuple[str, ...] = (
    "scenario",       # episode-level parameter sampling
    "placement",      # initial positions / headings of background traffic
    "platform",       # platform-type draws from the mix
    "intent",         # intent-type draws from the mix
    "mode_decision",  # stochastic maneuver-mode switching (doc 3.3)
    "branch_choice",  # discrete left/right style branch picks (doc 3.3)
    "wind",           # steady wind and gust process
    "sensor_noise",   # range / bearing measurement noise
    "sensor_detect",  # detection probability draws (Pd)
    "sensor_latency", # latency jitter draws
    "perception",     # reaction-layer notice + delay draws (doc 3.2(4), rev.3)
)

#: Streams re-randomised per branch rollout; everything else replays exactly.
BRANCH_STREAMS: tuple[str, ...] = ("mode_decision", "branch_choice")


def stable_hash(name: str) -> int:
    """Process-stable 32-bit hash of a stream name [-].

    Uses CRC-32 of the UTF-8 encoding. Deterministic across processes,
    machines and Python versions, unlike the builtin ``hash()``.

    Args:
        name: stream name.

    Returns:
        Unsigned 32-bit integer suitable for a ``SeedSequence`` spawn key.
    """
    return int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


class RNGHub:
    """Container of lazily created, deterministically seeded generators.

    A generator for stream ``name`` is seeded from
    ``np.random.SeedSequence(master_seed, spawn_key=(stable_hash(name),))``,
    so ``(master_seed, name)`` fully determines the stream.
    """

    def __init__(self, master_seed: int) -> None:
        """Create a hub.

        Args:
            master_seed: master seed [-]. Non-negative integer; it is recorded
                in the episode metadata so any episode can be replayed.
        """
        self._master_seed: int = int(master_seed)
        self._streams: dict[str, np.random.Generator] = {}

    # -- properties -------------------------------------------------------

    @property
    def master_seed(self) -> int:
        """The master seed [-] this hub was constructed with."""
        return self._master_seed

    @property
    def live_streams(self) -> tuple[str, ...]:
        """Names of all streams instantiated so far, in creation order."""
        return tuple(self._streams.keys())

    # -- core API ---------------------------------------------------------

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator for ``name``, creating it on first use.

        Args:
            name: stream name; should be one of :data:`STREAM_NAMES`.

        Returns:
            The persistent ``numpy`` Generator bound to ``(master_seed, name)``.
        """
        gen = self._streams.get(name)
        if gen is None:
            gen = self._make(self._master_seed, name)
            self._streams[name] = gen
        return gen

    def snapshot(self) -> dict[str, Any]:
        """Capture the bit-generator state of every live stream.

        Returns:
            ``{"master_seed": int, "streams": {name: bit_generator.state}}``.
            The returned structure is a deep-enough copy that later draws do
            not mutate it.
        """
        return {
            "master_seed": int(self._master_seed),
            "streams": {
                name: _copy_state(gen.bit_generator.state)
                for name, gen in self._streams.items()
            },
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore every stream from a :meth:`snapshot` result.

        Streams present in the snapshot but not yet instantiated are created;
        streams live in this hub but absent from the snapshot are dropped, so
        that the hub ends up exactly as it was when the snapshot was taken.

        Args:
            snap: a dict produced by :meth:`snapshot`.

        Raises:
            KeyError: if ``snap`` does not contain a ``"streams"`` entry.
        """
        states = snap["streams"]
        if "master_seed" in snap:
            self._master_seed = int(snap["master_seed"])
        new: dict[str, np.random.Generator] = {}
        for name, state in states.items():
            gen = self._streams.get(name)
            if gen is None:
                gen = self._make(self._master_seed, name)
            gen.bit_generator.state = _copy_state(state)
            new[name] = gen
        self._streams = new

    def reseed_branch(self, branch_seed: int, names: Iterable[str]) -> None:
        """Re-seed only the named streams; leave every other stream untouched.

        This is the mechanism behind branch rollouts (doc 3.3): the same
        initial condition and the same past trajectory are replayed N times
        with a different ``branch_seed``, so the empirical spread of the
        futures is caused purely by the agents' stochastic decision rules --
        not by different wind, placement or sensor noise.

        Args:
            branch_seed: seed [-] identifying this branch rollout.
            names: stream names to re-randomise, typically
                :data:`BRANCH_STREAMS`.
        """
        bs = int(branch_seed)
        for name in names:
            self._streams[name] = self._make(bs, name)

    # -- convenience ------------------------------------------------------

    def spawn_child(self, name: str, child_index: int) -> np.random.Generator:
        """Create an independent generator derived from a stream name.

        Useful for per-agent private streams: ``spawn_child("mode_decision",
        agent_id)`` gives agent ``agent_id`` its own reproducible generator
        that does not consume draws from the shared stream.

        Args:
            name: base stream name.
            child_index: integer discriminator [-], e.g. an agent id.

        Returns:
            A fresh ``numpy`` Generator (not stored in the hub).
        """
        seq = np.random.SeedSequence(
            self._master_seed,
            spawn_key=(stable_hash(name), int(child_index) & 0xFFFFFFFF),
        )
        return np.random.default_rng(seq)

    def reset(self) -> None:
        """Drop every live stream; subsequent calls re-create them freshly."""
        self._streams.clear()

    # -- internals --------------------------------------------------------

    @staticmethod
    def _make(seed: int, name: str) -> np.random.Generator:
        """Build the generator for ``(seed, name)``."""
        seq = np.random.SeedSequence(int(seed), spawn_key=(stable_hash(name),))
        return np.random.default_rng(seq)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"RNGHub(master_seed={self._master_seed}, "
                f"live={list(self._streams)})")


def _copy_state(state: Any) -> Any:
    """Recursively copy a bit-generator state dict (dicts / arrays / scalars)."""
    if isinstance(state, dict):
        return {k: _copy_state(v) for k, v in state.items()}
    if isinstance(state, np.ndarray):
        return state.copy()
    if isinstance(state, (list, tuple)):
        return type(state)(_copy_state(v) for v in state)
    return state
