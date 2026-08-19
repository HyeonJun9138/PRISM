"""Track identity management: ids, ageing, and drop-after-misses.

Doc 3.5 asks for one specific behaviour that a naive implementation would get
wrong: a track that dies and is re-acquired must come back with a **new**
``track_id`` and a reset age, not with its old identity. The reasoning is a
paper argument, not an implementation detail.

Against non-cooperative traffic the ownship has no identifier for the object
it is looking at -- no ICAO address, no Remote ID serial, no callsign. All it
has is a geometric track that its own tracker opened. When that track dies
(the object left the sensor envelope, or Pd failed ``drop_after_misses`` times
in a row), the tracker cannot know that a contact appearing later is the same
physical object. It therefore opens a *new* track. The dataset must reproduce
this discontinuity, because the prediction model has to learn the correct
epistemic response to it: a fresh, young track carries very little history and
must produce a wide predictive distribution, whereas a long-lived track may be
confident. A simulator that quietly kept the same id across a dropout would
train an over-confident predictor -- and doc 1 makes over-confidence the one
failure mode the whole paper exists to eliminate.

MISSING SAMPLES ARE NEVER INTERPOLATED. This rule is stated twice in the
paper spec and is enforced here: on a miss, and on any step with no update at
all, the record carries ``detected = False`` with a NaN position and a NaN
covariance. Nothing in this module ever writes a position it was not given by
:class:`sensors.model.Measurement`. Coasting, smoothing and gap-filling are
deliberately absent; the gap is the signal.

Units: times in seconds, positions in metres, covariances in m^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from scenario.config import SensorConfig
from sensors.model import Measurement

__all__ = ["TrackRecord", "TrackManager"]


def _nan_pos() -> np.ndarray:
    """Return a ``[2]`` float64 array of NaN (no position available)."""
    return np.full((2,), np.nan, dtype=np.float64)


def _nan_cov() -> np.ndarray:
    """Return a ``[2,2]`` float64 array of NaN (no covariance available)."""
    return np.full((2, 2), np.nan, dtype=np.float64)


def _jsonable(v: float) -> Any:
    """Convert a float to a JSON-safe value (NaN / Inf become ``None``)."""
    f = float(v)
    return None if not math.isfinite(f) else f


@dataclass(slots=True)
class TrackRecord:
    """Current tracker view of one physical object.

    ``agent_id`` is bookkeeping internal to the simulator: it links the record
    back to the ground-truth agent so the writer knows which slot to fill. It
    is never exposed to the prediction model, which sees only ``track_id``,
    ``track_age``, ``detected``, ``pos_xy`` and ``R_2x2``.

    Attributes:
        agent_id: ground-truth agent id [-].
        track_id: tracker-assigned identity [-]; ``-1`` when no live track.
            Monotonically increasing over the episode, never reused.
        track_age: number of sensor updates [-] the current track has received
            since it opened; ``0`` on the opening detection and ``0`` when no
            track is live. Missed updates do not age a track -- age counts the
            measurements actually supporting it.
        detected: True only if a fresh detection arrived at THIS update. False
            on a miss, on a step with no scheduled update, and after a drop.
        pos_xy: measured position ``[2]`` [m] in world axes, NaN unless
            ``detected``. Never interpolated.
        R_2x2: measurement covariance ``[2,2]`` [m^2] in world axes, NaN
            unless ``detected``.
        t_valid: time [s] the reported position was true (measurement time
            minus latency); NaN unless ``detected``.
        t_update: simulation time [s] of the most recent detection; NaN before
            the first one.
        misses: consecutive missed updates [-] on the live track.
        n_detections: detections accumulated by the live track [-]; ``0`` when
            no track is live.
    """

    agent_id: int
    track_id: int = -1
    track_age: int = 0
    detected: bool = False
    pos_xy: np.ndarray = field(default_factory=_nan_pos)
    R_2x2: np.ndarray = field(default_factory=_nan_cov)
    t_valid: float = float("nan")
    t_update: float = float("nan")
    misses: int = 0
    n_detections: int = 0

    @property
    def is_live(self) -> bool:
        """True while a track is open (``track_id >= 0``)."""
        return self.track_id >= 0

    def copy(self) -> "TrackRecord":
        """Return an independent copy (arrays are copied, not shared)."""
        return TrackRecord(
            agent_id=int(self.agent_id),
            track_id=int(self.track_id),
            track_age=int(self.track_age),
            detected=bool(self.detected),
            pos_xy=np.array(self.pos_xy, dtype=np.float64, copy=True),
            R_2x2=np.array(self.R_2x2, dtype=np.float64, copy=True),
            t_valid=float(self.t_valid),
            t_update=float(self.t_update),
            misses=int(self.misses),
            n_detections=int(self.n_detections),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (NaN becomes ``None``)."""
        return {
            "agent_id": int(self.agent_id),
            "track_id": int(self.track_id),
            "track_age": int(self.track_age),
            "detected": bool(self.detected),
            "pos_xy": [_jsonable(self.pos_xy[0]), _jsonable(self.pos_xy[1])],
            "R_2x2": [
                [_jsonable(self.R_2x2[i][j]) for j in range(2)] for i in range(2)
            ],
            "t_valid": _jsonable(self.t_valid),
            "t_update": _jsonable(self.t_update),
            "misses": int(self.misses),
            "n_detections": int(self.n_detections),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrackRecord":
        """Rebuild from :meth:`to_dict` output (``None`` becomes NaN)."""

        def _f(v: Any) -> float:
            return float("nan") if v is None else float(v)

        pos = d.get("pos_xy", [None, None])
        cov = d.get("R_2x2", [[None, None], [None, None]])
        return cls(
            agent_id=int(d["agent_id"]),
            track_id=int(d.get("track_id", -1)),
            track_age=int(d.get("track_age", 0)),
            detected=bool(d.get("detected", False)),
            pos_xy=np.array([_f(pos[0]), _f(pos[1])], dtype=np.float64),
            R_2x2=np.array(
                [[_f(cov[0][0]), _f(cov[0][1])], [_f(cov[1][0]), _f(cov[1][1])]],
                dtype=np.float64,
            ),
            t_valid=_f(d.get("t_valid")),
            t_update=_f(d.get("t_update")),
            misses=int(d.get("misses", 0)),
            n_detections=int(d.get("n_detections", 0)),
        )


class TrackManager:
    """Assigns and retires track identities from a measurement stream.

    Lifecycle of one physical object:

        1. First detection opens a track: a fresh, monotonically increasing
           ``track_id`` and ``track_age = 0``.
        2. Every further detection increments ``track_age`` and clears the
           miss counter.
        3. ``drop_after_misses`` consecutive missed updates close the track:
           ``track_id`` returns to ``-1`` and ``track_age`` to ``0``.
        4. A later detection of the same physical object opens a NEW track
           with a NEW id and age ``0`` -- a deliberate discontinuity (see the
           module docstring).

    The manager performs no data association: measurements arrive already
    keyed by ``agent_id`` because the simulator knows the truth. Track
    mis-association is listed as a second-tier extension in doc 3.7 and is
    intentionally absent here.
    """

    def __init__(self, drop_after_misses: int = 6, first_track_id: int = 0) -> None:
        """Create a track manager.

        Args:
            drop_after_misses: number of consecutive missed updates [-] that
                closes a track; must be >= 1.
            first_track_id: id [-] handed to the first track opened.
        """
        if int(drop_after_misses) < 1:
            raise ValueError("drop_after_misses must be >= 1")
        self._drop_after_misses = int(drop_after_misses)
        self._next_track_id = int(first_track_id)
        self._records: dict[int, TrackRecord] = {}
        self._last_t: float = float("nan")

    @classmethod
    def from_config(cls, cfg: SensorConfig, first_track_id: int = 0) -> "TrackManager":
        """Build a manager from a :class:`scenario.config.SensorConfig`."""
        return cls(
            drop_after_misses=int(cfg.drop_after_misses),
            first_track_id=int(first_track_id),
        )

    # -- properties --------------------------------------------------------

    @property
    def drop_after_misses(self) -> int:
        """Consecutive misses [-] that close a track."""
        return self._drop_after_misses

    @property
    def next_track_id(self) -> int:
        """The id [-] that will be handed to the next track opened."""
        return self._next_track_id

    @property
    def last_update_time(self) -> float:
        """Simulation time [s] of the most recent :meth:`update` call."""
        return self._last_t

    # -- main entry point --------------------------------------------------

    def update(
        self,
        t: float,
        measurements: Iterable[Measurement],
    ) -> dict[int, TrackRecord]:
        """Fold one batch of sensor reports into the track table.

        Args:
            t: current simulation time [s].
            measurements: the reports emitted by
                :meth:`sensors.model.SensorModel.measure` at this time. A
                report with ``detected = False`` is a MISS and counts toward
                the drop criterion; a target absent from the batch was simply
                not due for an update and is left untouched.

        Returns:
            ``{agent_id: TrackRecord}`` containing every agent with a live
            track plus every agent named in this batch (so a closure is
            visible to the caller). The records are copies: mutating them
            cannot corrupt the tracker.

            Records with ``detected = False`` carry NaN position and NaN
            covariance. The caller MUST write those NaNs through to
            ``obs_pos`` / ``obs_R`` unchanged -- never interpolate a gap
            (doc 3.5).
        """
        t = float(t)
        touched: set[int] = set()

        # A new update begins: nothing is detected until a measurement says so,
        # and last step's position must not leak into this step.
        for rec in self._records.values():
            rec.detected = False
            rec.pos_xy = _nan_pos()
            rec.R_2x2 = _nan_cov()
            rec.t_valid = float("nan")

        for m in measurements:
            aid = int(m.agent_id)
            touched.add(aid)
            rec = self._records.get(aid)
            if rec is None:
                rec = TrackRecord(agent_id=aid)
                self._records[aid] = rec

            if m.detected:
                if rec.track_id < 0:
                    # Re-acquisition (or first acquisition) -> NEW identity.
                    rec.track_id = self._next_track_id
                    self._next_track_id += 1
                    rec.track_age = 0
                    rec.n_detections = 1
                else:
                    rec.track_age += 1
                    rec.n_detections += 1
                rec.misses = 0
                rec.detected = True
                rec.pos_xy = np.array(m.pos_xy, dtype=np.float64, copy=True)
                rec.R_2x2 = np.array(m.R_2x2, dtype=np.float64, copy=True)
                rec.t_valid = float(m.t_valid)
                rec.t_update = t
            else:
                if rec.track_id >= 0:
                    rec.misses += 1
                    if rec.misses >= self._drop_after_misses:
                        self._close(rec)

        self._last_t = t
        return {
            aid: rec.copy()
            for aid, rec in self._records.items()
            if rec.track_id >= 0 or aid in touched
        }

    # -- inspection --------------------------------------------------------

    def records(self) -> dict[int, TrackRecord]:
        """Copy of the whole internal table, keyed by ``agent_id``."""
        return {aid: rec.copy() for aid, rec in self._records.items()}

    def live_tracks(self) -> dict[int, int]:
        """``{agent_id: track_id}`` for every currently open track."""
        return {
            aid: int(rec.track_id)
            for aid, rec in self._records.items()
            if rec.track_id >= 0
        }

    def forget(self, agent_id: int) -> None:
        """Drop all bookkeeping for one agent (it left the simulation)."""
        self._records.pop(int(agent_id), None)

    def reset(self, first_track_id: int | None = None) -> None:
        """Clear every record; optionally restart the track-id counter.

        Args:
            first_track_id: new starting id [-], or None to keep counting
                from where the previous episode stopped.
        """
        self._records.clear()
        self._last_t = float("nan")
        if first_track_id is not None:
            self._next_track_id = int(first_track_id)

    # -- snapshot / restore -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture the full tracker state for a branch rollout (doc 3.3).

        Returns:
            A JSON-safe dict; agent ids become string keys.
        """
        return {
            "drop_after_misses": int(self._drop_after_misses),
            "next_track_id": int(self._next_track_id),
            "last_t": _jsonable(self._last_t),
            "records": {
                str(aid): rec.to_dict() for aid, rec in self._records.items()
            },
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore the state captured by :meth:`snapshot`.

        Args:
            snap: a mapping produced by :meth:`snapshot`.
        """
        self._drop_after_misses = int(
            snap.get("drop_after_misses", self._drop_after_misses)
        )
        self._next_track_id = int(snap.get("next_track_id", self._next_track_id))
        lt = snap.get("last_t", None)
        self._last_t = float("nan") if lt is None else float(lt)
        self._records = {
            int(aid): TrackRecord.from_dict(d)
            for aid, d in dict(snap.get("records", {})).items()
        }

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _close(rec: TrackRecord) -> None:
        """Retire a track: the identity is gone and will never be reused.

        The kinematic payload is wiped to NaN rather than coasted forward.
        Coasting would be interpolation of a missing observation, which the
        dataset contract forbids (doc 3.5).
        """
        rec.track_id = -1
        rec.track_age = 0
        rec.misses = 0
        rec.n_detections = 0
        rec.detected = False
        rec.pos_xy = _nan_pos()
        rec.R_2x2 = _nan_cov()
        rec.t_valid = float("nan")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TrackManager(live={len(self.live_tracks())}, "
            f"next_id={self._next_track_id}, "
            f"drop_after={self._drop_after_misses})"
        )
