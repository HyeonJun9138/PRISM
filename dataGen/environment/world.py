"""Static airspace context: region bounds, cruise corridors and no-fly zones.

The world is a SQUARE region centred on the origin of the local tangent plane,
with side ``sqrt(area_km2) * 1000`` [m] (doc 3.7 minimum configuration). It
carries only the static context that doc 5.1 lists as a network input --
corridor boundaries and prohibited zones -- plus the boundary policy that keeps
traffic density stationary over an episode.

IMPORTANT (doc 3.1 / 3.3): nothing in this module places a "branch point" on
the map. Corridors and no-fly zones are *context*; multimodality must come from
the agents' stochastic decision rules in :mod:`agents.behavior`. A corridor is
a soft guidance region an agent may follow, not a rail it is snapped to.

BOUNDARY POLICY (one choice, documented as required)
----------------------------------------------------
Agents are NEVER killed at the boundary and are NEVER teleported (no toroidal
wrap): both would create discontinuities in the logged tracks, and a kill would
let the traffic density decay through the episode, which would confound the
density axis of the dataset split (doc 8.1). Instead a two-layer scheme is used:

1. **Soft turn-back (the normal case).** Inside a margin band of width
   ``margin_m`` along each wall, :meth:`World.containment_heading` returns a
   desired heading that blends the agent's current heading toward the inward
   wall normal, with a blend weight that grows linearly from 0 at the inner
   edge of the band to 1 at the wall. The agent then turns toward it at its own
   rate-limited turn rate, so the manoeuvre is physically legal and gentle.

2. **Hard reflection (the fallback).** If an agent still crosses the wall --
   a fast, sluggishly turning platform pushed out by wind can --
   :meth:`World.enforce_bounds` clamps the position back onto the wall and
   mirrors the outward velocity component. This is rare by construction and is
   reported through the return value so callers can count it.

Units are SI throughout: positions and lengths [m], angles [rad] CCW from +X.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np

from core.state import AgentState, wrap_angle
from scenario.config import ScenarioConfig

__all__ = [
    "Corridor",
    "NoFlyZone",
    "CorridorQuery",
    "World",
]


# --------------------------------------------------------------------------
# Static context primitives
# --------------------------------------------------------------------------


class Corridor(NamedTuple):
    """A straight cruise corridor, stored as ``(p0, p1, half_width)``.

    Attributes:
        p0: start point ``(x, y)`` [m] of the centreline.
        p1: end point ``(x, y)`` [m] of the centreline.
        half_width: lateral half-width [m]; the corridor covers points within
            this distance of the centreline segment.
    """

    p0: tuple[float, float]
    p1: tuple[float, float]
    half_width: float

    def length(self) -> float:
        """Centreline length [m]."""
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    def direction(self) -> tuple[float, float]:
        """Unit vector ``(ux, uy)`` [-] from ``p0`` toward ``p1``.

        Returns ``(1.0, 0.0)`` for a degenerate zero-length corridor.
        """
        dx = self.p1[0] - self.p0[0]
        dy = self.p1[1] - self.p0[1]
        n = math.hypot(dx, dy)
        if n <= 0.0:
            return 1.0, 0.0
        return dx / n, dy / n

    def heading(self) -> float:
        """Centreline heading [rad], CCW from +X, wrapped to (-pi, pi]."""
        ux, uy = self.direction()
        return wrap_angle(math.atan2(uy, ux))


class NoFlyZone(NamedTuple):
    """A circular prohibited area, stored as ``(center, radius)``.

    Attributes:
        center: centre ``(x, y)`` [m].
        radius: radius [m].
    """

    center: tuple[float, float]
    radius: float


class CorridorQuery(NamedTuple):
    """Result of :meth:`World.nearest_corridor`.

    Being a ``NamedTuple`` it unpacks positionally as well as by attribute.

    Attributes:
        index: index [-] into :attr:`World.corridors`, or ``-1`` if the world
            has no corridors (all other fields are then neutral).
        distance: perpendicular distance [m] from the query point to the
            corridor centreline SEGMENT (never negative).
        signed_offset: lateral offset [m] of the query point from the
            centreline, positive to the LEFT of the ``p0 -> p1`` direction
            (i.e. positive is +90 deg CCW from the corridor heading).
        s_along: arc length [m] of the projection along the centreline from
            ``p0``, clamped to ``[0, length]``.
        closest: closest point ``(x, y)`` [m] on the centreline segment.
        inside: True if ``distance <= half_width`` [-].
        heading: corridor centreline heading [rad], CCW from +X.
    """

    index: int
    distance: float
    signed_offset: float
    s_along: float
    closest: tuple[float, float]
    inside: bool
    heading: float


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------


class World:
    """The square airspace and its static context.

    Attributes:
        side_m: region side length [m].
        half_m: half side length [m]; the region is ``[-half, +half]^2``.
        margin_m: width [m] of the soft turn-back band inside each wall.
        corridors: list of :class:`Corridor` (empty if disabled in config).
        nfz: list of :class:`NoFlyZone`.
    """

    __slots__ = ("side_m", "half_m", "margin_m", "corridors", "nfz")

    #: Turn-back band width as a fraction [-] of the region HALF side.
    MARGIN_FRACTION: float = 0.12

    #: No-fly-zone radius sampling range as a fraction [-] of the half side.
    NFZ_RADIUS_FRAC_MIN: float = 0.06
    NFZ_RADIUS_FRAC_MAX: float = 0.14

    #: Corridor half-width sampling range as a fraction [-] of the half side.
    CORRIDOR_HALF_WIDTH_FRAC_MIN: float = 0.05
    CORRIDOR_HALF_WIDTH_FRAC_MAX: float = 0.12

    def __init__(
        self,
        cfg: ScenarioConfig,
        rng: Optional[np.random.Generator] = None,
        *,
        corridors: Optional[Sequence[Corridor]] = None,
        nfz: Optional[Sequence[NoFlyZone]] = None,
    ) -> None:
        """Build the region and sample its static context.

        Args:
            cfg: scenario configuration; supplies ``area_km2``,
                ``corridor_enabled`` and ``nfz_count``.
            rng: generator used to place corridors and no-fly zones; pass the
                ``'scenario'`` stream of :class:`~core.rng.RNGHub`. Required
                unless both ``corridors`` and ``nfz`` are supplied explicitly.
            corridors: explicit corridor list, bypassing the random placement
                (used by :meth:`restore` and by unit tests).
            nfz: explicit no-fly-zone list, bypassing the random placement.

        Raises:
            ValueError: if ``rng`` is ``None`` and the context is not fully
                supplied explicitly.
        """
        self.side_m: float = float(cfg.region_side_m())
        self.half_m: float = 0.5 * self.side_m
        self.margin_m: float = self.MARGIN_FRACTION * self.half_m

        if corridors is None or nfz is None:
            if rng is None:
                raise ValueError(
                    "World needs an rng unless corridors and nfz are both given"
                )

        if corridors is not None:
            self.corridors: list[Corridor] = [
                Corridor((float(c[0][0]), float(c[0][1])),
                         (float(c[1][0]), float(c[1][1])),
                         float(c[2]))
                for c in corridors
            ]
        elif cfg.corridor_enabled:
            self.corridors = self._sample_corridors(rng)  # type: ignore[arg-type]
        else:
            self.corridors = []

        if nfz is not None:
            self.nfz: list[NoFlyZone] = [
                NoFlyZone((float(z[0][0]), float(z[0][1])), float(z[1]))
                for z in nfz
            ]
        else:
            self.nfz = self._sample_nfz(
                rng,  # type: ignore[arg-type]
                int(max(0, cfg.nfz_count)),
            )

    # -- construction helpers ---------------------------------------------

    def _sample_corridors(self, rng: np.random.Generator) -> list[Corridor]:
        """Place one corridor crossing the whole region.

        The corridor is a chord through the origin region: a random heading is
        drawn, the centreline is offset laterally by a random amount, and the
        segment is extended well past the walls so that it spans the region
        end to end. One corridor is enough for the first-tier configuration --
        CORRIDOR_TRANSIT agents need a direction to follow, not a road network.

        Args:
            rng: generator ('scenario' stream).

        Returns:
            A single-element list of :class:`Corridor`.
        """
        h = self.half_m
        psi = float(rng.uniform(-math.pi, math.pi))
        ux, uy = math.cos(psi), math.sin(psi)
        # Left normal of the corridor direction.
        nx, ny = -uy, ux
        offset = float(rng.uniform(-0.35 * h, 0.35 * h))
        half_width = h * float(
            rng.uniform(
                self.CORRIDOR_HALF_WIDTH_FRAC_MIN,
                self.CORRIDOR_HALF_WIDTH_FRAC_MAX,
            )
        )
        # 1.5 * diagonal guarantees the segment leaves the square on both ends.
        reach = 1.5 * math.sqrt(2.0) * h
        cx, cy = offset * nx, offset * ny
        p0 = (cx - reach * ux, cy - reach * uy)
        p1 = (cx + reach * ux, cy + reach * uy)
        return [Corridor(p0, p1, half_width)]

    def _sample_nfz(self, rng: np.random.Generator, count: int) -> list[NoFlyZone]:
        """Place ``count`` non-overlapping circular no-fly zones.

        Zones are kept clear of the walls by their own radius and are rejected
        if they overlap an already placed zone. If a zone cannot be placed in a
        bounded number of tries, fewer zones are returned rather than looping
        forever -- the count is context, not a hard requirement.

        Args:
            rng: generator ('scenario' stream).
            count: requested number of zones [-].

        Returns:
            List of :class:`NoFlyZone`, length <= ``count``.
        """
        zones: list[NoFlyZone] = []
        h = self.half_m
        for _ in range(count):
            placed = False
            for _try in range(64):
                r = h * float(
                    rng.uniform(self.NFZ_RADIUS_FRAC_MIN, self.NFZ_RADIUS_FRAC_MAX)
                )
                lim = max(0.0, h - r)
                cx = float(rng.uniform(-lim, lim))
                cy = float(rng.uniform(-lim, lim))
                ok = True
                for z in zones:
                    if math.hypot(cx - z.center[0], cy - z.center[1]) < (r + z.radius):
                        ok = False
                        break
                if ok:
                    zones.append(NoFlyZone((cx, cy), r))
                    placed = True
                    break
            if not placed:
                break
        return zones

    # -- geometry queries -------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(xmin, ymin, xmax, ymax)`` [m]."""
        h = self.half_m
        return -h, -h, h, h

    def contains(self, pos: Sequence[float], margin: float = 0.0) -> bool:
        """True if ``pos`` [m] lies inside the region shrunk by ``margin`` [m].

        Args:
            pos: position ``(x, y)`` [m].
            margin: inward shrink [m]; 0 tests the raw square.

        Returns:
            True if inside.
        """
        lim = self.half_m - float(margin)
        return abs(float(pos[0])) <= lim and abs(float(pos[1])) <= lim

    def out_of_bounds(self, pos: Sequence[float]) -> bool:
        """True if ``pos`` [m] is strictly outside the region walls."""
        return not self.contains(pos, 0.0)

    def inside_nfz(self, pos: Sequence[float], clearance: float = 0.0) -> bool:
        """True if ``pos`` [m] is inside any no-fly zone grown by ``clearance``.

        Args:
            pos: position ``(x, y)`` [m].
            clearance: extra radial buffer [m] added to every zone radius.

        Returns:
            True if the point violates any zone.
        """
        x = float(pos[0])
        y = float(pos[1])
        c = float(clearance)
        for z in self.nfz:
            if math.hypot(x - z.center[0], y - z.center[1]) <= (z.radius + c):
                return True
        return False

    def nfz_index(self, pos: Sequence[float], clearance: float = 0.0) -> int:
        """Index [-] of the first violated no-fly zone, or ``-1`` if none."""
        x = float(pos[0])
        y = float(pos[1])
        c = float(clearance)
        for i, z in enumerate(self.nfz):
            if math.hypot(x - z.center[0], y - z.center[1]) <= (z.radius + c):
                return i
        return -1

    def nearest_nfz(self, pos: Sequence[float]) -> tuple[int, float]:
        """Nearest no-fly zone and its SIGNED surface distance.

        Args:
            pos: position ``(x, y)`` [m].

        Returns:
            ``(index, distance)`` where ``distance`` [m] is the range to the
            zone boundary, NEGATIVE inside the zone. ``(-1, inf)`` if the
            world has no zones.
        """
        best_i = -1
        best_d = float("inf")
        x = float(pos[0])
        y = float(pos[1])
        for i, z in enumerate(self.nfz):
            d = math.hypot(x - z.center[0], y - z.center[1]) - z.radius
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    def nearest_corridor(self, pos: Sequence[float]) -> CorridorQuery:
        """Project ``pos`` onto the closest corridor centreline segment.

        Args:
            pos: position ``(x, y)`` [m].

        Returns:
            A :class:`CorridorQuery`. When the world has no corridors the
            result is ``index = -1``, ``distance = inf``, ``inside = False``
            and the query point itself as ``closest``, so callers can branch
            on ``index < 0`` without special-casing shapes.
        """
        x = float(pos[0])
        y = float(pos[1])
        if not self.corridors:
            return CorridorQuery(
                index=-1,
                distance=float("inf"),
                signed_offset=0.0,
                s_along=0.0,
                closest=(x, y),
                inside=False,
                heading=0.0,
            )

        best: Optional[CorridorQuery] = None
        for i, c in enumerate(self.corridors):
            ux, uy = c.direction()
            length = c.length()
            dx = x - c.p0[0]
            dy = y - c.p0[1]
            s = dx * ux + dy * uy
            if s < 0.0:
                s = 0.0
            elif s > length:
                s = length
            qx = c.p0[0] + s * ux
            qy = c.p0[1] + s * uy
            dist = math.hypot(x - qx, y - qy)
            # Signed lateral offset: positive to the LEFT of p0->p1.
            signed = (-uy) * dx + ux * dy
            if best is None or dist < best.distance:
                best = CorridorQuery(
                    index=i,
                    distance=dist,
                    signed_offset=signed,
                    s_along=s,
                    closest=(qx, qy),
                    inside=dist <= c.half_width,
                    heading=wrap_angle(math.atan2(uy, ux)),
                )
        assert best is not None  # non-empty list guarantees this
        return best

    # -- sampling ---------------------------------------------------------

    def sample_free_position(
        self,
        rng: np.random.Generator,
        *,
        margin: Optional[float] = None,
        nfz_clearance: float = 0.0,
        max_tries: int = 64,
    ) -> tuple[float, float]:
        """Draw a uniform position inside the region and outside every NFZ.

        Args:
            rng: generator; pass the ``'placement'`` stream.
            margin: inward margin [m] kept from the walls; defaults to
                :attr:`margin_m` so freshly placed agents do not start inside
                the turn-back band.
            nfz_clearance: extra buffer [m] required from any no-fly zone.
            max_tries: rejection-sampling budget [-]. If it is exhausted the
                sample with the LARGEST no-fly-zone clearance seen is returned
                rather than raising: a crowded world must still be runnable.

        Returns:
            ``(x, y)`` position [m].
        """
        m = self.margin_m if margin is None else float(margin)
        lim = max(0.0, self.half_m - m)
        best: tuple[float, float] = (0.0, 0.0)
        best_clear = -float("inf")
        for _ in range(max(1, int(max_tries))):
            x = float(rng.uniform(-lim, lim))
            y = float(rng.uniform(-lim, lim))
            _, d = self.nearest_nfz((x, y))
            if d > best_clear:
                best_clear = d
                best = (x, y)
            if d > nfz_clearance:
                return x, y
        return best

    def sample_corridor_position(
        self,
        rng: np.random.Generator,
        *,
        index: int = 0,
        margin: Optional[float] = None,
    ) -> tuple[float, float]:
        """Draw a position inside a corridor, clipped to the region.

        Falls back to :meth:`sample_free_position` when the world has no
        corridors.

        Args:
            rng: generator; pass the ``'placement'`` stream.
            index: corridor index [-]; clamped into range.
            margin: inward wall margin [m], defaults to :attr:`margin_m`.

        Returns:
            ``(x, y)`` position [m].
        """
        if not self.corridors:
            return self.sample_free_position(rng, margin=margin)
        i = max(0, min(int(index), len(self.corridors) - 1))
        c = self.corridors[i]
        ux, uy = c.direction()
        m = self.margin_m if margin is None else float(margin)
        lim = max(0.0, self.half_m - m)
        for _ in range(64):
            s = float(rng.uniform(0.0, c.length()))
            lat = float(rng.uniform(-c.half_width, c.half_width))
            x = c.p0[0] + s * ux - uy * lat
            y = c.p0[1] + s * uy + ux * lat
            if abs(x) <= lim and abs(y) <= lim:
                return x, y
        return self.sample_free_position(rng, margin=margin)

    # -- boundary policy --------------------------------------------------

    def inward_normal(self, pos: Sequence[float]) -> tuple[float, float, float]:
        """Inward push direction and urgency inside the turn-back band.

        Each wall contributes a unit component along its inward normal, scaled
        by how deep into the band the point is (0 at the inner edge, 1 at the
        wall, and larger than 1 once outside). The four contributions are
        summed, which gives the correct diagonal push in a corner.

        Args:
            pos: position ``(x, y)`` [m].

        Returns:
            ``(nx, ny, urgency)`` where ``(nx, ny)`` is a UNIT inward vector
            [-] (``(0, 0)`` when the point is clear of the band) and
            ``urgency`` is in ``[0, 1]`` [-], 0 outside the band, 1 at or
            beyond a wall.
        """
        h = self.half_m
        m = self.margin_m
        if m <= 0.0:
            return 0.0, 0.0, 0.0
        x = float(pos[0])
        y = float(pos[1])
        d_left = x + h        # distance to the -X wall
        d_right = h - x       # distance to the +X wall
        d_bottom = y + h      # distance to the -Y wall
        d_top = h - y         # distance to the +Y wall

        w_left = max(0.0, 1.0 - d_left / m)     # push toward +X
        w_right = max(0.0, 1.0 - d_right / m)   # push toward -X
        w_bottom = max(0.0, 1.0 - d_bottom / m)  # push toward +Y
        w_top = max(0.0, 1.0 - d_top / m)       # push toward -Y

        nx = w_left - w_right
        ny = w_bottom - w_top
        urgency = min(1.0, max(w_left, w_right, w_bottom, w_top))
        norm = math.hypot(nx, ny)
        if norm <= 0.0:
            return 0.0, 0.0, 0.0
        return nx / norm, ny / norm, urgency

    def containment_heading(
        self,
        pos: Sequence[float],
        psi: float,
    ) -> Optional[float]:
        """Desired heading [rad] that turns an agent gently back inside.

        Returns ``None`` when no containment is needed, i.e. the agent is clear
        of the margin band, or it is inside the band but already flying inward.
        Letting an inward-flying agent alone is what keeps the manoeuvre gentle
        and avoids a permanent wall-hugging orbit.

        The returned heading is the direction of

            (1 - w) * u_hat + w * n_hat

        with ``u_hat`` the current heading unit vector, ``n_hat`` the inward
        normal and ``w`` the urgency, so the command is a small nudge at the
        band edge and a full about-face at the wall.

        Args:
            pos: position ``(x, y)`` [m].
            psi: current heading [rad], CCW from +X.

        Returns:
            Desired heading [rad] wrapped to (-pi, pi], or ``None``.
        """
        nx, ny, w = self.inward_normal(pos)
        if w <= 0.0:
            return None
        ux, uy = math.cos(float(psi)), math.sin(float(psi))
        if (ux * nx + uy * ny) >= 0.0 and self.contains(pos, 0.0):
            # Already heading inward and still inside: no intervention.
            return None
        bx = (1.0 - w) * ux + w * nx
        by = (1.0 - w) * uy + w * ny
        if bx == 0.0 and by == 0.0:
            # Exactly antiparallel at full urgency: turn onto the normal.
            return wrap_angle(math.atan2(ny, nx))
        return wrap_angle(math.atan2(by, bx))

    def enforce_bounds(self, state: AgentState) -> bool:
        """Hard fallback: reflect an agent that has crossed a wall.

        Position is clamped back onto the wall and the heading is mirrored
        about that wall, which reverses only the outward velocity component and
        leaves the tangential one untouched (a specular bounce). Speed is
        unchanged, so the manoeuvre stays inside the performance envelope in
        magnitude even though it is an instantaneous heading change; it exists
        only as a safety net and should fire rarely.

        Args:
            state: agent state, MUTATED in place.

        Returns:
            True if a reflection was applied, False if the agent was inside.
        """
        h = self.half_m
        hit = False
        if state.x < -h:
            state.x = -h
            state.psi = wrap_angle(math.pi - state.psi)  # mirror about +X axis
            hit = True
        elif state.x > h:
            state.x = h
            state.psi = wrap_angle(math.pi - state.psi)
            hit = True
        if state.y < -h:
            state.y = -h
            state.psi = wrap_angle(-state.psi)  # mirror about +Y axis
            hit = True
        elif state.y > h:
            state.y = h
            state.psi = wrap_angle(-state.psi)
            hit = True
        return hit

    # -- rendering / persistence ------------------------------------------

    def to_render_dict(self) -> dict[str, Any]:
        """Static context in the shape the Canvas 2D GUI consumes.

        Returns:
            ``{"side_m", "half_m", "margin_m", "bounds", "corridors", "nfz"}``
            with plain JSON types only ([xmin, ymin, xmax, ymax] for bounds,
            corridors as ``{"p0", "p1", "half_width"}``, zones as
            ``{"center", "radius"}``). All lengths [m].
        """
        xmin, ymin, xmax, ymax = self.bounds
        return {
            "side_m": float(self.side_m),
            "half_m": float(self.half_m),
            "margin_m": float(self.margin_m),
            "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "corridors": [
                {
                    "p0": [float(c.p0[0]), float(c.p0[1])],
                    "p1": [float(c.p1[0]), float(c.p1[1])],
                    "half_width": float(c.half_width),
                }
                for c in self.corridors
            ],
            "nfz": [
                {
                    "center": [float(z.center[0]), float(z.center[1])],
                    "radius": float(z.radius),
                }
                for z in self.nfz
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        """Capture the full world description as a JSON-safe dict.

        The world is static after construction, so this doubles as a complete
        serialisation: :meth:`restore` rebuilds an identical world without
        touching any RNG stream.
        """
        d = self.to_render_dict()
        return d

    def restore(self, snap: dict[str, Any]) -> None:
        """Rebuild this world in place from a :meth:`snapshot` dict.

        Args:
            snap: dict produced by :meth:`snapshot` (or :meth:`to_render_dict`).
        """
        self.side_m = float(snap["side_m"])
        self.half_m = float(snap.get("half_m", 0.5 * self.side_m))
        self.margin_m = float(
            snap.get("margin_m", self.MARGIN_FRACTION * self.half_m)
        )
        self.corridors = [
            Corridor(
                (float(c["p0"][0]), float(c["p0"][1])),
                (float(c["p1"][0]), float(c["p1"][1])),
                float(c["half_width"]),
            )
            for c in snap.get("corridors", [])
        ]
        self.nfz = [
            NoFlyZone(
                (float(z["center"][0]), float(z["center"][1])),
                float(z["radius"]),
            )
            for z in snap.get("nfz", [])
        ]

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "World":
        """Build a :class:`World` directly from a :meth:`snapshot` dict."""
        side = float(snap["side_m"])
        cfg = ScenarioConfig(
            area_km2=(side / 1000.0) ** 2,
            corridor_enabled=bool(snap.get("corridors")),
            nfz_count=len(snap.get("nfz", [])),
        )
        world = cls(cfg, corridors=[], nfz=[])
        world.restore(snap)
        return world

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"World(side={self.side_m:.0f} m, corridors={len(self.corridors)}, "
            f"nfz={len(self.nfz)})"
        )
