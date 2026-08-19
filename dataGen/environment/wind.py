"""Ambient wind field: steady wind plus an optional Ornstein-Uhlenbeck gust.

WHY WIND EXISTS AT ALL (doc 3.6, and it is not cosmetic)
--------------------------------------------------------
The whole paper rests on the claim that a constant-velocity (CV) extrapolator
cannot predict this traffic (doc 3.1). Wind attacks CV from two independent
directions:

1. **Systematic bias.** An agent controls its AIRSPEED vector; what a sensor
   observes is the GROUND velocity, air-relative velocity plus advected wind.
   Whenever an agent turns, its air-relative velocity rotates but the wind term
   does not, so the ground track curves in a way that a straight-line
   extrapolation of the *observed* velocity gets wrong in a direction that is
   correlated across the whole episode. A CV baseline therefore misses
   systematically, not just noisily -- exactly the failure mode the paper needs
   to demonstrate.

2. **Aleatoric uncertainty with a physical origin.** Because the episode wind
   is drawn per episode and is not observable from a single track, part of the
   future is genuinely unknowable from the input. That irreducible component is
   what a calibrated predictor must express as sigma. Without it, "calibrated
   uncertainty" would have no physical referent and the calibration experiment
   (doc 1) would be measuring nothing but model noise.

The gust process (an OU process, doc 3.6 "2nd-tier extension") adds a
time-correlated component. It is DISABLED by default: the first-tier minimum
configuration of doc 3.7 says "steady wind only". Turn it on with
``WindConfig.gust_enabled`` only for ablations.

Conventions (global, see :mod:`core.constants`):
    +X = East [m], +Y = North [m].
    A wind vector ``(wx, wy)`` [m/s] is the velocity of the AIR MASS, i.e. the
    direction the air moves TOWARD (NOT the meteorological "wind from"
    convention). ``WindConfig.dir_*_rad`` uses the same convention and is
    measured CCW from +X.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from core.state import wrap_angle
from scenario.config import WindConfig

__all__ = ["WindField"]


class WindField:
    """Per-episode wind: a constant steady component plus an optional gust.

    The steady component is sampled once per episode from the configured
    ranges and never changes afterwards; that constancy is deliberate, because
    a constant offset in the ground-velocity field is precisely the systematic
    error a CV extrapolator cannot absorb.

    The gust component is a two-dimensional Ornstein-Uhlenbeck process with
    independent East and North components,

        dg = -(g / tau) dt + sigma * sqrt(2 / tau) dW,

    integrated with the EXACT discretisation (no Euler-Maruyama error):

        g <- g * exp(-dt / tau)
             + sigma * sqrt(1 - exp(-2 dt / tau)) * z,   z ~ N(0, 1)

    so the process is stationary with standard deviation ``sigma`` [m/s] and
    correlation time ``tau`` [s] for ANY step size ``dt``.

    Attributes:
        cfg: the :class:`~scenario.config.WindConfig` this field was built from.
        speed_ms: sampled steady wind speed [m/s].
        dir_rad: sampled steady wind direction [rad], CCW from +X, wrapped to
            (-pi, pi]; the direction the air moves TOWARD.
    """

    __slots__ = ("cfg", "speed_ms", "dir_rad", "_wx", "_wy", "_gx", "_gy", "_t")

    def __init__(
        self,
        cfg: WindConfig,
        rng: Optional[np.random.Generator] = None,
        *,
        speed_ms: Optional[float] = None,
        dir_rad: Optional[float] = None,
    ) -> None:
        """Sample (or explicitly set) the steady wind for one episode.

        Args:
            cfg: wind configuration providing the sampling ranges [m/s, rad]
                and the gust parameters.
            rng: generator used for the steady-wind draw; pass the ``'wind'``
                stream of :class:`~core.rng.RNGHub`. May be ``None`` only if
                both ``speed_ms`` and ``dir_rad`` are given explicitly.
            speed_ms: override for the steady wind speed [m/s]; skips the draw.
            dir_rad: override for the steady wind direction [rad]; skips the
                draw. Same "moves toward" convention as the config.

        Raises:
            ValueError: if ``rng`` is ``None`` and the overrides are not both
                supplied.
        """
        self.cfg: WindConfig = cfg

        if speed_ms is None or dir_rad is None:
            if rng is None:
                raise ValueError(
                    "WindField needs an rng unless both speed_ms and dir_rad "
                    "are given"
                )
            s_lo = float(cfg.speed_min_ms)
            s_hi = float(cfg.speed_max_ms)
            d_lo = float(cfg.dir_min_rad)
            d_hi = float(cfg.dir_max_rad)
            # Two draws are always consumed, in this fixed order, so that the
            # 'wind' stream advances identically whether or not the range is
            # degenerate. Reproducibility depends on that.
            u_s = float(rng.random())
            u_d = float(rng.random())
            drawn_speed = s_lo + u_s * (s_hi - s_lo)
            drawn_dir = d_lo + u_d * (d_hi - d_lo)
            if speed_ms is None:
                speed_ms = drawn_speed
            if dir_rad is None:
                dir_rad = drawn_dir

        self.speed_ms: float = max(0.0, float(speed_ms))
        self.dir_rad: float = wrap_angle(float(dir_rad))

        self._wx: float = self.speed_ms * math.cos(self.dir_rad)
        self._wy: float = self.speed_ms * math.sin(self.dir_rad)

        # Gust state [m/s], East and North components of the OU process.
        self._gx: float = 0.0
        self._gy: float = 0.0
        # Simulation time [s] of the last completed :meth:`step`.
        self._t: float = 0.0

    # -- queries ----------------------------------------------------------

    @property
    def gust_enabled(self) -> bool:
        """True if the OU gust process is active [-] (doc 3.6 2nd tier)."""
        return bool(self.cfg.gust_enabled)

    @property
    def steady(self) -> tuple[float, float]:
        """Steady wind vector ``(wx, wy)`` [m/s] in world axes."""
        return self._wx, self._wy

    @property
    def gust(self) -> tuple[float, float]:
        """Current gust vector ``(gx, gy)`` [m/s]; ``(0, 0)`` when disabled."""
        return self._gx, self._gy

    @property
    def time_s(self) -> float:
        """Simulation time [s] accumulated by :meth:`step`."""
        return self._t

    def at(
        self,
        t: float,
        pos: Optional[Sequence[float]] = None,
    ) -> tuple[float, float]:
        """Wind vector ``(wx, wy)`` [m/s] at time ``t`` [s] and position ``pos``.

        The first-tier field is uniform in space and, for its steady part,
        constant in time; the gust part is whatever the OU state was after the
        most recent :meth:`step`. ``t`` and ``pos`` are therefore accepted but
        not used to interpolate anything -- they are in the signature so that a
        spatially varying field (building wake / shear layer, doc 3.6 second
        tier) can be substituted without touching a single call site.

        Args:
            t: simulation time [s] since episode start.
            pos: position ``(x, y)`` [m] at which the wind is requested; may be
                ``None``.

        Returns:
            ``(wx, wy)`` wind velocity components [m/s], +X East, +Y North,
            being the velocity of the air mass (direction the air moves
            toward).
        """
        del t, pos  # uniform field; see docstring
        return self._wx + self._gx, self._wy + self._gy

    def speed_at(self, t: float, pos: Optional[Sequence[float]] = None) -> float:
        """Wind speed magnitude [m/s] at ``(t, pos)``."""
        wx, wy = self.at(t, pos)
        return math.hypot(wx, wy)

    def direction_at(self, t: float, pos: Optional[Sequence[float]] = None) -> float:
        """Wind direction [rad] at ``(t, pos)``, CCW from +X, in (-pi, pi].

        Returns the steady direction when the total vector is exactly zero, so
        the value is always well defined for metadata logging.
        """
        wx, wy = self.at(t, pos)
        if wx == 0.0 and wy == 0.0:
            return self.dir_rad
        return wrap_angle(math.atan2(wy, wx))

    # -- evolution --------------------------------------------------------

    def step(self, dt: float, rng: Optional[np.random.Generator] = None) -> None:
        """Advance the gust process by ``dt`` seconds.

        When gusts are disabled this only advances the internal clock and
        draws NOTHING from ``rng``: the ``'wind'`` stream must not be consumed
        differently between a gust-off and a gust-on run of the same seed, or
        replay comparisons across ablations become meaningless.

        Args:
            dt: integration step [s]; must be >= 0. Any step size is exact for
                this discretisation.
            rng: generator for the gust innovation; pass the ``'wind'`` stream.
                Required only when gusts are enabled.

        Raises:
            ValueError: if ``dt`` is negative, or gusts are enabled and ``rng``
                is ``None``.
        """
        if dt < 0.0:
            raise ValueError(f"WindField.step: dt must be >= 0 s, got {dt}")
        self._t += float(dt)

        if not self.cfg.gust_enabled or dt == 0.0:
            return
        if rng is None:
            raise ValueError("WindField.step: rng is required when gusts are on")

        tau = float(self.cfg.gust_tau_s)
        sigma = float(self.cfg.gust_sigma_ms)
        if tau <= 0.0 or sigma <= 0.0:
            # Degenerate configuration: treat as no gust, but stay silent --
            # WindConfig.validate() is the place that reports it as a problem.
            self._gx = 0.0
            self._gy = 0.0
            return

        decay = math.exp(-float(dt) / tau)
        innov = sigma * math.sqrt(max(0.0, 1.0 - decay * decay))
        z = rng.standard_normal(2)
        self._gx = self._gx * decay + innov * float(z[0])
        self._gy = self._gy * decay + innov * float(z[1])

    def reset_gust(self) -> None:
        """Zero the gust state and the internal clock (keeps the steady wind)."""
        self._gx = 0.0
        self._gy = 0.0
        self._t = 0.0

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture the full mutable state (doc 3.3 branch rollouts).

        Returns:
            A JSON-safe dict holding the steady wind, the gust state and the
            internal clock. The configuration itself is NOT included: it is
            part of the scenario, not of the episode state.
        """
        return {
            "speed_ms": float(self.speed_ms),
            "dir_rad": float(self.dir_rad),
            "gx": float(self._gx),
            "gy": float(self._gy),
            "t": float(self._t),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore from a :meth:`snapshot` result.

        Args:
            snap: dict produced by :meth:`snapshot`.
        """
        self.speed_ms = max(0.0, float(snap["speed_ms"]))
        self.dir_rad = wrap_angle(float(snap["dir_rad"]))
        self._wx = self.speed_ms * math.cos(self.dir_rad)
        self._wy = self.speed_ms * math.sin(self.dir_rad)
        self._gx = float(snap.get("gx", 0.0))
        self._gy = float(snap.get("gy", 0.0))
        self._t = float(snap.get("t", 0.0))

    def to_render_dict(self) -> dict[str, Any]:
        """Compact description for the Canvas 2D GUI wind indicator.

        Returns:
            ``{"speed_ms", "dir_rad", "wx", "wy", "gust_enabled"}`` -- the
            total (steady + current gust) vector, so the arrow the GUI draws
            matches what the agents actually feel.
        """
        wx, wy = self.at(self._t, None)
        return {
            "speed_ms": float(math.hypot(wx, wy)),
            "dir_rad": float(self.direction_at(self._t, None)),
            "wx": float(wx),
            "wy": float(wy),
            "gust_enabled": bool(self.cfg.gust_enabled),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"WindField(speed={self.speed_ms:.2f} m/s, "
            f"dir={math.degrees(self.dir_rad):.1f} deg, "
            f"gust={'on' if self.cfg.gust_enabled else 'off'})"
        )
