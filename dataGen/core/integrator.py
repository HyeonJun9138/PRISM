"""Generic numerical integrators for the 2D agent state vector.

This module is deliberately *dependency free* (numpy + stdlib only, no other
``core`` import) so it can be reused, unit-tested and reasoned about in
isolation. It knows nothing about agents, envelopes or wind -- it only moves a
state vector forward in time given a derivative function.

State-vector layout used by the rest of dataGen (documented here so that every
caller agrees, but *not* enforced by the generic routines):

    index  symbol  unit     meaning
    -----  ------  -------  ----------------------------------------------
      0    x       [m]      East position, local tangent plane
      1    y       [m]      North position, local tangent plane
      2    psi     [rad]    heading, CCW from +X (NOT wrapped by the
                            integrator -- wrapping is the caller's job)
      3    v       [m/s]    airspeed magnitude (ground speed for unpowered)

Two families are provided:

* :func:`rk4` / :func:`rk4_time_varying` -- classical explicit 4th-order
  Runge-Kutta. This is the production integrator for powered agents: with a
  turn rate and a longitudinal acceleration held constant across the step it
  reproduces a coordinated turn to well below millimetre accuracy at
  ``SIM_DT = 0.05 s``.
* :func:`semi_implicit_euler` -- symplectic (velocity-first) Euler, kept for
  cheap sanity checks and for comparison plots in the GUI. It is 1st order and
  MUST NOT be used to generate the published dataset.

:func:`rk4_unicycle` is a scalar (pure-Python float) specialisation of
:func:`rk4` for the wind-advected unicycle. It is mathematically identical to
calling :func:`rk4` with the corresponding derivative closure -- see
``tests/test_kinematics.py``, which asserts the two agree -- but avoids the
per-call numpy array allocations that would otherwise dominate the runtime of
a 5,000-episode dataset (1,200 integration steps x ~10 agents x 5,000
episodes = ~60 million derivative evaluations).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "STATE_X",
    "STATE_Y",
    "STATE_PSI",
    "STATE_V",
    "STATE_DIM",
    "DerivFn",
    "TimeDerivFn",
    "rk4",
    "rk4_time_varying",
    "euler",
    "semi_implicit_euler",
    "integrate_substeps",
    "rk4_unicycle",
]

#: Index of the East position [m] in the canonical 2D state vector.
STATE_X: int = 0
#: Index of the North position [m] in the canonical 2D state vector.
STATE_Y: int = 1
#: Index of the heading [rad] in the canonical 2D state vector.
STATE_PSI: int = 2
#: Index of the speed [m/s] in the canonical 2D state vector.
STATE_V: int = 3
#: Length of the canonical 2D state vector [-].
STATE_DIM: int = 4

#: Autonomous derivative function: ``f(y) -> dy/dt`` with matching shape.
DerivFn = Callable[[np.ndarray], np.ndarray]
#: Time-varying derivative function: ``f(t, y) -> dy/dt``.
TimeDerivFn = Callable[[float, np.ndarray], np.ndarray]


# --------------------------------------------------------------------------
# Generic vector integrators
# --------------------------------------------------------------------------


def rk4(deriv_fn: DerivFn, y0: np.ndarray, dt: float) -> np.ndarray:
    """Advance an autonomous ODE one step with classical 4th-order Runge-Kutta.

    Solves ``dy/dt = deriv_fn(y)`` over one interval of length ``dt``.

    Args:
        deriv_fn: derivative callable ``f(y) -> dy/dt``. It must return an
            array broadcastable to the shape of ``y0`` and must not mutate its
            argument. Units of the returned array are the units of ``y0``
            divided by seconds.
        y0: state vector at the start of the step, any shape.
        dt: step length [s]. May be negative for backward integration.

    Returns:
        A NEW array holding the state at ``t + dt``. ``y0`` is never mutated.
    """
    y = np.asarray(y0, dtype=np.float64)
    h = float(dt)
    k1 = np.asarray(deriv_fn(y), dtype=np.float64)
    k2 = np.asarray(deriv_fn(y + (0.5 * h) * k1), dtype=np.float64)
    k3 = np.asarray(deriv_fn(y + (0.5 * h) * k2), dtype=np.float64)
    k4 = np.asarray(deriv_fn(y + h * k3), dtype=np.float64)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_time_varying(
    deriv_fn: TimeDerivFn,
    y0: np.ndarray,
    dt: float,
    t0: float = 0.0,
) -> np.ndarray:
    """Advance a time-dependent ODE one step with classical RK4.

    Solves ``dy/dt = deriv_fn(t, y)`` over ``[t0, t0 + dt]``. Used when the
    forcing term itself varies inside the step (e.g. a gust realisation
    sampled from an Ornstein-Uhlenbeck process).

    Args:
        deriv_fn: derivative callable ``f(t, y) -> dy/dt``; ``t`` is [s].
        y0: state vector at ``t0``, any shape.
        dt: step length [s].
        t0: start time [s] of the step.

    Returns:
        A NEW array holding the state at ``t0 + dt``.
    """
    y = np.asarray(y0, dtype=np.float64)
    h = float(dt)
    t = float(t0)
    k1 = np.asarray(deriv_fn(t, y), dtype=np.float64)
    k2 = np.asarray(deriv_fn(t + 0.5 * h, y + (0.5 * h) * k1), dtype=np.float64)
    k3 = np.asarray(deriv_fn(t + 0.5 * h, y + (0.5 * h) * k2), dtype=np.float64)
    k4 = np.asarray(deriv_fn(t + h, y + h * k3), dtype=np.float64)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler(deriv_fn: DerivFn, y0: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with explicit (forward) Euler -- reference only.

    First order and unconditionally less accurate than :func:`rk4`; provided
    so that convergence-order checks have something to compare against. Do not
    use it to generate dataset trajectories.

    Args:
        deriv_fn: derivative callable ``f(y) -> dy/dt``.
        y0: state vector at the start of the step.
        dt: step length [s].

    Returns:
        A NEW array holding the state at ``t + dt``.
    """
    y = np.asarray(y0, dtype=np.float64)
    return y + float(dt) * np.asarray(deriv_fn(y), dtype=np.float64)


def semi_implicit_euler(
    deriv_fn: DerivFn,
    y0: np.ndarray,
    dt: float,
    fast_idx: Sequence[int] = (STATE_PSI, STATE_V),
    slow_idx: Sequence[int] = (STATE_X, STATE_Y),
) -> np.ndarray:
    """Advance one step with semi-implicit (symplectic) Euler.

    The "fast" components (heading and speed for the canonical 2D state) are
    stepped first with an explicit Euler update; the "slow" components
    (position) are then stepped using the derivative evaluated at the ALREADY
    UPDATED fast components. For rotational motion this removes the
    systematic outward spiral of plain forward Euler.

    Any state index that appears in neither list is advanced with the
    derivative evaluated at the start of the step.

    Args:
        deriv_fn: derivative callable ``f(y) -> dy/dt``.
        y0: 1-D state vector at the start of the step.
        dt: step length [s].
        fast_idx: indices integrated first (rate-like states).
        slow_idx: indices integrated with the updated derivative
            (position-like states).

    Returns:
        A NEW 1-D array holding the state at ``t + dt``.

    Raises:
        ValueError: if ``y0`` is not one-dimensional.
    """
    y = np.array(y0, dtype=np.float64, copy=True)
    if y.ndim != 1:
        raise ValueError("semi_implicit_euler requires a 1-D state vector")
    h = float(dt)
    d0 = np.asarray(deriv_fn(y), dtype=np.float64)

    fast = np.asarray(fast_idx, dtype=np.intp)
    slow = np.asarray(slow_idx, dtype=np.intp)

    # Everything that is neither fast nor slow: plain explicit Euler.
    mask = np.ones(y.shape[0], dtype=bool)
    mask[fast] = False
    mask[slow] = False
    y_new = y.copy()
    y_new[mask] = y[mask] + h * d0[mask]

    # 1) fast states with the start-of-step derivative
    y_new[fast] = y[fast] + h * d0[fast]

    # 2) slow states with the derivative at the updated fast states
    y_mid = y.copy()
    y_mid[fast] = y_new[fast]
    d1 = np.asarray(deriv_fn(y_mid), dtype=np.float64)
    y_new[slow] = y[slow] + h * d1[slow]
    return y_new


def integrate_substeps(
    deriv_fn: DerivFn,
    y0: np.ndarray,
    dt: float,
    n_substeps: int,
    method: Callable[[DerivFn, np.ndarray, float], np.ndarray] = rk4,
) -> np.ndarray:
    """Advance ``n_substeps`` uniform sub-steps of total length ``dt``.

    Convenience wrapper used by convergence tests and by any caller that wants
    a finer internal step than ``SIM_DT`` without writing its own loop.

    Args:
        deriv_fn: derivative callable ``f(y) -> dy/dt``.
        y0: state vector at the start of the interval.
        dt: TOTAL interval length [s].
        n_substeps: number of equal sub-steps [-], must be >= 1.
        method: single-step integrator to apply, default :func:`rk4`.

    Returns:
        A NEW array holding the state at ``t + dt``.

    Raises:
        ValueError: if ``n_substeps < 1``.
    """
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    h = float(dt) / float(n_substeps)
    y = np.asarray(y0, dtype=np.float64)
    for _ in range(int(n_substeps)):
        y = method(deriv_fn, y, h)
    return y


# --------------------------------------------------------------------------
# Scalar specialisation: wind-advected unicycle (the hot path)
# --------------------------------------------------------------------------


def rk4_unicycle(
    x: float,
    y: float,
    psi: float,
    v: float,
    omega: float,
    a: float,
    wx_eff: float,
    wy_eff: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Scalar RK4 step of the wind-advected unicycle (coordinated turn).

    Integrates, with ``omega`` and ``a`` held CONSTANT across the step::

        dx/dt   = v*cos(psi) + wx_eff
        dy/dt   = v*sin(psi) + wy_eff
        dpsi/dt = omega
        dv/dt   = a

    Because ``dpsi/dt`` and ``dv/dt`` are constant, the RK4 stages for heading
    and speed collapse to the exact solution ``psi + omega*dt`` and
    ``v + a*dt``, and stages 2 and 3 of the position update coincide. All four
    stages are nevertheless written out explicitly so the scheme is auditable
    and provably identical to :func:`rk4` applied to the same derivative.

    Args:
        x: East position [m] at the start of the step.
        y: North position [m] at the start of the step.
        psi: heading [rad] at the start of the step (unwrapped in, unwrapped
            out -- the caller wraps).
        v: airspeed [m/s] at the start of the step.
        omega: turn rate [rad/s] held constant over the step, positive = CCW.
        a: longitudinal acceleration [m/s^2] held constant over the step.
        wx_eff: ALREADY SUSCEPTIBILITY-SCALED East wind component [m/s].
        wy_eff: ALREADY SUSCEPTIBILITY-SCALED North wind component [m/s].
        dt: step length [s].

    Returns:
        ``(x, y, psi, v)`` at ``t + dt``; ``psi`` is NOT wrapped and ``v`` is
        NOT clamped -- both are the caller's responsibility.
    """
    h = float(dt)
    half = 0.5 * h

    # stage 1 -- start of step
    dx1 = v * math.cos(psi) + wx_eff
    dy1 = v * math.sin(psi) + wy_eff

    # stage 2 -- midpoint using stage-1 rates
    psi2 = psi + half * omega
    v2 = v + half * a
    dx2 = v2 * math.cos(psi2) + wx_eff
    dy2 = v2 * math.sin(psi2) + wy_eff

    # stage 3 -- midpoint using stage-2 rates (identical to stage 2 here,
    # because dpsi/dt and dv/dt do not depend on the state)
    psi3 = psi + half * omega
    v3 = v + half * a
    dx3 = v3 * math.cos(psi3) + wx_eff
    dy3 = v3 * math.sin(psi3) + wy_eff

    # stage 4 -- end of step
    psi4 = psi + h * omega
    v4 = v + h * a
    dx4 = v4 * math.cos(psi4) + wx_eff
    dy4 = v4 * math.sin(psi4) + wy_eff

    sixth = h / 6.0
    x_new = x + sixth * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    y_new = y + sixth * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    psi_new = psi + h * omega
    v_new = v + h * a
    return x_new, y_new, psi_new, v_new
