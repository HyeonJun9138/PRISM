"""The physics of dataGen: powered and unpowered 2D motion, plus guidance laws.

This is the single place where an agent's motion is actually produced. Every
other module (agents, environment, simulation) proposes commands; this module
decides what the airframe is physically able to do with them and advances the
state.

Frame and sign conventions (doc convention, identical everywhere in dataGen):
    +X = East [m], +Y = North [m], local tangent plane.
    psi = heading [rad], CCW from +X, wrapped to (-pi, pi].
    omega > 0  =>  left turn  =>  CCW  =>  psi increases.
    a is longitudinal [m/s^2], along the body heading.
    wind_xy is the vector the AIR is moving toward, [m/s], world axes.

Why the command shaping order matters (doc 7.2)
-----------------------------------------------
The turn rate is a STATE, not a command. Commands are first clipped by the
speed-dependent envelope cap and then rate-limited by ``omega_dot_max`` (roll
onset). Without the rate limit, heading changes appear as instantaneous
kinks; reviewers notice that immediately and the trajectories stop looking
like aircraft. With it, every mode change costs a physically plausible roll-in
and roll-out, which is precisely the sub-second structure a constant-velocity
extrapolator cannot reproduce (doc 3.1).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.integrator import rk4, rk4_unicycle
from core.state import AgentState, PerformanceEnvelope, angle_diff, wrap_angle

__all__ = [
    "RHO_AIR",
    "step_powered",
    "step_passive",
    "step",
    "heading_error",
    "turn_rate_to_heading",
    "cross_track_guidance",
    "orbit_guidance",
    "min_turn_radius",
    "predict_cv",
]

#: Sea-level ISA air density [kg/m^3]. Used only by the passive (DEBRIS) drag
#: model; powered platforms are described by their performance envelope, not
#: by an aerodynamic model.
RHO_AIR: float = 1.225

#: Below this speed [m/s] a body's heading is treated as undefined and the
#: previous heading is retained instead of being recomputed from atan2 of a
#: numerically meaningless velocity vector.
_HEADING_EPS_V: float = 1e-6


# --------------------------------------------------------------------------
# 1) Powered motion -- coordinated turn / unicycle with wind
# --------------------------------------------------------------------------


def step_powered(
    state: AgentState,
    omega_cmd: float,
    a_cmd: float,
    env: PerformanceEnvelope,
    wind_xy: Sequence[float],
    dt: float,
) -> AgentState:
    """Advance one powered agent by ``dt`` under commanded turn rate and accel.

    Model (wind-advected unicycle / coordinated turn)::

        dx/dt   = v*cos(psi) + wind_x * env.wind_susceptibility
        dy/dt   = v*sin(psi) + wind_y * env.wind_susceptibility
        dpsi/dt = omega
        dv/dt   = a

    Command shaping, applied strictly in this order:

    a. ``a_cmd`` is clipped to ``[env.a_min, env.a_max]`` [m/s^2].
    b. the turn-rate cap is evaluated at the CURRENT airspeed,
       ``env.max_turn_rate(state.v)`` [rad/s] -- speed dependent, so a fast
       eVTOL genuinely turns more slowly than a slow one.
    c. ``omega_cmd`` is clipped to +/- that cap.
    d. the turn rate is RATE LIMITED toward the clipped command by
       ``env.omega_dot_max * dt`` [rad/s] (roll onset). This is what stops
       instantaneous heading snaps.
    e. the state is integrated with RK4 (:func:`core.integrator.rk4_unicycle`)
       holding the rate-limited ``omega`` and the clipped ``a`` constant.
    f. speed is clipped to ``[env.v_min, env.v_max]``; if it hits a bound the
       stored acceleration is zeroed, so the next step does not keep pushing
       against the limit.
    g. heading is wrapped to (-pi, pi].

    Note on step (f): the position integral inside the step uses the
    unclamped ``v(t) = v0 + a*t``. When a speed bound binds part way through
    the step the position error is bounded by ``0.5*|a|*dt^2`` (< 4 mm at
    ``SIM_DT = 0.05 s`` and ``|a| = 3 m/s^2``), which is far below the sensor
    noise floor and therefore irrelevant to the dataset.

    Args:
        state: current kinematic state; NEVER mutated.
        omega_cmd: commanded turn rate [rad/s], positive = left/CCW.
        a_cmd: commanded longitudinal acceleration [m/s^2].
        env: performance envelope of this platform.
        wind_xy: ambient wind ``(wx, wy)`` [m/s], the direction the air moves
            toward, in world axes.
        dt: integration step [s], expected to be ``SIM_DT``.

    Returns:
        A NEW :class:`~core.state.AgentState` at ``t + dt`` whose ``omega``
        and ``a`` fields hold the values actually flown over the step.
    """
    h = float(dt)

    # (a) longitudinal acceleration inside the envelope
    a_clamped = env.clamp_accel(float(a_cmd))

    # (b, c) speed-dependent turn-rate cap, then clip the command
    omega_cap = env.max_turn_rate(state.v)
    omega_des = float(omega_cmd)
    if omega_des > omega_cap:
        omega_des = omega_cap
    elif omega_des < -omega_cap:
        omega_des = -omega_cap

    # (d) roll-onset rate limit: the turn rate is a state that chases the
    #     command at no more than omega_dot_max [rad/s^2].
    d_omega_max = abs(env.omega_dot_max) * h
    d_omega = omega_des - state.omega
    if d_omega > d_omega_max:
        d_omega = d_omega_max
    elif d_omega < -d_omega_max:
        d_omega = -d_omega_max
    omega_new = state.omega + d_omega
    # Defensive guard: omega_new always lies between state.omega and omega_des,
    # so its magnitude can never exceed max(|state.omega|, omega_cap). The clip
    # below is therefore a no-op in exact arithmetic; it only removes float
    # round-off and protects against a state that entered out of envelope.
    allowed = omega_cap if omega_cap > abs(state.omega) else abs(state.omega)
    if omega_new > allowed:
        omega_new = allowed
    elif omega_new < -allowed:
        omega_new = -allowed

    # (e) RK4 with omega_new and a_clamped held constant across the step
    s = float(env.wind_susceptibility)
    wx_eff = s * float(wind_xy[0])
    wy_eff = s * float(wind_xy[1])
    x_new, y_new, psi_new, v_new = rk4_unicycle(
        float(state.x),
        float(state.y),
        float(state.psi),
        float(state.v),
        omega_new,
        a_clamped,
        wx_eff,
        wy_eff,
        h,
    )

    # (f) speed envelope; zero the stored accel when a bound binds
    v_clamped = env.clamp_speed(v_new)
    a_stored = a_clamped
    if v_clamped != v_new:
        a_stored = 0.0

    # (g) wrap heading
    return AgentState(
        x=x_new,
        y=y_new,
        v=v_clamped,
        psi=wrap_angle(psi_new),
        omega=omega_new,
        a=a_stored,
    )


# --------------------------------------------------------------------------
# 2) Passive motion -- unpowered ballistic body with quadratic drag (DEBRIS)
# --------------------------------------------------------------------------


def _drag_deriv(y: np.ndarray, k: float, wx: float, wy: float) -> np.ndarray:
    """Derivative of ``[x, y, vx, vy]`` for a body under quadratic air drag.

    Args:
        y: state ``[x m, y m, vx m/s, vy m/s]``.
        k: drag coefficient ``0.5 * rho * Cd*A / mass`` [1/m].
        wx: East wind component [m/s].
        wy: North wind component [m/s].

    Returns:
        ``[vx, vy, ax, ay]`` with accelerations in [m/s^2].
    """
    vx = y[2]
    vy = y[3]
    rx = vx - wx
    ry = vy - wy
    speed_rel = math.sqrt(rx * rx + ry * ry)
    return np.array([vx, vy, -k * speed_rel * rx, -k * speed_rel * ry],
                    dtype=np.float64)


def step_passive(
    state: AgentState,
    env: PerformanceEnvelope,
    wind_xy: Sequence[float],
    dt: float,
) -> AgentState:
    """Advance one UNPOWERED body (DEBRIS) by ``dt``: pure drag toward the air.

    There is no propulsion and no guidance. The only force is quadratic
    aerodynamic drag opposing the AIR-RELATIVE velocity::

        v_vec = ground velocity from (state.v, state.psi)
        v_rel = v_vec - wind_xy
        a_vec = -(0.5 * RHO_AIR * env.drag_area / env.mass) * |v_rel| * v_rel

    For an unpowered body ``state.v`` and ``state.psi`` are GROUND polar
    coordinates (doc convention in :class:`~core.state.AgentState`), so the
    wind is NOT added again through ``wind_susceptibility`` -- it enters only
    through the drag law. Asymptotically ``v_rel -> 0``: the debris becomes
    fully wind-slaved. That is exactly the contrast behaviour the dataset
    wants next to guided traffic, and it is a motion a constant-velocity
    extrapolator systematically mispredicts during the decay transient.

    The equations are integrated with the generic
    :func:`core.integrator.rk4` on the Cartesian state ``[x, y, vx, vy]``,
    then converted back to ``(v, psi)``. The returned ``omega`` and ``a``
    fields are the step-averaged turn rate [rad/s] and longitudinal
    acceleration [m/s^2] actually realised, so that logging and prediction
    consumers see meaningful values for unpowered bodies too.

    Args:
        state: current kinematic state; NEVER mutated.
        env: performance envelope; only ``mass`` [kg] and ``drag_area``
            (Cd*A) [m^2] are used.
        wind_xy: ambient wind ``(wx, wy)`` [m/s], world axes.
        dt: integration step [s].

    Returns:
        A NEW :class:`~core.state.AgentState` at ``t + dt``.

    Raises:
        ValueError: if ``env.mass`` is not strictly positive.
    """
    h = float(dt)
    mass = float(env.mass)
    if mass <= 0.0:
        raise ValueError("step_passive requires env.mass > 0 kg")
    k = 0.5 * RHO_AIR * max(float(env.drag_area), 0.0) / mass

    wx = float(wind_xy[0])
    wy = float(wind_xy[1])

    vx0 = state.v * math.cos(state.psi)
    vy0 = state.v * math.sin(state.psi)
    y0 = np.array([float(state.x), float(state.y), vx0, vy0], dtype=np.float64)

    y1 = rk4(lambda yy: _drag_deriv(yy, k, wx, wy), y0, h)

    x_new = float(y1[0])
    y_new = float(y1[1])
    vx1 = float(y1[2])
    vy1 = float(y1[3])

    v_new = math.hypot(vx1, vy1)
    if v_new > _HEADING_EPS_V:
        psi_new = math.atan2(vy1, vx1)
    else:
        psi_new = state.psi  # heading undefined at (numerically) zero speed

    if h != 0.0:
        omega_eff = angle_diff(psi_new, state.psi) / h
        a_eff = (v_new - state.v) / h
    else:
        omega_eff = state.omega
        a_eff = state.a

    return AgentState(
        x=x_new,
        y=y_new,
        v=v_new,
        psi=wrap_angle(psi_new),
        omega=omega_eff,
        a=a_eff,
    )


def step(
    state: AgentState,
    env: PerformanceEnvelope,
    wind_xy: Sequence[float],
    dt: float,
    omega_cmd: float = 0.0,
    a_cmd: float = 0.0,
) -> AgentState:
    """Dispatch to :func:`step_powered` or :func:`step_passive` by envelope.

    Convenience wrapper for the simulation engine so it does not have to test
    ``env.is_powered`` at every call site. Commands are ignored for unpowered
    bodies (they have no actuators).

    Args:
        state: current kinematic state; NEVER mutated.
        env: performance envelope; ``is_powered`` selects the model.
        wind_xy: ambient wind ``(wx, wy)`` [m/s], world axes.
        dt: integration step [s].
        omega_cmd: commanded turn rate [rad/s] (powered only).
        a_cmd: commanded longitudinal acceleration [m/s^2] (powered only).

    Returns:
        A NEW :class:`~core.state.AgentState` at ``t + dt``.
    """
    if env.is_powered:
        return step_powered(state, omega_cmd, a_cmd, env, wind_xy, dt)
    return step_passive(state, env, wind_xy, dt)


# --------------------------------------------------------------------------
# 3) Guidance helpers used by the intent controllers
# --------------------------------------------------------------------------


def heading_error(psi: float, psi_desired: float) -> float:
    """Signed heading error [rad] to be nulled, wrapped to (-pi, pi].

    Defined as ``psi_desired - psi``, so a POSITIVE error means the desired
    heading lies CCW of the current one and the agent must turn LEFT
    (positive omega). This sign convention matches
    ``omega > 0 => CCW`` used everywhere in dataGen.

    Args:
        psi: current heading [rad].
        psi_desired: desired heading [rad].

    Returns:
        Wrapped error [rad] in (-pi, pi].
    """
    return angle_diff(float(psi_desired), float(psi))


def turn_rate_to_heading(
    psi: float,
    psi_des: float,
    env: PerformanceEnvelope,
    v: float,
    gain: float = 1.0,
) -> float:
    """Proportional heading-hold law, saturated by the envelope.

    ``omega_cmd = clip(gain * heading_error, +/- env.max_turn_rate(v))``.

    A gain of 1.0 [1/s] means "null a 1 rad error in about 1 s if the airframe
    allows it"; the saturation is what makes the airframe, not the gain,
    decide the actual turn.

    Args:
        psi: current heading [rad].
        psi_des: desired heading [rad].
        env: performance envelope of this platform.
        v: current airspeed [m/s] (the cap is speed dependent).
        gain: proportional gain [1/s].

    Returns:
        Commanded turn rate [rad/s], within the envelope cap.
    """
    err = heading_error(psi, psi_des)
    return env.clamp_turn_rate(float(gain) * err, v)


def cross_track_guidance(
    pos: Sequence[float],
    psi: float,
    path_p0: Sequence[float],
    path_p1: Sequence[float],
    env: PerformanceEnvelope,
    v: float,
    k_ct: float = 0.02,
    k_hdg: float = 1.0,
) -> float:
    """Line-following guidance (corridor centreline tracking).

    The commanded heading is the path heading plus an intercept angle
    proportional to the signed cross-track error, saturated at +/- 90 deg so
    that a far-off agent flies perpendicular to the line rather than
    reversing::

        u        = unit vector from path_p0 to path_p1
        e_ct     = u x (pos - path_p0)      # >0 when pos is LEFT of the path
        psi_cmd  = atan2(u) + clip(-k_ct * e_ct, -pi/2, +pi/2)
        omega    = clip(k_hdg * (psi_cmd - psi), +/- max_turn_rate(v))

    Args:
        pos: current position ``(x, y)`` [m].
        psi: current heading [rad].
        path_p0: first point defining the line [m].
        path_p1: second point defining the line [m]; the direction of travel
            is ``p0 -> p1``.
        env: performance envelope of this platform.
        v: current airspeed [m/s].
        k_ct: cross-track gain [rad/m]; 0.02 gives a 45 deg intercept at about
            39 m off track.
        k_hdg: heading gain [1/s].

    Returns:
        Commanded turn rate [rad/s], within the envelope cap. If the two path
        points coincide the law degenerates to "hold current heading" and 0.0
        is returned.
    """
    x0 = float(path_p0[0])
    y0 = float(path_p0[1])
    ux = float(path_p1[0]) - x0
    uy = float(path_p1[1]) - y0
    norm = math.hypot(ux, uy)
    if norm <= 0.0:
        return 0.0
    ux /= norm
    uy /= norm

    dx = float(pos[0]) - x0
    dy = float(pos[1]) - y0
    e_ct = ux * dy - uy * dx  # signed cross-track error [m], + = left of path

    intercept = -float(k_ct) * e_ct
    if intercept > 0.5 * math.pi:
        intercept = 0.5 * math.pi
    elif intercept < -0.5 * math.pi:
        intercept = -0.5 * math.pi

    psi_path = math.atan2(uy, ux)
    psi_cmd = wrap_angle(psi_path + intercept)
    return turn_rate_to_heading(psi, psi_cmd, env, v, gain=k_hdg)


def orbit_guidance(
    pos: Sequence[float],
    psi: float,
    centre: Sequence[float],
    radius: float,
    direction: int,
    env: PerformanceEnvelope,
    v: float,
    k_orbit: float = 1.0,
    k_hdg: float = 1.0,
) -> float:
    """Loiter-circle (orbit) guidance with a curvature feed-forward term.

    Vector-field law: the desired heading is tangential on the circle and
    turns progressively radially inward/outward with the radial error::

        phi     = bearing of (pos - centre)
        psi_des = phi + direction * (pi/2 + atan(k_orbit * (r - radius)/radius))
        omega   = clip(direction * v / radius + k_hdg * (psi_des - psi), cap)

    The feed-forward ``direction * v / radius`` is the exact turn rate of a
    steady orbit, so a converged agent needs no error term at all and the
    circle does not drift outward.

    Args:
        pos: current position ``(x, y)`` [m].
        psi: current heading [rad].
        centre: orbit centre ``(x, y)`` [m].
        radius: orbit radius [m], must be > 0.
        direction: ``+1`` for CCW (left-hand orbit), ``-1`` for CW.
        env: performance envelope of this platform.
        v: current airspeed [m/s].
        k_orbit: radial convergence gain [-]; larger converges harder.
        k_hdg: heading gain [1/s].

    Returns:
        Commanded turn rate [rad/s], within the envelope cap.

    Raises:
        ValueError: if ``radius <= 0`` or ``direction`` is not +1/-1.
    """
    r_ref = float(radius)
    if r_ref <= 0.0:
        raise ValueError("orbit_guidance requires radius > 0 m")
    d = 1 if int(direction) >= 0 else -1
    if int(direction) not in (-1, 1):
        raise ValueError("orbit_guidance direction must be +1 (CCW) or -1 (CW)")

    dx = float(pos[0]) - float(centre[0])
    dy = float(pos[1]) - float(centre[1])
    r = math.hypot(dx, dy)
    if r <= 1e-9:
        # Exactly at the centre: the bearing is undefined. Fly out along the
        # current heading; the law re-engages as soon as r > 0.
        return env.clamp_turn_rate(0.0, v)

    phi = math.atan2(dy, dx)
    radial_term = math.atan(float(k_orbit) * (r - r_ref) / r_ref)
    psi_des = wrap_angle(phi + d * (0.5 * math.pi + radial_term))

    omega_ff = d * float(v) / r_ref
    omega_cmd = omega_ff + float(k_hdg) * heading_error(psi, psi_des)
    return env.clamp_turn_rate(omega_cmd, v)


def min_turn_radius(v: float, env: PerformanceEnvelope) -> float:
    """Minimum achievable turn radius [m] at airspeed ``v`` [m/s].

    ``r = v / env.max_turn_rate(v)``. Returns ``inf`` when the effective cap
    is zero (a body that cannot turn at all).

    Args:
        v: airspeed [m/s].
        env: performance envelope of this platform.

    Returns:
        Minimum turn radius [m], or ``float('inf')``.
    """
    cap = env.max_turn_rate(v)
    if cap <= 0.0:
        return float("inf")
    return max(float(v), 0.0) / cap


# --------------------------------------------------------------------------
# 4) The constant-velocity baseline -- ONE definition for the whole codebase
# --------------------------------------------------------------------------


def predict_cv(
    pos: np.ndarray,
    vel: np.ndarray,
    horizons_s: np.ndarray,
) -> np.ndarray:
    """Constant-velocity extrapolation ``p(t) = p0 + v0 * t``.

    This is the trivial baseline the whole paper is measured against
    (doc 3.1: if the simulator produces trajectories this predicts well, the
    contribution collapses). It is defined here, once, so that
    ``analysis/cv_baseline.py``, the GUI overlay and any diagnostic all use
    literally the same function.

    Args:
        pos: position ``(..., 2)`` [m], last axis is ``(x, y)``.
        vel: GROUND velocity ``(..., 2)`` [m/s], same leading shape as ``pos``.
        horizons_s: prediction horizons ``(H,)`` [s], typically
            ``np.arange(1, PRED_STEPS + 1) * LOG_DT``.

    Returns:
        Predicted positions ``(..., H, 2)`` [m] as float64. Callers that store
        the result cast to float32 at write time (storage/schema.py contract).

    Raises:
        ValueError: if the last axis of ``pos`` or ``vel`` is not 2, or if
            ``horizons_s`` is not one-dimensional.
    """
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    h = np.asarray(horizons_s, dtype=np.float64)
    if p.shape[-1] != 2 or v.shape[-1] != 2:
        raise ValueError("predict_cv expects pos/vel with last axis of size 2")
    if h.ndim != 1:
        raise ValueError("predict_cv expects horizons_s with shape (H,)")
    return p[..., None, :] + v[..., None, :] * h[:, None]
