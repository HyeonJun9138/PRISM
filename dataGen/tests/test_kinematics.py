"""Physics tests for :mod:`core.kinematics` and :mod:`core.integrator`.

These are the checks that protect the central claim of the paper: the
trajectories must be produced by a real, envelope-limited airframe model, not
by a heading that snaps instantaneously. Every assertion below has a physical
meaning stated in its docstring.

Run directly::

    python $PRISM_ROOT/dataGen/tests/test_kinematics.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.constants import G, SIM_DT  # noqa: E402
from core.integrator import (  # noqa: E402
    STATE_PSI,
    STATE_V,
    STATE_X,
    STATE_Y,
    euler,
    integrate_substeps,
    rk4,
    rk4_unicycle,
    semi_implicit_euler,
)
from core.kinematics import (  # noqa: E402
    RHO_AIR,
    cross_track_guidance,
    heading_error,
    min_turn_radius,
    orbit_guidance,
    predict_cv,
    step,
    step_passive,
    step_powered,
    turn_rate_to_heading,
)
from core.state import AgentState, PerformanceEnvelope, wrap_angle  # noqa: E402

# --------------------------------------------------------------------------
# Test fixtures: envelopes built locally so this file does not depend on
# agents/platforms.py (owned by another module, written in parallel).
# --------------------------------------------------------------------------


def fixed_wing_env() -> PerformanceEnvelope:
    """Fixed-wing UAS: 20-40 m/s, bank limited to 30 deg (doc 3.2 (3))."""
    return PerformanceEnvelope(
        v_min=20.0,
        v_max=40.0,
        a_max=2.0,
        a_min=-3.0,
        omega_max=1.0,          # deliberately non-binding: bank limit governs
        omega_dot_max=0.20,
        bank_max=30.0 * math.pi / 180.0,
        bank_limited=True,
        lateral_accel_max=0.0,
        can_hover=False,
        is_powered=True,
        wind_susceptibility=0.6,
        mass=8.0,
        drag_area=0.15,
    )


def evtol_env() -> PerformanceEnvelope:
    """eVTOL cruise: 50-80 m/s, gentle 25 deg bank (doc 3.2 (3))."""
    return PerformanceEnvelope(
        v_min=30.0,
        v_max=80.0,
        a_max=1.5,
        a_min=-2.0,
        omega_max=1.0,          # non-binding on purpose
        omega_dot_max=0.10,
        bank_max=25.0 * math.pi / 180.0,
        bank_limited=True,
        lateral_accel_max=0.0,
        can_hover=False,
        is_powered=True,
        wind_susceptibility=0.4,
        mass=1200.0,
        drag_area=2.5,
    )


def multirotor_env() -> PerformanceEnvelope:
    """Small multirotor: 0-20 m/s, tilt limited, hover capable."""
    return PerformanceEnvelope(
        v_min=0.0,
        v_max=20.0,
        a_max=4.0,
        a_min=-4.0,
        omega_max=1.5,
        omega_dot_max=2.0,
        bank_max=0.0,
        bank_limited=False,
        lateral_accel_max=6.0,
        can_hover=True,
        is_powered=True,
        wind_susceptibility=0.9,
        mass=2.5,
        drag_area=0.12,
    )


def debris_env() -> PerformanceEnvelope:
    """Unpowered light debris: no thrust, fully wind-slaved asymptotically."""
    return PerformanceEnvelope(
        v_min=0.0,
        v_max=60.0,
        a_max=0.0,
        a_min=0.0,
        omega_max=0.0,
        omega_dot_max=0.0,
        bank_max=0.0,
        bank_limited=False,
        lateral_accel_max=0.0,
        can_hover=True,
        is_powered=False,
        wind_susceptibility=1.0,
        mass=0.5,
        drag_area=0.20,
    )


def _fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit.

    Solves ``x^2 + y^2 = 2*a*x + 2*b*y + c`` and returns
    ``(cx [m], cy [m], r [m])``.
    """
    a_mat = np.stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)], axis=1)
    rhs = xs * xs + ys * ys
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0.0))


# --------------------------------------------------------------------------
# 1. Constant velocity
# --------------------------------------------------------------------------


def test_cv_travels_exactly_v_times_t() -> None:
    """Zero commands for 10 s must displace exactly v*10 m along the heading.

    This is the null hypothesis of the whole project: if the integrator itself
    drifts, every downstream CV-vs-model comparison is meaningless.
    """
    env = fixed_wing_env()
    psi0 = 0.7
    v0 = 30.0
    st = AgentState(x=100.0, y=-50.0, v=v0, psi=psi0, omega=0.0, a=0.0)
    t_total = 10.0
    n = int(round(t_total / SIM_DT))
    for _ in range(n):
        st = step_powered(st, 0.0, 0.0, env, (0.0, 0.0), SIM_DT)

    exp_x = 100.0 + v0 * t_total * math.cos(psi0)
    exp_y = -50.0 + v0 * t_total * math.sin(psi0)
    assert abs(st.x - exp_x) < 1e-6, (st.x, exp_x)
    assert abs(st.y - exp_y) < 1e-6, (st.y, exp_y)
    assert abs(st.v - v0) < 1e-12
    assert abs(wrap_angle(st.psi - psi0)) < 1e-12
    assert st.omega == 0.0


def test_cv_with_wind_adds_susceptibility_scaled_drift() -> None:
    """Wind advects the body by ``susceptibility * wind * t``, no more."""
    env = fixed_wing_env()
    wind = (5.0, -3.0)
    st = AgentState(x=0.0, y=0.0, v=25.0, psi=0.0)
    t_total = 20.0
    for _ in range(int(round(t_total / SIM_DT))):
        st = step_powered(st, 0.0, 0.0, env, wind, SIM_DT)
    s = env.wind_susceptibility
    assert abs(st.x - (25.0 * t_total + s * wind[0] * t_total)) < 1e-6
    assert abs(st.y - (s * wind[1] * t_total)) < 1e-6


# --------------------------------------------------------------------------
# 2. Sustained max-rate turn -> circle of radius v/omega
# --------------------------------------------------------------------------


def test_max_rate_turn_is_a_circle_of_radius_v_over_omega() -> None:
    """A sustained saturated turn traces a circle of radius v/omega (< 1 %).

    The transient at the start (roll onset) is discarded; only the steady
    portion is fitted.
    """
    env = fixed_wing_env()
    v0 = 30.0
    st = AgentState(x=0.0, y=0.0, v=v0, psi=0.0)
    omega_steady = env.max_turn_rate(v0)
    assert omega_steady > 0.0

    xs: list[float] = []
    ys: list[float] = []
    t = 0.0
    t_total = 60.0
    while t < t_total:
        st = step_powered(st, 10.0, 0.0, env, (0.0, 0.0), SIM_DT)  # saturating
        t += SIM_DT
        if t > 5.0:  # after roll onset (0.15/0.20 = 0.75 s) plus margin
            xs.append(st.x)
            ys.append(st.y)

    assert abs(st.omega - omega_steady) < 1e-12, (st.omega, omega_steady)
    assert abs(st.v - v0) < 1e-9  # a coordinated turn does not change speed

    r_expected = v0 / omega_steady
    _, _, r_fit = _fit_circle(np.asarray(xs), np.asarray(ys))
    rel = abs(r_fit - r_expected) / r_expected
    assert rel < 0.01, (r_fit, r_expected, rel)

    # And the analytic minimum radius helper agrees with the flown circle.
    assert abs(min_turn_radius(v0, env) - r_expected) < 1e-9

    # Positive omega must be a LEFT (CCW) turn: heading increased.
    st2 = AgentState(x=0.0, y=0.0, v=v0, psi=0.0, omega=omega_steady)
    st2 = step_powered(st2, omega_steady, 0.0, env, (0.0, 0.0), SIM_DT)
    assert st2.psi > 0.0


# --------------------------------------------------------------------------
# 3. Speed-dependent (bank-limited) turn-rate cap
# --------------------------------------------------------------------------


def test_bank_limited_cap_shrinks_with_speed() -> None:
    """An eVTOL at 80 m/s must turn strictly slower than the same one at 40.

    Analytically omega_cap = G*tan(bank_max)/v, so doubling the speed halves
    the cap; the test checks both the analytic cap and the flown turn rate.
    """
    env = evtol_env()
    cap_40 = env.max_turn_rate(40.0)
    cap_80 = env.max_turn_rate(80.0)
    assert cap_80 < cap_40
    assert abs(cap_40 / cap_80 - 2.0) < 1e-9
    assert abs(cap_40 - G * math.tan(env.bank_max) / 40.0) < 1e-12

    # Radius grows with the SQUARE of speed for a bank-limited platform.
    assert abs(min_turn_radius(80.0, env) / min_turn_radius(40.0, env) - 4.0) < 1e-9

    # Flown check: same saturating command, different speed -> different rate.
    def flown_rate(v: float) -> float:
        st = AgentState(x=0.0, y=0.0, v=v, psi=0.0)
        for _ in range(int(round(20.0 / SIM_DT))):
            st = step_powered(st, 5.0, 0.0, env, (0.0, 0.0), SIM_DT)
        return st.omega

    r40 = flown_rate(40.0)
    r80 = flown_rate(80.0)
    assert r80 < r40
    assert abs(r40 - cap_40) < 1e-12
    assert abs(r80 - cap_80) < 1e-12

    # A tilt-limited multirotor shows the same monotonic behaviour, with the
    # absolute omega_max binding at low speed.
    mr = multirotor_env()
    assert mr.max_turn_rate(2.0) >= mr.max_turn_rate(10.0)
    assert mr.max_turn_rate(1.0) == mr.omega_max  # 6.0/1.0 = 6.0 -> clipped
    assert abs(mr.max_turn_rate(12.0) - 6.0 / 12.0) < 1e-12


# --------------------------------------------------------------------------
# 4. Roll onset: omega_dot_max really limits the turn-rate build-up
# --------------------------------------------------------------------------


def test_omega_dot_max_limits_turn_onset() -> None:
    """After one dt the turn rate cannot exceed ``omega_dot_max * dt``.

    doc 7.2: without this, headings snap and reviewers see zigzags instead of
    aircraft.
    """
    env = fixed_wing_env()
    st = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0, omega=0.0)
    st1 = step_powered(st, 10.0, 0.0, env, (0.0, 0.0), SIM_DT)
    limit = env.omega_dot_max * SIM_DT
    assert abs(st1.omega) <= limit + 1e-15, (st1.omega, limit)
    assert abs(st1.omega - limit) < 1e-15  # it should saturate the rate limit

    # Reversal is limited symmetrically.
    st2 = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0, omega=0.10)
    st3 = step_powered(st2, -10.0, 0.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st3.omega - (0.10 - limit)) < 1e-15

    # The build-up is monotone and takes the expected number of steps to reach
    # the cap: cap / (omega_dot_max) seconds.
    st_n = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0, omega=0.0)
    cap = env.max_turn_rate(30.0)
    n_expected = math.ceil(cap / (env.omega_dot_max * SIM_DT))
    prev = 0.0
    for i in range(n_expected):
        st_n = step_powered(st_n, 10.0, 0.0, env, (0.0, 0.0), SIM_DT)
        assert st_n.omega >= prev - 1e-15
        assert st_n.omega <= min((i + 1) * limit, cap) + 1e-12
        prev = st_n.omega
    assert abs(st_n.omega - cap) < 1e-12

    # A heading step therefore lags a first-order response; over the first dt
    # the heading change is bounded by omega_new*dt (no instantaneous snap).
    st4 = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0, omega=0.0)
    st5 = step_powered(st4, 10.0, 0.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st5.psi) <= limit * SIM_DT + 1e-15


# --------------------------------------------------------------------------
# 5. Acceleration envelope
# --------------------------------------------------------------------------


def test_accel_clamped_and_zeroed_at_speed_bound() -> None:
    """Accel is clipped to the envelope, and zeroed once a speed bound binds."""
    env = fixed_wing_env()
    st = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0, a=0.0)
    st1 = step_powered(st, 0.0, 99.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st1.a - env.a_max) < 1e-15
    assert abs(st1.v - (30.0 + env.a_max * SIM_DT)) < 1e-12

    # Push to v_max and check the stored accel is dropped.
    st_n = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0)
    for _ in range(int(round(30.0 / SIM_DT))):
        st_n = step_powered(st_n, 0.0, 99.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st_n.v - env.v_max) < 1e-12
    assert st_n.a == 0.0

    # And symmetrically at v_min with full deceleration.
    st_m = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0)
    for _ in range(int(round(30.0 / SIM_DT))):
        st_m = step_powered(st_m, 0.0, -99.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st_m.v - env.v_min) < 1e-12
    assert st_m.a == 0.0


def test_step_returns_new_object_and_never_mutates() -> None:
    """``step_powered`` is pure: the input state object is untouched."""
    env = multirotor_env()
    st = AgentState(x=1.0, y=2.0, v=10.0, psi=0.3, omega=0.05, a=0.1)
    before = st.to_dict()
    out = step_powered(st, 1.0, 1.0, env, (2.0, 1.0), SIM_DT)
    assert out is not st
    assert st.to_dict() == before


def test_psi_stays_wrapped() -> None:
    """Heading is always returned inside (-pi, pi] no matter how many turns."""
    env = multirotor_env()
    st = AgentState(x=0.0, y=0.0, v=10.0, psi=3.0, omega=0.0)
    for _ in range(int(round(120.0 / SIM_DT))):
        st = step_powered(st, 0.6, 0.0, env, (0.0, 0.0), SIM_DT)
        assert -math.pi < st.psi <= math.pi


# --------------------------------------------------------------------------
# 6. Passive (DEBRIS) dynamics
# --------------------------------------------------------------------------


def test_debris_asymptotes_to_the_wind_vector() -> None:
    """20 m/s debris in a 10 m/s wind must decay onto the wind vector.

    Quadratic drag gives |v_rel|(t) = v_rel0 / (1 + k*v_rel0*t) with
    k = 0.5*rho*Cd*A/m, so the relative speed decays monotonically to zero and
    the body becomes fully wind-slaved. That is the contrast behaviour the
    dataset wants next to guided traffic.
    """
    env = debris_env()
    wind = (10.0, 0.0)
    st = AgentState(x=0.0, y=0.0, v=20.0, psi=0.0)  # 20 m/s along +X

    k = 0.5 * RHO_AIR * env.drag_area / env.mass
    v_rel0 = 20.0 - 10.0

    prev_rel = v_rel0
    t = 0.0
    while t < 60.0:
        st = step_passive(st, env, wind, SIM_DT)
        t += SIM_DT
        vx = st.v * math.cos(st.psi)
        vy = st.v * math.sin(st.psi)
        rel = math.hypot(vx - wind[0], vy - wind[1])
        assert rel <= prev_rel + 1e-12          # monotone decay, no overshoot
        prev_rel = rel

    vx = st.v * math.cos(st.psi)
    vy = st.v * math.sin(st.psi)
    assert abs(vy) < 1e-9                        # motion stays 1-D
    assert abs(vx - wind[0]) < 0.2, vx           # asymptote onto the wind
    assert vx > wind[0]                          # approached from above

    # Closed-form check of the decay law at t = 60 s.
    rel_analytic = v_rel0 / (1.0 + k * v_rel0 * 60.0)
    assert abs((vx - wind[0]) - rel_analytic) < 1e-3, (vx - wind[0], rel_analytic)


def test_debris_crosswind_turns_toward_the_wind() -> None:
    """Debris launched across the wind is progressively turned downwind."""
    env = debris_env()
    wind = (12.0, 0.0)
    st = AgentState(x=0.0, y=0.0, v=15.0, psi=0.5 * math.pi)  # due North
    for _ in range(int(round(90.0 / SIM_DT))):
        st = step_passive(st, env, wind, SIM_DT)
    vx = st.v * math.cos(st.psi)
    vy = st.v * math.sin(st.psi)
    assert abs(vx - wind[0]) < 0.5, vx
    assert abs(vy) < 0.5, vy
    assert abs(st.psi) < 0.05                    # heading swung round to East


def test_step_dispatch_matches_explicit_calls() -> None:
    """``step`` dispatches on ``is_powered`` and matches the explicit calls."""
    penv = fixed_wing_env()
    st = AgentState(x=0.0, y=0.0, v=30.0, psi=0.2)
    assert step(st, penv, (1.0, 2.0), SIM_DT, 0.1, 0.5).to_dict() == \
        step_powered(st, 0.1, 0.5, penv, (1.0, 2.0), SIM_DT).to_dict()

    denv = debris_env()
    sd = AgentState(x=0.0, y=0.0, v=18.0, psi=1.0)
    assert step(sd, denv, (4.0, 1.0), SIM_DT, 0.9, 0.9).to_dict() == \
        step_passive(sd, denv, (4.0, 1.0), SIM_DT).to_dict()


# --------------------------------------------------------------------------
# 7. Guidance laws
# --------------------------------------------------------------------------


def test_heading_error_sign_and_wrapping() -> None:
    """Positive error means "turn left"; the error is wrapped the short way."""
    assert abs(heading_error(0.0, 0.5) - 0.5) < 1e-15
    assert abs(heading_error(0.5, 0.0) + 0.5) < 1e-15
    # Short way round: from +3.0 rad to -3.0 rad is +0.283 rad (through pi).
    e = heading_error(3.0, -3.0)
    assert e > 0.0 and abs(e - (2.0 * math.pi - 6.0)) < 1e-12

    env = fixed_wing_env()
    om = turn_rate_to_heading(0.0, 2.0, env, 30.0, gain=1.0)
    assert abs(om - env.max_turn_rate(30.0)) < 1e-15  # saturated left
    om2 = turn_rate_to_heading(0.0, -2.0, env, 30.0, gain=1.0)
    assert abs(om2 + env.max_turn_rate(30.0)) < 1e-15


def test_turn_rate_to_heading_converges() -> None:
    """Heading hold drives psi onto the commanded value and holds it."""
    env = multirotor_env()
    st = AgentState(x=0.0, y=0.0, v=12.0, psi=0.0)
    psi_des = -2.0
    for _ in range(int(round(30.0 / SIM_DT))):
        om = turn_rate_to_heading(st.psi, psi_des, env, st.v, gain=1.0)
        st = step_powered(st, om, 0.0, env, (0.0, 0.0), SIM_DT)
    assert abs(heading_error(st.psi, psi_des)) < 1e-3
    assert abs(st.omega) < 1e-3


def test_cross_track_guidance_captures_the_line() -> None:
    """An agent offset from a corridor centreline converges onto it."""
    env = fixed_wing_env()
    p0 = (0.0, 0.0)
    p1 = (5000.0, 0.0)                       # centreline along +X
    st = AgentState(x=0.0, y=300.0, v=30.0, psi=0.0)  # 300 m left of the line
    for _ in range(int(round(120.0 / SIM_DT))):
        om = cross_track_guidance(
            (st.x, st.y), st.psi, p0, p1, env, st.v, k_ct=0.02, k_hdg=1.0
        )
        st = step_powered(st, om, 0.0, env, (0.0, 0.0), SIM_DT)
    assert abs(st.y) < 5.0, st.y             # captured the centreline
    assert abs(st.psi) < 0.05                # and flying along it
    assert st.x > 3000.0                     # while still making progress

    # Degenerate path (p0 == p1) must not raise or produce NaN.
    assert cross_track_guidance((1.0, 1.0), 0.0, p0, p0, env, 30.0) == 0.0

    # Sign check: being LEFT of the path commands a RIGHT (negative) turn.
    om_left = cross_track_guidance((0.0, 100.0), 0.0, p0, p1, env, 30.0)
    om_right = cross_track_guidance((0.0, -100.0), 0.0, p0, p1, env, 30.0)
    assert om_left < 0.0 < om_right


def test_orbit_guidance_holds_a_loiter_circle() -> None:
    """Orbit guidance converges to the commanded radius, both directions."""
    env = multirotor_env()
    radius = 120.0
    centre = (0.0, 0.0)
    for direction in (1, -1):
        st = AgentState(x=200.0, y=0.0, v=12.0, psi=0.5 * math.pi * direction)
        radii: list[float] = []
        t = 0.0
        while t < 200.0:
            om = orbit_guidance(
                (st.x, st.y), st.psi, centre, radius, direction, env, st.v
            )
            st = step_powered(st, om, 0.0, env, (0.0, 0.0), SIM_DT)
            t += SIM_DT
            if t > 60.0:
                radii.append(math.hypot(st.x, st.y))
        arr = np.asarray(radii)
        assert abs(float(arr.mean()) - radius) < 5.0, float(arr.mean())
        assert float(arr.std()) < 5.0
        # Circulation sense matches the requested direction.
        assert math.copysign(1.0, st.omega) == float(direction)

    # The steady-state feed-forward is exactly the circular rate.
    st_on = AgentState(x=radius, y=0.0, v=12.0, psi=0.5 * math.pi)
    om = orbit_guidance((st_on.x, st_on.y), st_on.psi, centre, radius, 1,
                        env, st_on.v)
    assert abs(om - 12.0 / radius) < 1e-12


# --------------------------------------------------------------------------
# 8. Constant-velocity baseline
# --------------------------------------------------------------------------


def test_predict_cv_shapes_and_values() -> None:
    """predict_cv is p0 + v0*t, broadcasting over any leading agent axis."""
    horizons = np.arange(1, 31) * 0.5
    out = predict_cv(np.array([10.0, -5.0]), np.array([3.0, 4.0]), horizons)
    assert out.shape == (30, 2)
    assert abs(out[0, 0] - (10.0 + 3.0 * 0.5)) < 1e-12
    assert abs(out[-1, 1] - (-5.0 + 4.0 * 15.0)) < 1e-12

    pos = np.zeros((7, 2))
    vel = np.tile(np.array([1.0, 0.0]), (7, 1))
    batch = predict_cv(pos, vel, horizons)
    assert batch.shape == (7, 30, 2)
    assert abs(batch[3, 9, 0] - 5.0) < 1e-12

    # A CV agent's own future must equal the CV prediction exactly.
    env = fixed_wing_env()
    st = AgentState(x=0.0, y=0.0, v=30.0, psi=0.4)
    vel0 = np.array(st.ground_velocity((0.0, 0.0), env.wind_susceptibility))
    for _ in range(int(round(15.0 / SIM_DT))):
        st = step_powered(st, 0.0, 0.0, env, (0.0, 0.0), SIM_DT)
    pred = predict_cv(np.array([0.0, 0.0]), vel0, np.array([15.0]))
    assert abs(pred[0, 0] - st.x) < 1e-6
    assert abs(pred[0, 1] - st.y) < 1e-6


# --------------------------------------------------------------------------
# 9. Integrator module itself
# --------------------------------------------------------------------------


def test_scalar_rk4_matches_generic_rk4() -> None:
    """The hot-path scalar unicycle RK4 equals the generic array RK4."""
    x, y, psi, v = 3.0, -7.0, 0.9, 28.0
    omega, a, wx, wy, dt = 0.11, -0.4, 1.5, -0.8, SIM_DT

    def deriv(s: np.ndarray) -> np.ndarray:
        return np.array(
            [s[STATE_V] * math.cos(s[STATE_PSI]) + wx,
             s[STATE_V] * math.sin(s[STATE_PSI]) + wy,
             omega,
             a],
            dtype=np.float64,
        )

    generic = rk4(deriv, np.array([x, y, psi, v]), dt)
    scalar = rk4_unicycle(x, y, psi, v, omega, a, wx, wy, dt)
    for i in range(4):
        assert abs(generic[i] - scalar[i]) < 1e-12, (i, generic[i], scalar[i])
    assert generic[STATE_X] == generic[0] and generic[STATE_Y] == generic[1]


def test_rk4_is_fourth_order_on_the_unicycle() -> None:
    """Halving dt must cut the RK4 arc-position error by roughly 2^4 = 16.

    Compared against the exact coordinated-turn solution after 4 s, so the
    check measures true local accuracy rather than a symmetric full-circle
    cancellation.
    """
    v, omega = 25.0, 0.5
    t_end = 4.0
    r = v / omega
    exact_x = r * math.sin(omega * t_end)
    exact_y = r * (1.0 - math.cos(omega * t_end))

    def arc_error(n_steps: int) -> float:
        """Position error [m] at t = 4 s using ``n_steps`` uniform RK4 steps."""
        dt = t_end / n_steps
        x = y = psi = 0.0
        for _ in range(n_steps):
            x, y, psi, _ = rk4_unicycle(x, y, psi, v, omega, 0.0, 0.0, 0.0, dt)
        return math.hypot(x - exact_x, y - exact_y)

    e1 = arc_error(4)
    e2 = arc_error(8)
    e3 = arc_error(16)
    assert e3 < e2 < e1, (e1, e2, e3)
    assert 12.0 < e1 / e2 < 20.0, (e1, e2)       # 2^4 = 16
    assert 12.0 < e2 / e3 < 20.0, (e2, e3)
    # At the production step the integrator is far below the sensor floor.
    e_sim = arc_error(int(round(t_end / SIM_DT)))
    assert e_sim < 1e-7, e_sim


def test_semi_implicit_euler_beats_forward_euler_on_a_circle() -> None:
    """Symplectic Euler must not spiral outward the way forward Euler does."""
    v, omega = 20.0, 0.3
    dt = 0.1
    n = int(round((2.0 * math.pi / omega) / dt))

    def deriv(s: np.ndarray) -> np.ndarray:
        return np.array(
            [s[3] * math.cos(s[2]), s[3] * math.sin(s[2]), omega, 0.0],
            dtype=np.float64,
        )

    y_si = np.array([0.0, 0.0, 0.0, v])
    y_fe = y_si.copy()
    for _ in range(n):
        y_si = semi_implicit_euler(deriv, y_si, dt)
        y_fe = euler(deriv, y_fe, dt)
    err_si = math.hypot(y_si[0], y_si[1])
    err_fe = math.hypot(y_fe[0], y_fe[1])
    assert err_si < err_fe, (err_si, err_fe)
    assert abs(y_si[2] - omega * n * dt) < 1e-9
    assert abs(y_si[3] - v) < 1e-12


def test_integrate_substeps_refines() -> None:
    """Sub-stepping a fixed interval converges toward the exact solution."""
    v, omega = 30.0, 0.25

    def deriv(s: np.ndarray) -> np.ndarray:
        return np.array(
            [s[3] * math.cos(s[2]), s[3] * math.sin(s[2]), omega, 0.0],
            dtype=np.float64,
        )

    y0 = np.array([0.0, 0.0, 0.0, v])
    total = 4.0
    coarse = integrate_substeps(deriv, y0, total, 1)
    fine = integrate_substeps(deriv, y0, total, 400)

    # Exact coordinated-turn solution from the origin heading +X.
    r = v / omega
    exact = np.array([r * math.sin(omega * total),
                      r * (1.0 - math.cos(omega * total))])
    e_coarse = math.hypot(coarse[0] - exact[0], coarse[1] - exact[1])
    e_fine = math.hypot(fine[0] - exact[0], fine[1] - exact[1])
    assert e_fine < e_coarse
    assert e_fine < 1e-6

    try:
        integrate_substeps(deriv, y0, total, 0)
        raise AssertionError("n_substeps=0 should raise ValueError")
    except ValueError:
        pass


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def main() -> int:
    """Run every ``test_*`` function in this module. Returns a process code."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - test harness
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
