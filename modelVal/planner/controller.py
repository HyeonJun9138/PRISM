"""PlannerPilot: the Frenet planner flying the modelVal ownship.

Plugs into SimulationEngine's ownship_controller hook (the same seam the
keyboard uses): ``command(t, state, envelope, ctx) -> (omega_cmd, a_cmd)``.

Two cadences:
  * replan()  -- 2 Hz, called from LiveSession._on_logged_frame right after
                 the predictor ran, so the freshest risk picture is used;
  * command() -- 20 Hz (every SIM_DT engine tick), cheap pure-pursuit
                 tracking of the active plan within the eVTOL envelope.

The reference path is the scenario world's corridor centerline (straight,
kappa == 0). The planner corridor width, speed grid, and dynamic limits are
derived from the actual world + ownship envelope at bind() time, and are
IDENTICAL across risk arms -- only the RiskProvider differs.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import numpy as np

from core.constants import NMAC_RADIUS_M
from core.kinematics import heading_error, turn_rate_to_heading

from planner.frenet import (Candidate, FrenetState, PlannerConfig,
                            PlanResult, ReferencePath, eval_trajectory_at,
                            plan_trajectory)
from planner.risk import BaseRisk, make_risk

TRACK_LOOKAHEAD_S = 0.6     # pursue the plan this far ahead of "now"
SPEED_TAU_S = 1.2           # accel command time constant
HEADING_GAIN = 1.3          # turn_rate_to_heading gain (snappy tracking;
                            # the envelope cap still has the final say)
PLAN_STALE_S = 2.5          # no replan for this long -> corridor-hold fallback
WIRE_DECIMATE = 4           # plan polyline decimation for the UI


class PlannerPilot:
    """Ownship controller driven by the ported Frenet planner."""

    def __init__(self, risk: Optional[BaseRisk] = None) -> None:
        self.lock = threading.Lock()
        self.risk: BaseRisk = risk or make_risk("none")
        self.path: Optional[ReferencePath] = None
        self.cfg: Optional[PlannerConfig] = None
        self.plan: Optional[Candidate] = None
        self.plan_valid_count = 0
        self.plan_total = 0
        self.plan_sim_t = 0.0
        self.plan_ms = 0.0
        self.plan_ms_max = 0.0
        self.replans = 0
        self._last_result: Optional[PlanResult] = None

    # ------------------------------------------------------------- binding
    def bind(self, world: Any, envelope: Any,
             ownship_track: tuple[float, float, float, float]) -> None:
        """(Re)attach to a freshly built scenario. Clears any active plan."""
        x0, y0, vgx, vgy = ownship_track
        v_des = math.hypot(vgx, vgy)

        corridors = getattr(world, "corridors", None) or []
        if corridors:
            corr = corridors[0]
            p0 = np.asarray(corr.p0, dtype=np.float64)
            p1 = np.asarray(corr.p1, dtype=np.float64)
            width = 2.0 * float(corr.half_width)
        else:
            # no corridor in this preset: reference = nominal ground track,
            # corridor width = a generous flight band
            u = np.asarray([vgx, vgy], dtype=np.float64)
            n = np.linalg.norm(u)
            u = u / n if n > 1e-6 else np.asarray([1.0, 0.0])
            p0 = np.asarray([x0, y0]) - u * 2000.0
            p1 = np.asarray([x0, y0]) + u * 20000.0
            width = 1200.0

        # ~24 m resolution keeps project()/interp cheap on a straight line
        n_pts = max(2, int(np.linalg.norm(p1 - p0) / 24.0))
        pts = np.linspace(p0, p1, n_pts)

        v_min = float(getattr(envelope, "v_min", 6.0))
        v_max = float(getattr(envelope, "v_max", 75.0))
        lat_max = float(getattr(envelope, "lateral_accel_max", 4.5))
        a_max = float(getattr(envelope, "a_max", 2.5))
        v_des = float(np.clip(v_des if v_des > 1.0 else 0.8 * v_max,
                              v_min, v_max))

        cfg = PlannerConfig(
            desired_speed=v_des,
            corridor_width=width,
            lateral_step=max(8.0, width / 9.0),
            vehicle_radius=6.0,
            # 30 s route planning, 15 s risk accounting: candidates reach
            # to 30 s (graceful corridor return is part of the plan), but
            # risk is charged only where a forecast exists (risk_horizon,
            # same rule for every arm). Short/mid rungs keep tight
            # reactive maneuvers available. plan_dt widened for compute.
            horizon=29.0,
            duration_max=30.0,
            extra_durations=(6.5, 13.0),
            plan_dt=0.5,
            risk_horizon=15.0,
            time_normalized_costs=True,
            max_lateral_accel=lat_max,
            max_total_accel=a_max,
            speed_cap_lo=v_min,
            speed_cap_hi=v_max,
            speed_abs_max=v_max,
            # separation standard as soft standoff: full risk cost within
            # NMAC of a predicted position, sigma-decay only beyond it
            soft_standoff=NMAC_RADIUS_M,
            # cost weights FIXED across risk arms. safety is deliberately
            # dominant (12 vs 3.2 combined comfort terms): separation is
            # produced by the soft risk field, so it must outrank comfort
            # whenever risk is present; comfort tie-breaks in clear air.
            # offset is a preference (corridor centering), NOT safety --
            # kept low so lateral escapes stay cheap. weight_center adds an
            # ABSOLUTE terminal-offset charge on top: the pool-relative
            # costs alone let "park offset, fly straight" (zero jerk) win
            # forever after an avoidance; 0.5 is enough that the gentlest
            # long-duration return quintic beats parking in clear air,
            # while staying an order of magnitude under safety(12) so
            # escape decisions are untouched. Identical across risk arms.
            weight_jerk=0.8, weight_safety=12.0,
            weight_offset=1.0, weight_fuel=0.5, weight_center=0.5,
            weight_speed=3.0,   # hold cruise; braking must be bought by risk
            # deep-brake rung: "arrive later" as a plannable escape
            extra_speed_factors=(0.58,),
        )

        with self.lock:
            self.path = ReferencePath(pts)
            self.cfg = cfg
            self.plan = None
            self._last_result = None
            self.plan_valid_count = 0
            self.plan_total = 0

    def set_risk(self, risk: BaseRisk) -> None:
        with self.lock:
            self.risk = risk
            self.plan = None          # never track a plan made by another arm

    def reset_stats(self) -> None:
        with self.lock:
            self.plan_ms_max = 0.0
            self.replans = 0

    # ------------------------------------------------------------ planning
    def replan(self, sim_t: float, ep_dict: dict, t0: int, preds: dict,
               own_state: Any) -> None:
        """2 Hz: refresh risk, project ownship to Frenet, plan."""
        with self.lock:
            path, cfg, risk = self.path, self.cfg, self.risk
            prev_plan, prev_epoch = self.plan, self.plan_sim_t
        if path is None or cfg is None:
            return

        risk.update(ep_dict, t0, preds)

        proj = path.project(own_state.x, own_state.y)
        # heading_error(a, b) == b - a wrapped; we want dpsi = psi - theta_ref
        dpsi = heading_error(proj["theta"], own_state.psi)
        v = max(0.0, float(own_state.v))
        sd = max(0.1, v * math.cos(dpsi))
        dd = v * math.sin(dpsi)

        # accel continuity from the previous plan (measured accel is noisy)
        sdd = ddd = 0.0
        if prev_plan is not None:
            prev_fs = eval_trajectory_at(prev_plan, sim_t - prev_epoch)
            sdd, ddd = prev_fs.sdd, prev_fs.ddd

        fs = FrenetState(s=proj["s"], d=proj["d"], sd=sd, sdd=sdd,
                         dd=dd, ddd=ddd)

        t_start = time.perf_counter()
        result = plan_trajectory(fs, cfg, path, risk, mode="2d")
        ms = (time.perf_counter() - t_start) * 1000.0

        with self.lock:
            self.plan = result.best
            self._last_result = result
            self.plan_sim_t = sim_t
            self.plan_valid_count = result.metrics["valid"]
            self.plan_total = result.metrics["total"]
            self.plan_ms = round(ms, 1)
            self.plan_ms_max = round(max(self.plan_ms_max, ms), 1)
            self.replans += 1

    # ------------------------------------------------------------ tracking
    def command(self, t: float, state: Any, envelope: Any,
                ctx: dict) -> tuple[float, float]:
        """20 Hz engine hook: pure-pursuit the active plan."""
        with self.lock:
            plan, plan_t = self.plan, self.plan_sim_t
            path, cfg = self.path, self.cfg

        if plan is None or path is None or (t - plan_t) > PLAN_STALE_S:
            return self._corridor_hold(t, state, envelope, ctx)

        tau = (t - plan_t) + TRACK_LOOKAHEAD_S
        target = eval_trajectory_at(plan, tau)
        pt = path.point_at(target.s, target.d)

        dx, dy = pt["x"] - state.x, pt["y"] - state.y
        dist = math.hypot(dx, dy)
        if dist > 1.5:
            psi_des = math.atan2(dy, dx)
        else:  # on top of the carrot: hold the plan's own heading
            psi_des = pt["theta"] + math.atan2(target.dd,
                                               max(0.1, target.sd))
        omega = turn_rate_to_heading(state.psi, psi_des, envelope, state.v,
                                     HEADING_GAIN)

        v_des = math.hypot(target.sd, target.dd)
        if cfg is not None:
            v_des = float(np.clip(v_des, cfg.speed_cap_lo, cfg.speed_cap_hi))
        a_cmd = (v_des - state.v) / SPEED_TAU_S
        a_cmd = float(np.clip(a_cmd, getattr(envelope, "a_min", -3.0),
                              getattr(envelope, "a_max", 3.0)))
        return float(omega), a_cmd

    def _corridor_hold(self, t: float, state: Any, envelope: Any,
                       ctx: dict) -> tuple[float, float]:
        """No plan yet / stale plan: hold corridor heading and ref speed
        (mirrors the engine's default controller behavior)."""
        psi_ref = float(ctx.get("psi_ref", state.psi))
        v_ref = float(ctx.get("v_ref", state.v))
        if self.path is not None:
            proj = self.path.project(state.x, state.y)
            # ease back toward the centerline while holding course
            corr = math.atan2(-proj["d"], 220.0)
            psi_ref = proj["theta"] + corr
        omega = turn_rate_to_heading(state.psi, psi_ref, envelope, state.v,
                                     0.6)
        a_cmd = float(np.clip((v_ref - state.v) * 0.30,
                              getattr(envelope, "a_min", -3.0),
                              getattr(envelope, "a_max", 3.0)))
        return float(omega), a_cmd

    # ------------------------------------------------------------------ UI
    def wire(self, sim_t: float) -> dict:
        """JSON-safe snapshot of the active plan for the console."""
        with self.lock:
            plan = self.plan
            out = {
                "risk": self.risk.describe(),
                "viz": {**self.risk.viz(),
                        "standoff": (self.cfg.soft_standoff
                                     if self.cfg else 0.0),
                        # draw extrapolation only as far as risk is CHARGED
                        "T": min(self.plan.T if self.plan else 7.0,
                                 self.cfg.risk_horizon if self.cfg
                                 else 15.0)},
                "plan_ms": self.plan_ms,
                "plan_ms_max": self.plan_ms_max,
                "valid": self.plan_valid_count,
                "total": self.plan_total,
                "replans": self.replans,
            }
            res = self._last_result
            if res is not None:
                cands = []
                for c in res.candidates:
                    s = c.samples
                    n = len(s["x"])
                    idx = list(range(0, n, 4))
                    if idx[-1] != n - 1:
                        idx.append(n - 1)
                    cands.append({
                        "p": [[round(float(s["x"][i]), 1),
                               round(float(s["y"][i]), 1)] for i in idx],
                        "v": 1 if c.valid else 0,
                        "c": None if math.isinf(c.total_cost)
                             else round(c.total_cost, 3),
                    })
                out["cands"] = cands
            if plan is None:
                out["poly"] = []
                return out
            s = plan.samples
            idx = list(range(0, len(s["x"]), WIRE_DECIMATE))
            if idx[-1] != len(s["x"]) - 1:
                idx.append(len(s["x"]) - 1)
            out.update({
                "poly": [[round(float(s["x"][i]), 1),
                          round(float(s["y"][i]), 1)] for i in idx],
                "age_s": round(sim_t - self.plan_sim_t, 2),
                "T": plan.T,
                "target_speed": round(plan.target_speed, 1),
                "target_d": round(plan.target_d, 1),
                "min_clearance": None if math.isinf(plan.min_clearance)
                                 else round(plan.min_clearance, 1),
                "emergency": bool(self.plan_valid_count == 0),
                "cost": None if math.isinf(plan.total_cost)
                        else round(plan.total_cost, 3),
            })
            return out
