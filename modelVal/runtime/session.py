"""The live session: dataGen engine at wall-clock pace, keyboard ownship,
2 Hz prediction + scoring, endless scenario rotation.

Thread layout: ONE session thread ticks the engine at SIM_DT * speed pace.
Every LOG_DECIMATION ticks (the engine's own logging cadence) it snapshots a
display frame, runs the predictor over the fresh observation row, settles the
scoreboard, and publishes a JSON-safe frame into a deque the SSE stream reads.

The keyboard maps to the ownship EXACTLY through the engine's pluggable
controller hook: the command is clamped by the same eVTOL envelope the
dataset's ownship flew with -- the pilot cannot cheat the physics.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from dataclasses import replace as dc_replace

from core.constants import (DWC_HMD_M, LOG_DECIMATION, NMAC_RADIUS_M, SIM_DT)
from core.types import PLATFORM_COLORS, TYPE_NAMES, PlatformType
from scenario.config import PlatformMixConfig
from scenario.presets import config_for_episode
from simulation.engine import SimulationEngine

from runtime.predictor import LivePredictor
from runtime.scoring import RollingScore

from planner.controller import PlannerPilot
from planner.risk import make_risk

SESSION_DURATION_S = 150.0        # scenario rotates every 2.5 min: a 55 m/s
                                  # ownship outruns slow traffic in ~2 min, so
                                  # longer sessions end in empty sky
PRED_DECIMATE = 3                 # send every 3rd step of each branch to the UI

# Console-only viewing presets: overlays on the CANONICAL dataset splits.
# The dataset presets themselves are never touched -- these exist so a human
# can watch longer sessions, thinner (realistic) traffic, or deliberately
# absurd stress levels. keys: base split, session duration, top-level config
# overrides, encounter overrides, platform-mix replacement.
CONSOLE_PRESETS: dict[str, dict] = {
    "train":    {"base": "train",    "dur": 150.0},
    "test_ood": {"base": "test_ood", "dur": 150.0},
    # longer look at the OOD mix (10 min before rotation)
    "ood_long": {"base": "test_ood", "dur": 600.0},
    # realistic sparse low-altitude traffic: avoidance should WIN here
    "real_sparse": {"base": "test_ood", "dur": 300.0,
                    "cfg": {"density_per_km2": 8.0,
                            "n_agents_min": 2, "n_agents_max": 6},
                    "enc": {"n_designed_min": 1, "n_designed_max": 2}},
    # 3x density, designed encounters maxed out
    "hell": {"base": "test_ood", "dur": 150.0,
             "cfg": {"density_per_km2": 300.0,
                     "n_agents_min": 18, "n_agents_max": 28},
             "enc": {"n_designed_min": 5, "n_designed_max": 8}},
    # bird/debris rush: the HARD difficulty band, in bulk
    "hell_birds": {"base": "test_ood", "dur": 150.0,
                   "cfg": {"density_per_km2": 200.0,
                           "n_agents_min": 14, "n_agents_max": 22},
                   "enc": {"n_designed_min": 4, "n_designed_max": 6},
                   "mix": {int(PlatformType.BIRD): 0.50,
                           int(PlatformType.DEBRIS): 0.20,
                           int(PlatformType.UNKNOWN): 0.15,
                           int(PlatformType.MULTIROTOR): 0.15}},
}


class KeyState:
    """Shared pilot input. turn/accel in [-1, 1]; brake latches max decel."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.turn = 0.0
        self.accel = 0.0
        self.brake = False

    def set(self, turn: float, accel: float, brake: bool) -> None:
        with self.lock:
            self.turn = float(np.clip(turn, -1.0, 1.0))
            self.accel = float(np.clip(accel, -1.0, 1.0))
            self.brake = bool(brake)

    def command(self, t, state, envelope, ctx) -> tuple[float, float]:
        """Engine ownship-controller hook (doc 7.2 continuous action space)."""
        with self.lock:
            turn, accel, brake = self.turn, self.accel, self.brake
        omega = turn * envelope.max_turn_rate(state.v)
        a = envelope.a_min if brake else accel * (envelope.a_max if accel >= 0
                                                  else -envelope.a_min)
        return float(omega), float(a)


class LiveSession:
    """One running validation session (engine + predictor + scoreboard)."""

    def __init__(self, predictor: LivePredictor, preset: str = "test_ood",
                 seed: int = 0, speed: float = 1.0) -> None:
        self.predictor = predictor
        self.keys = KeyState()
        self.pilot = PlannerPilot()
        self.control_mode = "manual"          # "manual" | "planner"
        self._pilot_ctx = None                # last logged (t, ep, t0, preds, own)
        self.score = RollingScore()
        self.lock = threading.RLock()
        self.frames: deque = deque(maxlen=40)
        self.frame_seq = 0
        self.preset = preset
        self.seed = int(seed)
        self.speed = float(speed)
        self.paused = True
        self.running = True
        self.scenario_count = 0
        self.nmac_events = 0
        self.min_sep_session = float("inf")
        self._was_nmac = False
        self._statics: dict | None = None
        self._build_engine()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="modelval-session")
        self._thread.start()

    # ----------------------------------------------------------- lifecycle
    def _build_engine(self) -> None:
        spec = CONSOLE_PRESETS.get(self.preset,
                                   {"base": self.preset,
                                    "dur": SESSION_DURATION_S})
        cfg = config_for_episode(spec["base"], self.seed)
        over: dict = dict(spec.get("cfg", {}))
        if "enc" in spec:
            over["encounter"] = dc_replace(cfg.encounter, **spec["enc"])
        if "mix" in spec:
            over["platform_mix"] = PlatformMixConfig(
                weights={int(k): float(v) for k, v in spec["mix"].items()})
        cfg = cfg.copy_with(duration_s=float(spec.get("dur",
                                                      SESSION_DURATION_S)),
                            name=f"live/{self.preset}/{self.seed}", **over)
        self.engine = SimulationEngine(cfg, ownship_controller=self._command)
        scn = self.engine.scenario
        self.pilot.bind(scn.world, scn.ownship.envelope, scn.ownship_track)
        self._pilot_ctx = None      # stale: references the previous engine
        self.score.clear_pending()
        self.scenario_count += 1
        self._statics = None          # resend world statics on next frame

    def rotate(self, seed: Optional[int] = None) -> None:
        with self.lock:
            self.seed = int(seed) if seed is not None else self.seed + 1
            self._build_engine()

    def set_preset(self, preset: str) -> None:
        with self.lock:
            self.preset = preset
            self._build_engine()
            self.reset_score()

    def reset_score(self) -> None:
        """Fresh scoreboard: rolling windows must not mix conditions (a
        preset or model change would otherwise leave stale numbers on the
        tiles for minutes)."""
        with self.lock:
            self.score = RollingScore()
            self.nmac_events = 0
            self.min_sep_session = float("inf")
            self.predictor.latency_max_ms = 0.0

    def stop(self) -> None:
        self.running = False

    # ------------------------------------------------------- ownship pilot
    def _command(self, t, state, envelope, ctx) -> tuple[float, float]:
        """Engine ownship hook: dispatch to the Frenet pilot or keyboard."""
        if self.control_mode == "planner":
            return self.pilot.command(t, state, envelope, ctx)
        return self.keys.command(t, state, envelope, ctx)

    def set_pilot(self, control: Optional[str] = None,
                  risk: Optional[str] = None) -> dict:
        """Switch control source and/or risk arm.

        A RISK change restarts the CURRENT scenario (same seed): the whole
        point of switching arms is watching the same traffic get avoided
        differently, and rolling scores must not mix conditions. A control
        change alone hands over mid-flight (no restart) so the pilot can
        grab/release the stick."""
        restarted = False
        if risk is not None:
            self.pilot.set_risk(make_risk(str(risk)))
            self.reset_score()
            self.rotate(seed=self.seed)           # same-seed replay
            restarted = True
        if control is not None:
            if control not in ("manual", "planner"):
                raise ValueError(f"unknown control mode '{control}'")
            self.control_mode = str(control)
        if (not restarted and self.control_mode == "planner"
                and self._pilot_ctx is not None):
            self.pilot.replan(*self._pilot_ctx)   # take effect NOW
        return {"control": self.control_mode, "risk": self.pilot.risk.name,
                "restarted": restarted, "seed": self.seed}

    # ----------------------------------------------------------- main loop
    def _run(self) -> None:
        next_tick = time.perf_counter()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                next_tick = time.perf_counter()
                continue
            with self.lock:
                alive = self.engine.step(log=True)
                tick = self.engine._tick
                if tick % LOG_DECIMATION == 0:
                    self._on_logged_frame()
                if not alive:
                    self.rotate()
            # wall-clock pacing (speed x real time), skipping sleep on lag
            next_tick += SIM_DT / max(0.1, self.speed)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    # ------------------------------------------------------ frame assembly
    def _on_logged_frame(self) -> None:
        eng = self.engine
        t0 = eng._log_index - 1
        if t0 < 0:
            return
        ep = eng.ep
        # obs_track_id: a current-track checkpoint (track_rule "current",
        # R3 major 3) refuses an episode without it -- see LivePredictor
        ep_dict = {k: getattr(ep, k) for k in
                   ("gt_pos", "gt_valid", "gt_type", "gt_intent",
                    "obs_detected", "obs_pos", "obs_R", "obs_track_age",
                    "own_pos", "obs_track_id")}

        preds = {}
        if self.predictor.loaded:
            preds = self.predictor.predict_frame(ep_dict, t0)
            self.score.add_predictions(t0, preds)
        self.score.settle(ep_dict, t0)

        # stashed so a pilot/risk switch can replan IMMEDIATELY (snappy UI)
        # instead of waiting for the next logged frame
        self._pilot_ctx = (eng.t, ep_dict, t0, preds,
                           eng.scenario.ownship.state)
        if self.control_mode == "planner":
            self.pilot.replan(eng.t, ep_dict, t0, preds,
                              eng.scenario.ownship.state)

        own = eng.scenario.ownship.state
        # separation bookkeeping (session-level, distinct from per-scenario)
        sep = float("inf")
        for ag in eng.scenario.agents:
            if ag.alive:
                sep = min(sep, math.hypot(ag.state.x - own.x, ag.state.y - own.y))
        self.min_sep_session = min(self.min_sep_session, sep)
        is_nmac = sep < NMAC_RADIUS_M
        if is_nmac and not self._was_nmac:
            self.nmac_events += 1
        self._was_nmac = is_nmac

        frame = self._assemble_frame(t0, preds, sep)
        with self.lock:
            self.frame_seq += 1
            frame["seq"] = self.frame_seq
            self.frames.append(frame)

    def _assemble_frame(self, t0: int, preds: dict, sep: float) -> dict[str, Any]:
        eng = self.engine
        own = eng.scenario.ownship.state
        agents = []
        for slot, ag in enumerate(eng.scenario.agents):
            if not ag.alive:
                continue
            det = bool(eng.ep.obs_detected[t0, slot])
            entry = {
                "slot": slot, "x": round(ag.state.x, 1), "y": round(ag.state.y, 1),
                "psi": round(ag.state.psi, 3), "v": round(ag.state.v, 1),
                "type": int(ag.platform), "det": det,
            }
            if det:
                op = eng.ep.obs_pos[t0, slot]
                entry["ox"] = round(float(op[0]), 1)
                entry["oy"] = round(float(op[1]), 1)
            agents.append(entry)

        pred_wire = {}
        for a, p in preds.items():
            idx = list(range(PRED_DECIMATE - 1, p["mu_w"].shape[1], PRED_DECIMATE))
            tp = p["type_probs"]
            top1 = int(np.argmax(tp))
            pred_wire[str(a)] = {
                "pi": [round(float(v), 3) for v in p["pi"]],
                "mu": np.round(p["mu_w"][:, idx], 1).tolist(),
                "sig": np.round(p["sigma"][:, idx], 1).tolist(),
                "steps": idx,
                "type": [top1, round(float(tp[top1]), 2)],
            }

        frame = {
            "t": round(eng.t, 2),
            "scenario": self.scenario_count,
            "preset": self.preset,
            "seed": self.seed,
            "own": {"x": round(own.x, 1), "y": round(own.y, 1),
                    "psi": round(own.psi, 3), "v": round(own.v, 1),
                    "omega_deg": round(math.degrees(own.omega), 1)},
            "agents": agents,
            "preds": pred_wire,
            "pilot": ({"control": self.control_mode,
                       **self.pilot.wire(eng.t)}
                      if self.control_mode == "planner" else
                      {"control": self.control_mode,
                       "risk": self.pilot.risk.name}),
            "metrics": {
                **self.score.summary(),
                "latency_ms": self.predictor.latency_ms,
                "latency_max_ms": self.predictor.latency_max_ms,
                "tracks_predicted": len(preds),
                "sep_m": None if not math.isfinite(sep) else round(sep, 0),
                "min_sep_session_m": None if not math.isfinite(self.min_sep_session)
                                     else round(self.min_sep_session, 0),
                "nmac_events": self.nmac_events,
                "nmac_now": bool(sep < NMAC_RADIUS_M),
                "dwc_now": bool(sep < DWC_HMD_M),
                "model": self.predictor.info.get("path", "").split("/")[-3:-1] or None,
            },
        }
        if self._statics is None:
            self._statics = self._world_statics()
        frame["statics"] = self._statics if self.frame_seq % 20 == 0 or \
            self.frame_seq <= 1 else None
        if frame["statics"] is None:
            frame.pop("statics")
        return frame

    def _world_statics(self) -> dict:
        w = self.engine.scenario.world
        d = w.to_render_dict() if hasattr(w, "to_render_dict") else {}
        d["nmac_m"] = NMAC_RADIUS_M
        d["dwc_m"] = DWC_HMD_M
        d["type_names"] = {int(k): v for k, v in TYPE_NAMES.items()}
        d["type_colors"] = {int(k): v for k, v in PLATFORM_COLORS.items()}
        return d

    # ---------------------------------------------------------------- api
    def latest_frames(self, after_seq: int) -> list[dict]:
        with self.lock:
            return [f for f in self.frames if f["seq"] > after_seq]
