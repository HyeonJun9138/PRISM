"""Synthetic mini-dataset matching the dataGen storage schema.

Used by `main.py demo` and the self-tests so the training system can be
verified end-to-end even while the dataGen build is still in flight. The
motion model is deliberately simple (CV / coordinated-turn with occasional
mode flips) but the OBSERVATION model reproduces the properties the real
pipeline depends on: NaN at undetected steps, range-grown covariance,
track-age bookkeeping, per-agent detection dropout.
"""
from __future__ import annotations

import json
import os

import numpy as np

T_STEPS = 120
LOG_DT = 0.5


def make_synthetic_run(run_dir: str, n_episodes: int = 8, n_agents: int = 5,
                       seed: int = 1234) -> str:
    ep_dir = os.path.join(run_dir, "episodes")
    os.makedirs(ep_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    index_lines = []
    for e in range(n_episodes):
        arrays = _make_episode(rng, n_agents)
        np.savez_compressed(os.path.join(ep_dir, f"ep_{e:06d}.npz"), **arrays)
        index_lines.append({"episode_index": e, "seed": seed + e,
                            "n_agents": n_agents, "nmac_occurred": False})
    with open(os.path.join(run_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for line in index_lines:
            f.write(json.dumps(line) + "\n")
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"preset": "synthetic", "created": "synthetic",
                   "n_episodes": n_episodes}, f)
    return run_dir


def _make_episode(rng: np.random.Generator, N: int) -> dict:
    T = T_STEPS
    gt_pos = np.zeros((T, N, 2), np.float32)
    gt_vel = np.zeros((T, N, 2), np.float32)
    gt_psi = np.zeros((T, N), np.float32)
    gt_valid = np.ones((T, N), bool)
    gt_mode = np.zeros((T, N), np.int8)
    gt_intent = rng.integers(0, 3, size=N).astype(np.int8)[None, :].repeat(T, 0)
    gt_type = rng.integers(1, 4, size=N).astype(np.int8)     # trainable classes

    for a in range(N):
        p = rng.uniform(-800, 800, 2)
        v = rng.uniform(8, 30)
        psi = rng.uniform(-np.pi, np.pi)
        omega = 0.0
        flip_every = rng.integers(10, 40)
        for t in range(T):
            if t % flip_every == 0 and t > 0:
                # committed stochastic manoeuvre: the CV-defeating structure
                omega = rng.choice([-1.0, 0.0, 0.0, 1.0]) * rng.uniform(0.02, 0.12)
            psi += omega
            p = p + v * LOG_DT * np.array([np.cos(psi), np.sin(psi)])
            gt_pos[t, a] = p
            gt_vel[t, a] = v * np.array([np.cos(psi), np.sin(psi)])
            gt_psi[t, a] = np.arctan2(np.sin(psi), np.cos(psi))
            gt_mode[t, a] = 0 if omega == 0 else (1 if omega > 0 else 2)

    own_pos = np.zeros((T, 2), np.float32)
    own_vel = np.zeros((T, 2), np.float32)
    own_psi = np.zeros(T, np.float32)
    for t in range(T):
        own_pos[t] = np.array([-1000 + 40 * LOG_DT * t, 0.0])
        own_vel[t] = np.array([40.0, 0.0])

    obs_detected = np.zeros((T, N), bool)
    obs_pos = np.full((T, N, 2), np.nan, np.float32)
    obs_R = np.full((T, N, 2, 2), np.nan, np.float32)
    obs_track_id = np.full((T, N), -1, np.int32)
    obs_track_age = np.zeros((T, N), np.int32)

    next_track = 0
    age = np.zeros(N, np.int32)
    tid = np.full(N, -1, np.int32)
    miss = np.zeros(N, np.int32)
    for t in range(T):
        for a in range(N):
            r_vec = gt_pos[t, a] - own_pos[t]
            r = float(np.hypot(*r_vec))
            pd = 0.95 if r < 1200 else max(0.0, 0.95 * (1 - (r - 1200) / 2500))
            if rng.random() < pd:
                theta = np.arctan2(r_vec[1], r_vec[0])
                sig_r = 3.0 + 0.01 * r
                sig_t = 0.0087
                dr = rng.normal(0, sig_r)
                dth = rng.normal(0, sig_t)
                rn, thn = r + dr, theta + dth
                obs_pos[t, a] = own_pos[t] + rn * np.array([np.cos(thn), np.sin(thn)])
                J = np.array([[np.cos(theta), -r * np.sin(theta)],
                              [np.sin(theta), r * np.cos(theta)]])
                obs_R[t, a] = J @ np.diag([sig_r ** 2, sig_t ** 2]) @ J.T
                if tid[a] < 0:
                    tid[a] = next_track; next_track += 1; age[a] = 0
                else:
                    age[a] += 1
                miss[a] = 0
                obs_detected[t, a] = True
                obs_track_id[t, a] = tid[a]
                obs_track_age[t, a] = age[a]
            else:
                miss[a] += 1
                if miss[a] > 6:
                    tid[a] = -1

    return dict(t=(np.arange(T) * LOG_DT).astype(np.float32),
                gt_pos=gt_pos, gt_vel=gt_vel, gt_psi=gt_psi, gt_valid=gt_valid,
                gt_mode=gt_mode, gt_intent=gt_intent, gt_type=gt_type,
                own_pos=own_pos, own_vel=own_vel, own_psi=own_psi,
                obs_detected=obs_detected, obs_pos=obs_pos, obs_R=obs_R,
                obs_track_id=obs_track_id, obs_track_age=obs_track_age,
                agent_ids=np.arange(N, dtype=np.int32))
