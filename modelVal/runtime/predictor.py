"""Live predictor: loads a trained checkpoint and predicts every visible
track from the engine's in-memory observation arrays.

The feature pipeline IS modelTraining's (data.dataset.build_sample) -- the
whole point of this console is to exercise the trained artifact exactly as
trained. Predictions come back in each target's frame; this module transforms
them to world axes for display and scoring.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import torch

from common.constants import HIST_STEPS, POS_SCALE, PRED_STEPS
from data.dataset import build_sample, index_track_rule, window_detections
from model.network import build_model

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

MODELS_ROOT = f"{PRISM_ROOT}/Logs/Models"


def list_models() -> list[dict]:
    """Loadable artifacts under Logs/Models: final models + best checkpoints."""
    out = []
    if not os.path.isdir(MODELS_ROOT):
        return out
    for job in sorted(os.listdir(MODELS_ROOT), reverse=True):
        jd = os.path.join(MODELS_ROOT, job)
        for rel, kind in (("final/model_final.pt", "final"),
                          ("checkpoints/best.pt", "best")):
            p = os.path.join(jd, rel)
            if os.path.exists(p):
                out.append({"job": job, "kind": kind,
                            "path": p.replace("\\", "/"),
                            "mb": round(os.path.getsize(p) / 1e6, 1)})
    return out


class LivePredictor:
    """Owns the loaded model + a prediction cache for the current frame."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.data_cfg: dict = {}
        self.info: dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latency_ms = 0.0
        self.latency_max_ms = 0.0
        self._lat_ema = None

    # ---------------------------------------------------------------- model
    def load(self, path: str) -> dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        model_cfg = ck.get("model_cfg") or ck.get("cfg", {}).get("model")
        data_cfg = ck.get("data_cfg") or ck.get("cfg", {}).get("data")
        if model_cfg is None or data_cfg is None:
            raise ValueError(f"{path}: not a trainer artifact (no config inside)")
        model = build_model(model_cfg).to(self.device)
        model.load_state_dict(ck["model"])
        model.eval()
        with self.lock:
            self.model = model
            self.data_cfg = dict(data_cfg)
            self._lat_ema = None
            self.latency_max_ms = 0.0
            self.info = {
                "path": path.replace("\\", "/"),
                "epoch": int(ck.get("epoch", -1)),
                "params": sum(p.numel() for p in model.parameters()),
                "K": int(model_cfg.get("n_modes", 3)),
                "d_model": int(model_cfg.get("d_model", 0)),
                "device": str(self.device),
                "track_rule": index_track_rule(data_cfg),
                "loaded_at": time.strftime("%H:%M:%S"),
            }
        return dict(self.info)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def predict_frame(self, ep_dict: dict, t0: int) -> dict[int, dict[str, Any]]:
        """Predict every eligible track at logged step t0.

        Returns {agent_slot: {mu_w [K,30,2] world-metres, sigma [K,30,2] m,
                              rho [K,30], pi [K], type_probs [C]}}.
        Eligibility mirrors training exactly: >= min_obs usable detections in
        the window under the checkpoint's own track rule (current-track for
        models trained after 2026-09-26, slot for legacy ones).
        """
        if self.model is None or t0 < HIST_STEPS - 1:
            return {}
        det = window_detections(ep_dict, t0, index_track_rule(self.data_cfg))
        n_det = det.sum(axis=0)
        min_obs = max(2, int(self.data_cfg.get("min_obs", 3)))
        targets = [a for a in range(det.shape[1])
                   if n_det[a] >= min_obs and bool(ep_dict["gt_valid"][t0, a])]
        if not targets:
            return {}

        samples = [build_sample(ep_dict, t0, a, self.data_cfg, with_labels=False)
                   for a in targets]
        x = torch.as_tensor(np.stack([s["x"] for s in samples])).to(self.device)
        m = torch.as_tensor(np.stack([s["agent_mask"] for s in samples])).to(self.device)

        t_start = time.perf_counter()
        with self.lock:
            out = self.model(x, m)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t_start) * 1000.0
        self._lat_ema = ms if self._lat_ema is None else 0.85 * self._lat_ema + 0.15 * ms
        self.latency_ms = round(self._lat_ema, 2)
        self.latency_max_ms = round(max(self.latency_max_ms, ms), 2)

        mu = out["mu"].float().cpu().numpy() * POS_SCALE          # [B,K,30,2] frame-m
        sig = np.exp(out["log_sigma"].float().cpu().numpy()) * POS_SCALE
        rho = out["rho"].float().cpu().numpy()
        pi = torch.softmax(out["mode_logits"].float(), dim=1).cpu().numpy()
        typ = torch.softmax(out["type_logits"].float(), dim=1).cpu().numpy()

        result: dict[int, dict[str, Any]] = {}
        for i, (a, s) in enumerate(zip(targets, samples)):
            ox, oy, theta = s["frame"]
            c, sn = np.cos(theta), np.sin(theta)
            R_inv = np.array([[c, -sn], [sn, c]])                 # frame -> world
            mu_w = mu[i] @ R_inv.T + np.array([ox, oy])
            result[a] = {"mu_w": mu_w.astype(np.float32),
                         "sigma": sig[i].astype(np.float32),
                         "rho": rho[i].astype(np.float32),
                         "pi": pi[i].astype(np.float32),
                         "theta": float(theta),
                         "type_probs": typ[i].astype(np.float32)}
        return result
