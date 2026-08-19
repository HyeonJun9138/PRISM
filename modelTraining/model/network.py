"""PRISM trajectory predictor: K-mode mixture-of-Gaussians network.

Architecture (doc 5.1 rev.3, 5.3):

  per-timestep embed -> temporal Transformer (per agent, 20 steps)
                     -> attention pooling  -> one embedding per agent
                     -> agent Transformer  (interaction across A slots)
                     -> target slot + masked scene pool -> trunk MLP
                     -> heads: trajectory [K,30,5] / mode logits [K]
                              / platform-type logits [C]

Design notes
  * Undetected timesteps are NOT masked out of temporal attention: the gap
    pattern (detected flag, gap feature) is information the model must see to
    learn "no data -> widen sigma" (doc 3.5).
  * mu is predicted as per-step deltas then cumulatively summed, which biases
    early training toward smooth, dynamically plausible trajectories.
  * sigma is bounded via log-space clamping; rho via a scaled tanh, so the
    2x2 covariance is always positive-definite.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from common.constants import (F_DIM, LOG_SIGMA_MAX, LOG_SIGMA_MIN, N_TYPES,
                              PARAMS_PER_STEP, PRED_STEPS, RHO_LIMIT)


class AttnPool(nn.Module):
    """Learned-query attention pooling over the time axis."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:   # [B*, T, d] -> [B*, d]
        w = torch.softmax(h @ self.q / math.sqrt(h.shape[-1]), dim=-1)
        return torch.einsum("bt,btd->bd", w, h)


class PrismPredictor(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg.get("d_model", 128))
        self.d = d
        self.K = int(cfg.get("n_modes", 3))
        self.cumulative_mu = bool(cfg.get("cumulative_mu", True))
        heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        drop = float(cfg.get("dropout", 0.1))

        self.embed = nn.Sequential(nn.Linear(F_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.time_pe = nn.Parameter(torch.randn(1, 20, d) * 0.02)

        enc = nn.TransformerEncoderLayer(d, heads, d_ff, drop,
                                         batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, int(cfg.get("temporal_layers", 2)))
        self.pool = AttnPool(d)

        agent_enc = nn.TransformerEncoderLayer(d, heads, d_ff, drop,
                                               batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(agent_enc, int(cfg.get("agent_layers", 2)))

        self.trunk = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(2 * d, 2 * d), nn.GELU())
        self.traj_head = nn.Linear(2 * d, self.K * PRED_STEPS * PARAMS_PER_STEP)
        self.mode_head = nn.Linear(2 * d, self.K)
        self.type_head = nn.Linear(2 * d, N_TYPES)

        # small init on the trajectory head keeps step-deltas near zero at start
        nn.init.normal_(self.traj_head.weight, std=1e-3)
        nn.init.zeros_(self.traj_head.bias)

    # ------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor, agent_mask: torch.Tensor) -> dict:
        """x [B,A,T,F], agent_mask [B,A] -> dict of raw-but-bounded params."""
        B, A, T, _ = x.shape
        h = self.embed(x.reshape(B * A, T, -1)) + self.time_pe[:, :T]
        h = self.temporal(h)                     # [B*A,T,d]
        emb = self.pool(h).reshape(B, A, self.d)  # [B,A,d]

        pad = agent_mask < 0.5                    # True = ignore slot
        # guard: never let every slot of a row be padded (slot 0 is the target)
        pad = pad.clone()
        pad[:, 0] = False
        z = self.interaction(emb, src_key_padding_mask=pad)   # [B,A,d]

        scene_w = agent_mask / agent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        scene = torch.einsum("ba,bad->bd", scene_w, z)
        feat = self.trunk(torch.cat([z[:, 0], scene], dim=-1))  # [B,2d]

        raw = self.traj_head(feat).reshape(B, self.K, PRED_STEPS, PARAMS_PER_STEP)
        mu = raw[..., 0:2]
        if self.cumulative_mu:
            mu = torch.cumsum(mu, dim=2)
        log_sig = raw[..., 2:4].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        rho = RHO_LIMIT * torch.tanh(raw[..., 4])

        return {
            "mu": mu,                              # [B,K,30,2] scaled units
            "log_sigma": log_sig,                  # [B,K,30,2]
            "rho": rho,                            # [B,K,30]
            "mode_logits": self.mode_head(feat),   # [B,K]
            "type_logits": self.type_head(feat),   # [B,C]
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_cfg: dict) -> PrismPredictor:
    return PrismPredictor(model_cfg)
