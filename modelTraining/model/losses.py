"""Losses for the K-mode mixture predictor.

Primary recipe (doc 5.1 rev.3):
  * winner-takes-all (WTA) NLL on the best mode  -> fights mode collapse
  * cross-entropy pushing mode probabilities toward the winner
  * auxiliary platform-type cross-entropy
  * optional FULL-MIXTURE NLL, ramped in late in training, which calibrates
    the mode weights pi as genuine probabilities (the calibration story needs
    pi to be meaningful, not just argmax-correct).

All loss math runs in float32 regardless of autocast dtype.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll_steps(y: torch.Tensor, out: dict) -> torch.Tensor:
    """Per-step negative log density of y under each mode.

    y [B,30,2], out fields [B,K,30,*] -> nll [B,K,30] (float32).
    Bivariate Gaussian with per-step (mu, sigma, rho).
    """
    mu = out["mu"].float()
    log_sig = out["log_sigma"].float()
    rho = out["rho"].float()
    y = y.float().unsqueeze(1)                      # [B,1,30,2]

    sig = torch.exp(log_sig)
    zx = (y[..., 0] - mu[..., 0]) / sig[..., 0]
    zy = (y[..., 1] - mu[..., 1]) / sig[..., 1]
    omr2 = (1.0 - rho ** 2).clamp_min(1e-6)
    q = (zx ** 2 - 2.0 * rho * zx * zy + zy ** 2) / omr2
    log_det = log_sig[..., 0] + log_sig[..., 1] + 0.5 * torch.log(omr2)
    return 0.5 * q + log_det + LOG_2PI              # [B,K,30]


def masked_mean(v: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float()
    return (v * m).sum(dim) / m.sum(dim).clamp_min(1.0)


def compute_losses(out: dict, batch: dict, loss_cfg: dict,
                   mixture_weight: float) -> tuple[torch.Tensor, dict]:
    """Return (total_loss, float components dict).

    loss_cfg["variant"]:
      "nll" (default) -- the K-mode mixture recipe of doc 5.1.
      "mse"           -- point-estimate baseline (doc 8.2): squared error on
                         mu only; sigma/rho/pi receive NO gradient. Use with
                         n_modes=1 and best_metric val/minADE.
    """
    y, y_valid = batch["y"], batch["y_valid"]                 # [B,30,2],[B,30]

    if str(loss_cfg.get("variant", "nll")) == "mse":
        err2 = ((out["mu"].float() - y.float().unsqueeze(1)) ** 2).sum(-1)
        per_mode = masked_mean(err2, y_valid.unsqueeze(1), dim=2)   # [B,K]
        l_traj = per_mode.min(dim=1).values.mean()
        l_type = F.cross_entropy(out["type_logits"].float(), batch["type_label"])
        total = l_traj + float(loss_cfg.get("w_type", 0.2)) * l_type
        return total, {"loss": float(total.detach()),
                       "loss_traj": float(l_traj.detach()),
                       "loss_mode": 0.0, "loss_type": float(l_type.detach()),
                       "loss_mix": 0.0, "w_mix": 0.0}

    nll_steps = gaussian_nll_steps(y, out)                    # [B,K,30]
    per_mode = masked_mean(nll_steps, y_valid.unsqueeze(1), dim=2)   # [B,K]

    winner = per_mode.detach().argmin(dim=1)                  # [B]
    l_traj = per_mode.gather(1, winner[:, None]).squeeze(1).mean()

    log_pi = F.log_softmax(out["mode_logits"].float(), dim=1)
    l_mode = F.nll_loss(log_pi, winner)

    # full-mixture NLL: -log sum_k pi_k * prod_t N_t  (sum over valid steps,
    # normalised per step so its scale matches l_traj)
    n_valid = y_valid.float().sum(dim=1).clamp_min(1.0)       # [B]
    sum_nll = (nll_steps * y_valid.unsqueeze(1)).sum(dim=2)   # [B,K]
    l_mix = (-torch.logsumexp(log_pi - sum_nll, dim=1) / n_valid).mean()

    l_type = F.cross_entropy(out["type_logits"].float(), batch["type_label"])

    total = (l_traj
             + float(loss_cfg.get("w_mode", 1.0)) * l_mode
             + float(loss_cfg.get("w_type", 0.2)) * l_type
             + mixture_weight * l_mix)

    comps = {"loss": float(total.detach()), "loss_traj": float(l_traj.detach()),
             "loss_mode": float(l_mode.detach()), "loss_type": float(l_type.detach()),
             "loss_mix": float(l_mix.detach()), "w_mix": mixture_weight}
    return total, comps


def mixture_weight_at(epoch: int, total_epochs: int, loss_cfg: dict) -> float:
    """Linear ramp of the mixture-NLL weight from ramp_start to the end."""
    start = float(loss_cfg.get("mixture_ramp_start", 0.7))
    w_max = float(loss_cfg.get("w_mixture_max", 0.5))
    frac = (epoch + 1) / max(1, total_epochs)
    if frac <= start or w_max <= 0:
        return 0.0
    return w_max * min(1.0, (frac - start) / max(1e-9, 1.0 - start))
