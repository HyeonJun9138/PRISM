"""Evaluation metrics (doc section 9), computed on-GPU in float32.

* minADE_K / minFDE_K  in METRES (un-scaled)          - doc 9.1
* mixture NLL per step                                 - doc 9.2
* Monte-Carlo HPD coverage at 50/70/90 % for the 5/10/15 s horizons - doc 9.3
  (coverage indicator: pdf(GT) >= (1-p)-quantile of pdf(samples drawn from
   the predicted mixture); exact for HPD regions as S -> inf)
* type accuracy, mode-usage entropy                    - doc 9.5
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common.constants import (COVERAGE_LEVELS, COVERAGE_MC_SAMPLES,
                              COVERAGE_STEPS, POS_SCALE)
from model.losses import gaussian_nll_steps, masked_mean


@torch.no_grad()
def batch_metrics(out: dict, batch: dict, mc_samples: int = COVERAGE_MC_SAMPLES) -> dict:
    y, y_valid = batch["y"].float(), batch["y_valid"].float()
    mu = out["mu"].float()
    B, K = mu.shape[:2]

    # ---------------------------------------------------------- minADE/minFDE
    err = torch.linalg.vector_norm(mu - y.unsqueeze(1), dim=-1)      # [B,K,30]
    ade = masked_mean(err, y_valid.unsqueeze(1), dim=2)              # [B,K]
    min_ade = ade.min(dim=1).values * POS_SCALE

    last = (y_valid.cumsum(dim=1).argmax(dim=1)).clamp_min(0)        # [B]
    fde_all = err.gather(2, last[:, None, None].expand(B, K, 1)).squeeze(2)
    min_fde = fde_all.min(dim=1).values * POS_SCALE

    # ------------------------------------------------------------ mixture NLL
    nll_steps = gaussian_nll_steps(y, out)                           # [B,K,30]
    log_pi = F.log_softmax(out["mode_logits"].float(), dim=1)        # [B,K]
    mix_step_ll = torch.logsumexp(log_pi[:, :, None] - nll_steps, dim=1)  # [B,30]
    nll = -masked_mean(mix_step_ll, y_valid, dim=1)                  # [B]

    # ---------------------------------------------------------------- modes
    winner = ade.argmin(dim=1)
    usage = torch.bincount(winner, minlength=K).float() / max(1, B)
    entropy = float(-(usage.clamp_min(1e-9) * usage.clamp_min(1e-9).log()).sum()
                    / math.log(max(2, K)))
    pi_mean = torch.softmax(out["mode_logits"].float(), dim=1).mean(dim=0)

    # ------------------------------------------------------------- coverage
    cov = _mc_coverage(out, y, y_valid, log_pi, mc_samples)

    # ---------------------------------------------------------------- types
    type_pred = out["type_logits"].argmax(dim=1)
    type_acc = (type_pred == batch["type_label"]).float().mean()

    res = {"minADE": float(min_ade.mean()), "minFDE": float(min_fde.mean()),
           "NLL": float(nll.mean()), "type_acc": float(type_acc),
           "mode_entropy": entropy,
           "n": int(B)}
    for k in range(K):
        res[f"pi_{k}"] = float(pi_mean[k])
        res[f"use_{k}"] = float(usage[k])
    res.update(cov)
    # per-sample arrays for stratified aggregation by the caller
    res["_per_sample"] = {
        "minADE": (min_ade).cpu(), "minFDE": (min_fde).cpu(),
        "NLL": nll.cpu(), "type_label": batch["type_label"].cpu(),
        "intent": batch.get("meta_intent", torch.full((B,), -1)).cpu(),
    }
    return res


@torch.no_grad()
def _mc_coverage(out: dict, y: torch.Tensor, y_valid: torch.Tensor,
                 log_pi: torch.Tensor, S: int) -> dict:
    """Monte-Carlo HPD coverage at COVERAGE_LEVELS x COVERAGE_STEPS."""
    mu = out["mu"].float(); log_sig = out["log_sigma"].float(); rho = out["rho"].float()
    B, K = mu.shape[:2]
    res = {}
    for step in COVERAGE_STEPS:
        m = mu[:, :, step]                       # [B,K,2]
        ls = log_sig[:, :, step]                 # [B,K,2]
        r = rho[:, :, step]                      # [B,K]
        valid = y_valid[:, step] > 0.5           # [B]
        if valid.sum() == 0:
            continue
        # sample S points from the mixture per batch row
        comp = torch.multinomial(log_pi.exp().clamp_min(1e-9), S, replacement=True)  # [B,S]
        idx = comp[:, :, None]
        m_s = torch.gather(m, 1, idx.expand(B, S, 2))
        ls_s = torch.gather(ls, 1, idx.expand(B, S, 2))
        r_s = torch.gather(r, 1, comp)
        sig_s = ls_s.exp()
        e1 = torch.randn(B, S, device=m.device)
        e2 = torch.randn(B, S, device=m.device)
        x1 = m_s[..., 0] + sig_s[..., 0] * e1
        x2 = m_s[..., 1] + sig_s[..., 1] * (r_s * e1 + torch.sqrt((1 - r_s ** 2).clamp_min(1e-6)) * e2)
        samples = torch.stack([x1, x2], dim=-1)  # [B,S,2]

        pdf_samples = _mixture_logpdf(samples, m, ls, r, log_pi)     # [B,S]
        pdf_gt = _mixture_logpdf(y[:, None, step], m, ls, r, log_pi)[:, 0]  # [B]

        for p in COVERAGE_LEVELS:
            thresh = torch.quantile(pdf_samples, 1.0 - p, dim=1)     # [B]
            inside = (pdf_gt >= thresh) & valid
            res[f"cov{int(p*100)}_{(step+1)//2}s"] = float(inside.float().sum() / valid.float().sum())
    return res


def _mixture_logpdf(x: torch.Tensor, mu: torch.Tensor, log_sig: torch.Tensor,
                    rho: torch.Tensor, log_pi: torch.Tensor) -> torch.Tensor:
    """log pdf of x [B,S,2] under mixture with [B,K,...] params -> [B,S]."""
    x = x[:, :, None, :]                          # [B,S,1,2]
    m = mu[:, None]                               # [B,1,K,2]
    sig = log_sig[:, None].exp()
    r = rho[:, None]                              # [B,1,K]
    zx = (x[..., 0] - m[..., 0]) / sig[..., 0]
    zy = (x[..., 1] - m[..., 1]) / sig[..., 1]
    omr2 = (1 - r ** 2).clamp_min(1e-6)
    q = (zx ** 2 - 2 * r * zx * zy + zy ** 2) / omr2
    log_n = -0.5 * q - log_sig[:, None, :, 0] - log_sig[:, None, :, 1] \
            - 0.5 * torch.log(omr2) - math.log(2 * math.pi)
    return torch.logsumexp(log_pi[:, None] + log_n, dim=2)


class MetricAccumulator:
    """Aggregates batch_metrics over a validation epoch, incl. stratified
    per-platform-type breakdown for the paper tables."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.n = 0
        self.per = {"minADE": [], "minFDE": [], "NLL": [],
                    "type_label": [], "intent": []}

    def update(self, m: dict) -> None:
        n = m.pop("n")
        per = m.pop("_per_sample")
        for k, v in m.items():
            if isinstance(v, (int, float)):
                self.sums[k] = self.sums.get(k, 0.0) + v * n
        self.n += n
        for k in self.per:
            self.per[k].append(per[k])

    def result(self) -> dict:
        import torch as _t
        out = {k: v / max(1, self.n) for k, v in self.sums.items()}
        if self.per["minADE"]:
            lab = _t.cat(self.per["type_label"])
            ade = _t.cat(self.per["minADE"])
            fde = _t.cat(self.per["minFDE"])
            by_type = {}
            for t in lab.unique().tolist():
                sel = lab == t
                by_type[int(t)] = {"minADE": float(ade[sel].mean()),
                                   "minFDE": float(fde[sel].mean()),
                                   "n": int(sel.sum())}
            out["by_type"] = by_type
        return out
