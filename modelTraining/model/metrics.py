"""Evaluation metrics (doc section 9), computed on-GPU in float32.

LEGACY METRICS -- every paper number since 2026-08 rests on these; their
values must never change (review round 1 re-ran the s4 evaluation to prove it):

* minADE_K / minFDE_K  in METRES (un-scaled)          - doc 9.1
* "NLL" = per-step MARGINAL mixture NLL               - doc 9.2
      NLL_i = (1/|V_i|) sum_{t in V_i} -log sum_k pi_k N(y_t; mu_kt, Sigma_kt)
  At every valid step t the K branches are re-mixed on their own: it scores
  the POSITION MARGINAL of each step, not a trajectory (the branch is NOT
  shared across steps). Averaged over the valid steps V_i of a sample, then
  over samples. Coordinates are in POS_SCALE units (metres / 100), so the
  density is per (100 m)^2: add 2 ln(100) = 9.2103 per step for a density per
  m^2. Reported in the paper as "NLL" (s4 test_id: 1.0601).
* Monte-Carlo HPD coverage at 50/70/90 % for the 5/10/15 s horizons - doc 9.3
  (coverage indicator: pdf(GT) >= (1-p)-quantile of pdf(samples drawn from
   the predicted mixture); exact for HPD regions as S -> inf). Until review
   round 1 the draws came from the GLOBAL torch RNG, whose default seed is
   non-deterministic per process here (torch 2.5: torch.initial_seed() and
   torch.cuda.initial_seed() differ on every start), so historical coverage
   values carry MC noise of a few 1e-4 between runs (s4 test_id: re-running
   the unchanged code moved cov90_15s 0.90080 -> 0.90105, max |d| 7.9e-4 over
   the nine cells; every deterministic metric reproduced bit-for-bit).
   Evaluations now pass a dedicated generator seeded with MC_SEED, so a
   report is bit-reproducible; the training loop keeps the global RNG. The
   pre-round-1 code with its global RNG seeded to MC_SEED reproduces all 20
   legacy scalars of the round-1 s4 test_id report bit-for-bit (2026-09-26).
* type accuracy, mode-usage entropy                    - doc 9.5

Aggregation of the legacy scalars (MetricAccumulator): batch values weighted
by batch size. For per-sample means (minADE, minFDE, NLL, type_acc, pi_k,
use_k) that equals the sample mean. Coverage (a ratio whose denominator is the
number of samples with valid GT at that step) and mode_entropy (a nonlinear
function of the batch usage histogram) are batch-weighted averages of
per-batch values; the pooled per-stratum tables below recompute them pooled.

REVIEW-ROUND-1 METRICS (R3 major 7, minor 1; ``extended_metrics``), used by
analysis/evaluate.py and the baselines harness, not by the training loop:

* top1ADE/top1FDE [m]: the branch with the highest pi (argmax of the logits).
* meanADE/meanFDE [m]: the mixture mean sum_k pi_k mu_kt.
  ADE = mean over valid steps; FDE = error at the LAST valid step (the same
  step minFDE uses).
* trajNLL / trajNLL_total: trajectory-level NLL with ONE branch shared over
  the whole horizon (paper Eq. (6)):
      trajNLL_total_i = -log sum_k pi_k prod_{t in V_i} N(y_t; mu_kt, Sigma_kt)
      trajNLL_i       = trajNLL_total_i / |V_i|   (same normalisation as l_mix
                                                   in model/losses.py)
  Same POS_SCALE density units as "NLL". Within a branch the steps are
  treated as independent (the model outputs per-step marginals only); for
  K = 1 trajNLL equals NLL exactly.
* ES [m]: energy score of the joint trajectory over the valid steps,
      ES = E||X - y|| - 0.5 E||X - X'||
  on the concatenated valid-step position vector (dimension 2|V_i|), from
  M = 64 draws: k ~ pi, then INDEPENDENT per-step bivariate Gaussians given k
  (assumption: the model specifies no within-branch temporal correlation).
  Unbiased estimator: the pair term averages over the M(M-1) ordered pairs.
  Draws come from a DEDICATED generator (fixed seed ES_SEED) so the legacy MC
  coverage stream is untouched; two models scored on the same index with the
  same batch size receive identical random numbers (common random numbers).
  On a different index the same sample gets other draws (MC noise ~0.1 %).
* H_pi [nats]: branch entropy -sum_k pi_k ln pi_k (max ln K).
* branch_sep_15s [m]: mean pairwise distance between the branch means at the
  15 s step (unweighted over the K(K-1)/2 pairs; 0 for K = 1).
* n_distinct: branches with pi > 0.05 whose 15 s mean lies more than
  2 sigma_iso from the 15 s mean of EVERY higher-weight branch, where
  sigma_iso = sqrt((sx^2 + sy^2) / 2) at 15 s of that higher-weight branch.
  Branches are ranked by pi (ties: lower index first). Range 1..K.

Per-sample table (SampleTable) + summarize(): every metric above, legacy and
new, per sample, so a report can be stratified by any label (track history,
type, intent, avoidance window) and two evaluations on the same index can be
joined per sample for paired comparisons (see save_per_sample/join_per_sample).

Provenance: code_fingerprint() lives here because this is the one module both
modelTraining/analysis/evaluate.py and baselines/ import.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common.constants import (COVERAGE_LEVELS, COVERAGE_MC_SAMPLES,
                              COVERAGE_STEPS, INTENT_NAMES, POS_SCALE,
                              PRED_STEPS, TYPE_NAMES)
from model.losses import gaussian_nll_steps, masked_mean

ES_SAMPLES = 64             # energy-score draws per sample
ES_SEED = 20260926          # dedicated generator seed (never the global RNG)
MC_SEED = 20260925          # coverage generator seed of evaluations (batch_metrics)
DISTINCT_PI_MIN = 0.05      # n_distinct: minimum branch weight
DISTINCT_SIGMA_MULT = 2.0   # n_distinct: separation in sigma_iso units
SEP_STEP = PRED_STEPS - 1   # 0-based step index of the 15 s horizon


@torch.no_grad()
def batch_metrics(out: dict, batch: dict, mc_samples: int = COVERAGE_MC_SAMPLES,
                  generator: torch.Generator | None = None) -> dict:
    """Legacy metric set of one batch (module doc).

    ``generator`` (optional) feeds the MC-HPD coverage draws. None = the
    global RNG of the device, the historical behaviour (training loop);
    evaluations pass make_generator(device, MC_SEED) so a report reproduces
    bit-for-bit. Only the random draws differ -- the estimator is the same.
    """
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
    pi_all = torch.softmax(out["mode_logits"].float(), dim=1)        # [B,K]
    pi_mean = pi_all.mean(dim=0)

    # ------------------------------------------------------------- coverage
    cov_flags: dict = {}
    cov = _mc_coverage(out, y, y_valid, log_pi, mc_samples, flags=cov_flags,
                       generator=generator)

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
    # per-sample arrays for stratified aggregation by the caller. The first
    # five keys are the historical ones (MetricAccumulator reads only those);
    # the rest carry the per-sample inputs of every other legacy scalar so the
    # evaluation can stratify them (review round 1). Nothing above changed.
    res["_per_sample"] = {
        "minADE": (min_ade).cpu(), "minFDE": (min_fde).cpu(),
        "NLL": nll.cpu(), "type_label": batch["type_label"].cpu(),
        "intent": batch.get("meta_intent", torch.full((B,), -1)).cpu(),
        "type_pred": type_pred.cpu(),
        "winner": winner.cpu(),
        "pi": pi_all.cpu(),
        "cov": cov_flags["cov"],
        "reacq": batch.get("meta_reacq", torch.full((B,), -1)).cpu(),
    }
    return res


@torch.no_grad()
def _mc_coverage(out: dict, y: torch.Tensor, y_valid: torch.Tensor,
                 log_pi: torch.Tensor, S: int, flags: dict | None = None,
                 generator: torch.Generator | None = None) -> dict:
    """Monte-Carlo HPD coverage at COVERAGE_LEVELS x COVERAGE_STEPS.

    ``flags`` (optional dict) receives ``flags["cov"]``: int8 CPU tensor
    [B, len(COVERAGE_LEVELS), len(COVERAGE_STEPS)] with 1 = inside, 0 =
    outside, -1 = no valid GT at that step. Filling it draws no random
    numbers. ``generator`` None = global RNG (see batch_metrics).
    """
    mu = out["mu"].float(); log_sig = out["log_sigma"].float(); rho = out["rho"].float()
    B, K = mu.shape[:2]
    res = {}
    fl = torch.full((B, len(COVERAGE_LEVELS), len(COVERAGE_STEPS)), -1,
                    dtype=torch.int8, device=mu.device)
    for hi, step in enumerate(COVERAGE_STEPS):
        m = mu[:, :, step]                       # [B,K,2]
        ls = log_sig[:, :, step]                 # [B,K,2]
        r = rho[:, :, step]                      # [B,K]
        valid = y_valid[:, step] > 0.5           # [B]
        if valid.sum() == 0:
            continue
        # sample S points from the mixture per batch row
        comp = torch.multinomial(log_pi.exp().clamp_min(1e-9), S, replacement=True,
                                 generator=generator)                             # [B,S]
        idx = comp[:, :, None]
        m_s = torch.gather(m, 1, idx.expand(B, S, 2))
        ls_s = torch.gather(ls, 1, idx.expand(B, S, 2))
        r_s = torch.gather(r, 1, comp)
        sig_s = ls_s.exp()
        e1 = torch.randn(B, S, device=m.device, generator=generator)
        e2 = torch.randn(B, S, device=m.device, generator=generator)
        x1 = m_s[..., 0] + sig_s[..., 0] * e1
        x2 = m_s[..., 1] + sig_s[..., 1] * (r_s * e1 + torch.sqrt((1 - r_s ** 2).clamp_min(1e-6)) * e2)
        samples = torch.stack([x1, x2], dim=-1)  # [B,S,2]

        pdf_samples = _mixture_logpdf(samples, m, ls, r, log_pi)     # [B,S]
        pdf_gt = _mixture_logpdf(y[:, None, step], m, ls, r, log_pi)[:, 0]  # [B]

        for li, p in enumerate(COVERAGE_LEVELS):
            thresh = torch.quantile(pdf_samples, 1.0 - p, dim=1)     # [B]
            inside = (pdf_gt >= thresh) & valid
            res[f"cov{int(p*100)}_{(step+1)//2}s"] = float(inside.float().sum() / valid.float().sum())
            fl[:, li, hi] = torch.where(valid, inside.to(torch.int8),
                                        torch.full_like(fl[:, li, hi], -1))
    if flags is not None:
        flags["cov"] = fl.cpu()
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


# ========================================================= review round 1
# R3 major 7 / minor 1: top-1 and mixture-mean errors, trajectory-level NLL,
# energy score, branch-weight diagnostics; per-sample tables for strata and
# paired comparisons. Nothing here touches the global RNG.

def make_generator(device: torch.device | str, seed: int = ES_SEED) -> torch.Generator:
    """Fresh dedicated generator on ``device`` (evaluations create one per
    run: seed MC_SEED for the coverage draws, ES_SEED for the energy score),
    so a run is reproducible on its own."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g


@torch.no_grad()
def extended_metrics(out: dict, batch: dict, gen: torch.Generator,
                     es_samples: int = ES_SAMPLES) -> dict:
    """Per-sample review-round-1 metrics (module doc) -> dict of CPU tensors [B].

    ``gen`` must live on the device of ``out`` (make_generator(device)).
    """
    y, yv = batch["y"].float(), batch["y_valid"].float()           # [B,T,2],[B,T]
    mu = out["mu"].float()                                           # [B,K,T,2]
    log_sig = out["log_sigma"].float()
    rho = out["rho"].float()
    logits = out["mode_logits"].float()
    B, K, T = mu.shape[:3]
    dev = mu.device
    log_pi = F.log_softmax(logits, dim=1)
    pi = log_pi.exp()
    n_valid = yv.sum(dim=1)                                          # [B]
    last = (yv.cumsum(dim=1).argmax(dim=1)).clamp_min(0)             # as minFDE
    rows = torch.arange(B, device=dev)

    # ------------------------------------------------ top-1 / mixture mean
    err = torch.linalg.vector_norm(mu - y.unsqueeze(1), dim=-1)      # [B,K,T]
    top = logits.argmax(dim=1)                                       # [B]
    top_err = err[rows, top]                                         # [B,T]
    top1_ade = masked_mean(top_err, yv, dim=1) * POS_SCALE
    top1_fde = top_err.gather(1, last[:, None]).squeeze(1) * POS_SCALE
    mbar = (pi[:, :, None, None] * mu).sum(dim=1)                    # [B,T,2]
    mean_err = torch.linalg.vector_norm(mbar - y, dim=-1)            # [B,T]
    mean_ade = masked_mean(mean_err, yv, dim=1) * POS_SCALE
    mean_fde = mean_err.gather(1, last[:, None]).squeeze(1) * POS_SCALE

    # ------------------------------------------------ trajectory-level NLL
    nll_steps = gaussian_nll_steps(y, out)                           # [B,K,T]
    sum_nll = (nll_steps * yv.unsqueeze(1)).sum(dim=2)               # [B,K]
    traj_total = -torch.logsumexp(log_pi - sum_nll, dim=1)           # [B]
    traj_step = traj_total / n_valid.clamp_min(1.0)

    # ------------------------------------------------ branch diagnostics
    h_pi = -(pi * log_pi).sum(dim=1)                                 # nats
    m15 = mu[:, :, SEP_STEP]                                         # [B,K,2]
    dist = torch.linalg.vector_norm(m15[:, :, None] - m15[:, None, :], dim=-1)  # [B,K,K]
    if K > 1:
        iu = torch.triu_indices(K, K, offset=1, device=dev)
        sep = dist[:, iu[0], iu[1]].mean(dim=1) * POS_SCALE
    else:
        sep = torch.zeros(B, device=dev)
    s15 = log_sig[:, :, SEP_STEP].exp()                              # [B,K,2]
    sig_iso = torch.sqrt(0.5 * (s15[..., 0] ** 2 + s15[..., 1] ** 2))  # [B,K]
    # rank by the logits (the order of pi without float ties from softmax)
    order = torch.sort(logits, dim=1, descending=True, stable=True).indices  # [B,K]
    n_distinct = torch.zeros(B, device=dev)
    for j in range(K):
        bj = order[:, j]
        ok = pi[rows, bj] > DISTINCT_PI_MIN
        for i in range(j):
            bi = order[:, i]
            ok = ok & (dist[rows, bi, bj] > DISTINCT_SIGMA_MULT * sig_iso[rows, bi])
        n_distinct = n_distinct + ok.float()

    # ------------------------------------------------ energy score
    M = int(es_samples)
    u = torch.rand(B, M, generator=gen, device=dev)
    cdf = pi.cumsum(dim=1)                                           # [B,K]
    comp = (u[:, :, None] > cdf[:, None, :]).sum(dim=-1).clamp_max(K - 1)  # [B,M]
    z = torch.randn(B, M, T, 2, generator=gen, device=dev)
    gi = comp[:, :, None, None].expand(B, M, T, 2)
    mu_s = mu.gather(1, gi)                                          # [B,M,T,2]
    sg_s = log_sig.exp().gather(1, gi)
    r_s = rho.gather(1, comp[:, :, None].expand(B, M, T))            # [B,M,T]
    x1 = mu_s[..., 0] + sg_s[..., 0] * z[..., 0]
    x2 = mu_s[..., 1] + sg_s[..., 1] * (r_s * z[..., 0]
                                        + torch.sqrt((1 - r_s ** 2).clamp_min(1e-6)) * z[..., 1])
    X = torch.stack([x1, x2], dim=-1)                                # [B,M,T,2]
    w = yv[:, None, :, None]
    term1 = ((X - y[:, None]) * w).flatten(2).norm(dim=-1).mean(dim=1)   # [B]
    Xf = (X * w).flatten(2)                                          # invalid steps -> 0
    Xf = Xf - Xf.mean(dim=1, keepdim=True)                           # distances unchanged
    D = torch.cdist(Xf, Xf, compute_mode="donot_use_mm_for_euclid_dist")  # [B,M,M]
    term2 = D.sum(dim=(1, 2)) / max(1, M * (M - 1))
    es = (term1 - 0.5 * term2) * POS_SCALE

    return {"top1ADE": top1_ade.cpu(), "top1FDE": top1_fde.cpu(),
            "meanADE": mean_ade.cpu(), "meanFDE": mean_fde.cpu(),
            "trajNLL": traj_step.cpu(), "trajNLL_total": traj_total.cpu(),
            "n_valid": n_valid.cpu(), "ES": es.cpu(), "H_pi": h_pi.cpu(),
            "branch_sep_15s": sep.cpu(), "n_distinct": n_distinct.cpu(),
            "top1": top.cpu()}


# ---------------------------------------------------------- per-sample table

#: per-sample float metrics, all aggregated as plain means over samples
MEAN_KEYS = ("minADE", "minFDE", "NLL", "top1ADE", "top1FDE", "meanADE",
             "meanFDE", "trajNLL", "trajNLL_total", "ES", "H_pi",
             "branch_sep_15s", "n_distinct", "n_valid")
_INT_KEYS = {"type_label": np.int16, "intent": np.int16, "type_pred": np.int16,
             "winner": np.int8, "top1": np.int8, "reacq": np.int8}


class SampleTable:
    """Column store of per-sample metrics (legacy + extended), in the order
    the samples were evaluated."""

    def __init__(self) -> None:
        self._cols: dict[str, list] = defaultdict(list)

    def add(self, per_sample: dict, ext: dict | None = None) -> None:
        for src in (per_sample, ext or {}):
            for k, v in src.items():
                self._cols[k].append(v.detach().cpu().numpy()
                                     if isinstance(v, torch.Tensor) else np.asarray(v))

    def __len__(self) -> int:
        c = self._cols.get("minADE")
        return int(sum(len(x) for x in c)) if c else 0

    def finalize(self) -> dict[str, np.ndarray]:
        out = {}
        for k, parts in self._cols.items():
            a = np.concatenate(parts, axis=0)
            if k in _INT_KEYS:
                a = a.astype(_INT_KEYS[k])
            elif k == "cov":
                a = a.astype(np.int8)
            else:
                a = a.astype(np.float32)
            out[k] = a
        return out


def summarize(cols: dict[str, np.ndarray], sel: np.ndarray | None = None) -> dict:
    """Every metric, pooled over the selected samples (float64 accumulation).

    Coverage = inside / samples with valid GT at that step; mode_entropy =
    normalised entropy of the pooled winner histogram (module doc: the legacy
    "overall" values average these per batch instead).
    """
    n_all = len(cols["minADE"])
    sel = np.ones(n_all, bool) if sel is None else np.asarray(sel, bool)
    n = int(sel.sum())
    res: dict = {"n": n}
    if n == 0:
        return res
    for k in MEAN_KEYS:
        if k in cols:
            res[k] = float(np.mean(cols[k][sel].astype(np.float64)))
    if "type_pred" in cols:
        res["type_acc"] = float(np.mean(cols["type_pred"][sel] == cols["type_label"][sel]))
    if "pi" in cols:
        K = cols["pi"].shape[1]
        pim = cols["pi"][sel].astype(np.float64).mean(axis=0)
        usage = np.bincount(cols["winner"][sel].astype(np.int64), minlength=K)[:K] / n
        u = np.clip(usage, 1e-9, None)
        res["mode_entropy"] = float(-(u * np.log(u)).sum() / math.log(max(2, K)))
        for k in range(K):
            res[f"pi_{k}"] = float(pim[k])
            res[f"use_{k}"] = float(usage[k])
        if "n_distinct" in cols:
            nd = np.rint(cols["n_distinct"][sel]).astype(np.int64)
            res["n_distinct_frac"] = {str(c): float(np.mean(nd == c)) for c in range(1, K + 1)}
    if "cov" in cols:
        c = cols["cov"][sel]
        for li, p in enumerate(COVERAGE_LEVELS):
            for hi, step in enumerate(COVERAGE_STEPS):
                v = c[:, li, hi]
                nv = int((v >= 0).sum())
                res[f"cov{int(p*100)}_{(step+1)//2}s"] = (float((v == 1).sum() / nv)
                                                         if nv else None)
    return res


def stratify(cols: dict[str, np.ndarray], labels: np.ndarray,
             names: dict | None = None, skip_negative: bool = False) -> dict:
    """{stratum name: summarize(...)} for every distinct label value."""
    labels = np.asarray(labels)
    out = {}
    for v in np.unique(labels).tolist():
        if skip_negative and v < 0:
            continue
        key = names.get(int(v), str(v)) if names else str(v)
        out[key] = summarize(cols, labels == v)
    return out


def standard_strata(cols: dict[str, np.ndarray],
                    avoid: np.ndarray | None = None) -> dict:
    """The report's stratified tables: all / reacq / type / intent / avoid."""
    st = {"all": summarize(cols),
          "reacq": stratify(cols, cols["reacq"]),
          "type": stratify(cols, cols["type_label"],
                           {i: n for i, n in enumerate(TYPE_NAMES)}),
          "intent": stratify(cols, cols["intent"],
                             {i: n for i, n in enumerate(INTENT_NAMES)})}
    if avoid is not None:
        st["avoid"] = stratify(cols, avoid, {-1: "unknown", 0: "inactive",
                                             1: "active"})
    return st


def overall_extended(cols: dict[str, np.ndarray]) -> dict:
    """New headline scalars for report["overall"] (pooled sample means)."""
    s = summarize(cols)
    keys = ("top1ADE", "top1FDE", "meanADE", "meanFDE", "trajNLL",
            "trajNLL_total", "ES", "H_pi", "branch_sep_15s", "n_distinct")
    return {k: s[k] for k in keys if k in s}


METRIC_DEFINITIONS = {
    "minADE/minFDE": "m; best of K branch means (min over branches of the "
                     "valid-step mean error / error at the last valid step)",
    "NLL": "per-step MARGINAL mixture NLL: mean over valid steps of "
           "-log sum_k pi_k N(y_t; mu_kt, Sigma_kt) (branches re-mixed at every "
           "step), then mean over samples; density per (POS_SCALE m)^2 = per "
           "(100 m)^2 (add 2 ln 100 = 9.2103 per step for per m^2)",
    "top1ADE/top1FDE": "m; branch with the highest pi",
    "meanADE/meanFDE": "m; mixture mean sum_k pi_k mu_kt",
    "trajNLL": "trajectory NLL, one branch shared over the horizon: "
               "-log sum_k pi_k prod_{t valid} N(y_t; mu_kt, Sigma_kt), divided "
               "by the number of valid steps (as l_mix); same units as NLL; "
               "steps independent within a branch",
    "trajNLL_total": "the same, not divided (per-sample total)",
    "ES": "m; energy score of the concatenated valid-step trajectory, "
          f"M={ES_SAMPLES} draws (k ~ pi, independent per-step Gaussians given "
          f"k), unbiased pair term, dedicated generator seed {ES_SEED}",
    "H_pi": "nats; branch entropy -sum pi ln pi",
    "branch_sep_15s": "m; mean pairwise distance between branch means at 15 s "
                      "(0 for K=1)",
    "n_distinct": f"branches with pi > {DISTINCT_PI_MIN} whose 15 s mean is more "
                  f"than {DISTINCT_SIGMA_MULT} sigma_iso (of the higher-weight "
                  "branch; sigma_iso = sqrt((sx^2+sy^2)/2)) from every "
                  "higher-weight branch's 15 s mean",
    "covP_Hs": "MC-HPD coverage (128 draws/sample from a dedicated generator "
               f"seeded {MC_SEED}; pre-round-1 reports used the unseeded global "
               "RNG); 'overall' averages per-batch fractions weighted by batch "
               "size (legacy), 'strata' pool inside/valid over the stratum",
    "mode_entropy": "normalised entropy of the winner (argmin ADE) histogram; "
                    "'overall' = batch-weighted mean of per-batch values "
                    "(legacy), 'strata' = pooled histogram",
    "reacq": "0 = history window holds only detections of the track open at "
             "t0; 1 = it also holds detections of an earlier, closed track of "
             "the same object (reacquired); 2 = no open track at t0 (slot "
             "eligibility only); -1 = no track ids stored",
    "avoid": "1 = the target's reaction layer is active at any logged step "
             "t0+1..t0+30 (labels/<run>/<episode>.npz: reaction_active [T,N]); "
             "0 = never active; -1 = no label file for the episode",
}


# ------------------------------------------------------------ strata labels

def load_avoid_labels(files: list[str], file_idx: np.ndarray, t0: np.ndarray,
                      agent: np.ndarray, labels_dir: str) -> tuple[np.ndarray, dict]:
    """Avoidance-window stratum per sample (METRIC_DEFINITIONS["avoid"]).

    Label file of an episode <run_dir>/episodes/<name>.npz is
    <labels_dir>/<basename(run_dir)>/<name>.npz holding ``reaction_active``
    bool [T, N] per logged step. A missing file marks the episode's samples
    -1 (counted); a file without the key or too short raises.
    """
    lab = np.full(len(t0), -1, np.int8)
    stats = {"labels_dir": str(labels_dir).replace("\\", "/"), "n_episodes": 0,
             "n_episodes_missing": 0, "missing_examples": []}
    file_idx = np.asarray(file_idx)
    order = np.argsort(file_idx, kind="stable")
    bounds = np.flatnonzero(np.diff(file_idx[order])) + 1
    for grp in np.split(order, bounds):
        if grp.size == 0:
            continue
        path = files[int(file_idx[grp[0]])]
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        lp = os.path.join(labels_dir, run_name, os.path.basename(path))
        stats["n_episodes"] += 1
        if not os.path.exists(lp):
            stats["n_episodes_missing"] += 1
            if len(stats["missing_examples"]) < 5:
                stats["missing_examples"].append(lp.replace("\\", "/"))
            continue
        with np.load(lp) as z:
            if "reaction_active" not in z.files:
                raise KeyError(f"{lp}: no 'reaction_active' array")
            ra = np.asarray(z["reaction_active"]).astype(bool)
        ts = np.asarray(t0)[grp].astype(np.int64)
        ags = np.asarray(agent)[grp].astype(np.int64)
        if ra.ndim != 2 or ra.shape[0] < int(ts.max()) + 1 + PRED_STEPS \
                or ra.shape[1] <= int(ags.max()):
            raise ValueError(f"{lp}: reaction_active shape {ra.shape} does not "
                             f"cover t0+{PRED_STEPS} / agent {int(ags.max())}")
        cs = np.concatenate([np.zeros((1, ra.shape[1]), np.int64),
                             np.cumsum(ra, axis=0, dtype=np.int64)])
        n_act = cs[ts + 1 + PRED_STEPS, ags] - cs[ts + 1, ags]     # steps t0+1..t0+30
        lab[grp] = (n_act > 0).astype(np.int8)
    stats["n_samples_unknown"] = int((lab < 0).sum())
    stats["n_samples_active"] = int((lab == 1).sum())
    stats["n_samples_inactive"] = int((lab == 0).sum())
    return lab, stats


# ------------------------------------------------------ per-sample file i/o

#: arrays written by save_per_sample (besides the key arrays)
PER_SAMPLE_KEYS = ("minADE", "minFDE", "NLL", "top1ADE", "top1FDE", "meanADE",
                   "meanFDE", "trajNLL", "trajNLL_total", "n_valid", "ES",
                   "H_pi", "branch_sep_15s", "n_distinct", "cov", "reacq",
                   "type_label", "type_pred", "intent", "winner", "top1", "pi")


def save_per_sample(path: str, files: list[str], file_idx: np.ndarray,
                    t0: np.ndarray, agent: np.ndarray, cols: dict,
                    meta: dict, extra: dict | None = None) -> str:
    """Write one evaluation's per-sample arrays, keyed by (file, t0, agent).

    ``files`` is the episode-path table and ``file_idx`` indexes it; ``cov``
    is int8 [S, levels, horizons] (1 in / 0 out / -1 no valid GT) with the
    axes given by ``cov_levels`` / ``cov_horizons_s``. ``meta_json`` holds the
    evaluation's provenance. Loads with plain np.load (no pickle).
    """
    arrays = {"files": np.asarray([str(f).replace("\\", "/") for f in files], dtype=str),
              "file_idx": np.asarray(file_idx, np.int32),
              "t0": np.asarray(t0, np.int32), "agent": np.asarray(agent, np.int16),
              "cov_levels": np.asarray(COVERAGE_LEVELS, np.float32),
              "cov_horizons_s": np.asarray([(s + 1) // 2 for s in COVERAGE_STEPS], np.int16),
              "meta_json": np.asarray(json.dumps(meta, default=str))}
    for k in PER_SAMPLE_KEYS:
        if k in cols:
            arrays[k] = cols[k]
    for k, v in (extra or {}).items():
        arrays[k] = np.asarray(v)
    n = len(arrays["t0"])
    for k, v in arrays.items():
        if k not in ("files", "cov_levels", "cov_horizons_s", "meta_json") \
                and v.shape[0] != n:
            raise ValueError(f"per-sample array {k} has {v.shape[0]} rows, expected {n}")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_per_sample(path: str) -> dict:
    """np.load a per-sample file into a plain dict (+ parsed ``meta``)."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.pop("meta_json")))
    return d


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def sample_keys(d: dict) -> list[tuple]:
    """(normalised episode path, t0, agent) per sample of a per-sample dict."""
    files = [_norm_path(f) for f in d["files"]]
    return [(files[int(f)], int(t), int(a))
            for f, t, a in zip(d["file_idx"], d["t0"], d["agent"])]


def join_per_sample(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (ia, ib) of the samples present in BOTH per-sample dicts,
    matched on (episode path, t0, agent), in the order of ``a``. Use for
    paired comparisons, e.g. a[\"minADE\"][ia] - b[\"minADE\"][ib]."""
    pos_b = {k: i for i, k in enumerate(sample_keys(b))}
    ia, ib = [], []
    for i, k in enumerate(sample_keys(a)):
        j = pos_b.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.asarray(ia, np.int64), np.asarray(ib, np.int64)


# ---------------------------------------------------------------- provenance

FP_ROOTS = ("modelTraining", "modelVal", "baselines")


def code_fingerprint(base: str | None = None, roots: tuple = FP_ROOTS) -> dict:
    """SHA-256 over the sorted .py files of modelTraining, modelVal and
    baselines (``__pycache__`` excluded), dataGen's source_fingerprint
    convention: for each file in sorted relative-path order, hash
    relpath (from ``base``, '/'-separated) + NUL + file bytes. Also one digest
    per root, so a report can be matched per subsystem."""
    base = base or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths: list[str] = []
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(os.path.join(base, r)):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in {"__pycache__", ".git", ".idea"})
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    paths.append(os.path.join(dirpath, fn))
    rels = sorted((os.path.relpath(p, base).replace("\\", "/"), p) for p in paths)
    h = hashlib.sha256()
    per_root = {r: hashlib.sha256() for r in roots}
    n_bytes = 0
    for rel, p in rels:
        with open(p, "rb") as fh:
            data = fh.read()
        for hh in (h, per_root[rel.split("/", 1)[0]]):
            hh.update(rel.encode("utf-8"))
            hh.update(b"\0")
            hh.update(data)
        n_bytes += len(data)
    return {"sha256": h.hexdigest(), "n_files": len(rels), "n_bytes": n_bytes,
            "roots": list(roots),
            "per_root": {r: v.hexdigest() for r, v in per_root.items()},
            "definition": "sha256 over sorted relpath + NUL + bytes of every .py "
                          "under the roots, __pycache__ excluded"}


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
