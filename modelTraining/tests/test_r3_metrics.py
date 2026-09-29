"""Review-round-1 metric tests (R3 major 7, minor 1). Standalone runner:

    python tests/test_r3_metrics.py

Known-answer checks for the metrics added to model/metrics.py:
  1. legacy scalars: batch_metrics is value-identical to the verbatim
     pre-round-1 implementation (same global-RNG draws), the new per-sample
     coverage flags aggregate to the legacy fractions, and a dedicated
     generator leaves the global RNG untouched
  2. single Gaussian (K=1): top-1 / mean / min errors, trajNLL == NLL,
     closed-form NLL, H=0, separation 0, one distinct branch, energy score
     against the closed forms (central chi, dimension 60 and 2; noncentral
     Rice, dimension 2)
  3. two-branch mixture: top-1 vs mixture-mean vs best-of-K errors, the
     per-step MARGINAL "NLL" and the trajectory NLL against numpy references
     (they differ), entropy, separation, distinct-branch rule incl. the
     higher-weight sigma, the pi threshold and a 3-branch chain; energy score
     against an independent numpy Monte-Carlo reference (correlated branch)
  4. per-sample table: pooled summaries, strata, save/load/join round trip
  5. avoidance labels: window t0+1..t0+30, missing files -> -1 and counted
  6. code fingerprint: deterministic, sensitive to one byte
  7. run_evaluation end-to-end on synthetic data: index-rule override,
     provenance keys, strata, per-sample file, labels, reproducibility
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.constants import (COVERAGE_LEVELS, COVERAGE_STEPS,  # noqa: E402
                              POS_SCALE, PRED_STEPS)
from model.losses import gaussian_nll_steps, masked_mean  # noqa: E402
from model.metrics import (SampleTable, batch_metrics,  # noqa: E402
                           code_fingerprint, extended_metrics, join_per_sample,
                           load_avoid_labels, load_per_sample, make_generator,
                           save_per_sample, standard_strata, summarize)

T = PRED_STEPS
LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------ verbatim pre-round-1 code
# Copied from model/metrics.py as of 2026-08-13 (pre_d0 backup). Do not edit:
# it is the reference the legacy scalars must keep matching.

@torch.no_grad()
def _legacy_batch_metrics(out, batch, mc_samples):
    y, y_valid = batch["y"].float(), batch["y_valid"].float()
    mu = out["mu"].float()
    B, K = mu.shape[:2]
    err = torch.linalg.vector_norm(mu - y.unsqueeze(1), dim=-1)
    ade = masked_mean(err, y_valid.unsqueeze(1), dim=2)
    min_ade = ade.min(dim=1).values * POS_SCALE
    last = (y_valid.cumsum(dim=1).argmax(dim=1)).clamp_min(0)
    fde_all = err.gather(2, last[:, None, None].expand(B, K, 1)).squeeze(2)
    min_fde = fde_all.min(dim=1).values * POS_SCALE
    nll_steps = gaussian_nll_steps(y, out)
    log_pi = F.log_softmax(out["mode_logits"].float(), dim=1)
    mix_step_ll = torch.logsumexp(log_pi[:, :, None] - nll_steps, dim=1)
    nll = -masked_mean(mix_step_ll, y_valid, dim=1)
    winner = ade.argmin(dim=1)
    usage = torch.bincount(winner, minlength=K).float() / max(1, B)
    entropy = float(-(usage.clamp_min(1e-9) * usage.clamp_min(1e-9).log()).sum()
                    / math.log(max(2, K)))
    pi_mean = torch.softmax(out["mode_logits"].float(), dim=1).mean(dim=0)
    cov = _legacy_mc_coverage(out, y, y_valid, log_pi, mc_samples)
    type_pred = out["type_logits"].argmax(dim=1)
    type_acc = (type_pred == batch["type_label"]).float().mean()
    res = {"minADE": float(min_ade.mean()), "minFDE": float(min_fde.mean()),
           "NLL": float(nll.mean()), "type_acc": float(type_acc),
           "mode_entropy": entropy, "n": int(B)}
    for k in range(K):
        res[f"pi_{k}"] = float(pi_mean[k])
        res[f"use_{k}"] = float(usage[k])
    res.update(cov)
    res["_per_sample"] = {"minADE": min_ade, "minFDE": min_fde, "NLL": nll}
    return res


@torch.no_grad()
def _legacy_mc_coverage(out, y, y_valid, log_pi, S):
    mu = out["mu"].float(); log_sig = out["log_sigma"].float(); rho = out["rho"].float()
    B, K = mu.shape[:2]
    res = {}
    for step in COVERAGE_STEPS:
        m = mu[:, :, step]; ls = log_sig[:, :, step]; r = rho[:, :, step]
        valid = y_valid[:, step] > 0.5
        if valid.sum() == 0:
            continue
        comp = torch.multinomial(log_pi.exp().clamp_min(1e-9), S, replacement=True)
        idx = comp[:, :, None]
        m_s = torch.gather(m, 1, idx.expand(B, S, 2))
        ls_s = torch.gather(ls, 1, idx.expand(B, S, 2))
        r_s = torch.gather(r, 1, comp)
        sig_s = ls_s.exp()
        e1 = torch.randn(B, S, device=m.device)
        e2 = torch.randn(B, S, device=m.device)
        x1 = m_s[..., 0] + sig_s[..., 0] * e1
        x2 = m_s[..., 1] + sig_s[..., 1] * (r_s * e1 + torch.sqrt((1 - r_s ** 2).clamp_min(1e-6)) * e2)
        samples = torch.stack([x1, x2], dim=-1)
        pdf_samples = _legacy_logpdf(samples, m, ls, r, log_pi)
        pdf_gt = _legacy_logpdf(y[:, None, step], m, ls, r, log_pi)[:, 0]
        for p in COVERAGE_LEVELS:
            thresh = torch.quantile(pdf_samples, 1.0 - p, dim=1)
            inside = (pdf_gt >= thresh) & valid
            res[f"cov{int(p*100)}_{(step+1)//2}s"] = float(inside.float().sum() / valid.float().sum())
    return res


def _legacy_logpdf(x, mu, log_sig, rho, log_pi):
    x = x[:, :, None, :]
    m = mu[:, None]
    sig = log_sig[:, None].exp()
    r = rho[:, None]
    zx = (x[..., 0] - m[..., 0]) / sig[..., 0]
    zy = (x[..., 1] - m[..., 1]) / sig[..., 1]
    omr2 = (1 - r ** 2).clamp_min(1e-6)
    q = (zx ** 2 - 2 * r * zx * zy + zy ** 2) / omr2
    log_n = -0.5 * q - log_sig[:, None, :, 0] - log_sig[:, None, :, 1] \
            - 0.5 * torch.log(omr2) - math.log(2 * math.pi)
    return torch.logsumexp(log_pi[:, None] + log_n, dim=2)


# ------------------------------------------------------------------ helpers

def _random_out(B, K, seed):
    g = torch.Generator().manual_seed(seed)
    mu = torch.cumsum(torch.randn(B, K, T, 2, generator=g) * 0.2, dim=2)
    return {"mu": mu,
            "log_sigma": torch.randn(B, K, T, 2, generator=g) * 0.3 - 1.0,
            "rho": torch.tanh(torch.randn(B, K, T, generator=g)) * 0.9,
            "mode_logits": torch.randn(B, K, generator=g),
            "type_logits": torch.randn(B, 7, generator=g)}


def _batch(y, y_valid, B):
    return {"y": y, "y_valid": y_valid,
            "type_label": torch.randint(0, 7, (B,), generator=torch.Generator().manual_seed(5)),
            "meta_intent": torch.zeros(B, dtype=torch.long),
            "meta_reacq": torch.zeros(B, dtype=torch.long)}


def _const_out(mu_k, sig_k, rho_k, pi, B):
    """Broadcast per-branch trajectories [K,T,2] / sigmas [K,T,2] / rho [K,T]
    to a batch of identical samples."""
    mu = torch.as_tensor(mu_k, dtype=torch.float32)[None].expand(B, -1, -1, -1).clone()
    sig = torch.as_tensor(sig_k, dtype=torch.float32)[None].expand(B, -1, -1, -1).clone()
    rho = torch.as_tensor(rho_k, dtype=torch.float32)[None].expand(B, -1, -1).clone()
    logits = torch.log(torch.as_tensor(pi, dtype=torch.float32))[None].expand(B, -1).clone()
    return {"mu": mu, "log_sigma": sig.log(), "rho": rho, "mode_logits": logits,
            "type_logits": torch.zeros(B, 7)}


def _np_logpdf(y, mu, sx, sy, r):
    zx, zy = (y[0] - mu[0]) / sx, (y[1] - mu[1]) / sy
    q = (zx * zx - 2 * r * zx * zy + zy * zy) / (1 - r * r)
    return -0.5 * q - math.log(sx * sy) - 0.5 * math.log(1 - r * r) - LOG2PI


def _logsumexp(a):
    a = np.asarray(a, np.float64)
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


def _i1(x, terms=60):
    return sum((x / 2) ** (2 * k + 1) / (math.factorial(k) * math.factorial(k + 1))
               for k in range(terms))


def _rice_mean(nu, s):
    x = -nu * nu / (2 * s * s)
    lag = math.exp(x / 2) * ((1 - x) * float(np.i0(-x / 2)) - x * _i1(-x / 2))
    return s * math.sqrt(math.pi / 2) * lag


def _close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: {a!r} vs {b!r} (tol {tol})"


# -------------------------------------------------------------------- tests

def test_legacy_unchanged() -> None:
    B, K = 300, 3
    out = _random_out(B, K, 1)
    y = torch.cumsum(torch.randn(B, T, 2, generator=torch.Generator().manual_seed(2)) * 0.2, 1)
    yv = torch.ones(B, T)
    yv[:40, 27:] = 0.0          # some samples lose GT before 15 s
    yv[40:45, 9:] = 0.0         # and a few before 5 s
    batch = _batch(y, yv, B)
    torch.manual_seed(123)
    ref = _legacy_batch_metrics(out, batch, 64)
    torch.manual_seed(123)
    new = batch_metrics(out, batch, mc_samples=64)
    for k, v in ref.items():
        if k == "_per_sample":
            continue
        assert k in new and new[k] == v, f"legacy scalar {k} changed: {v!r} -> {new.get(k)!r}"
    for k in ("minADE", "minFDE", "NLL"):
        assert torch.equal(ref["_per_sample"][k], new["_per_sample"][k]), k
    # per-sample coverage flags aggregate to the legacy batch fractions
    fl = new["_per_sample"]["cov"]
    for li, p in enumerate(COVERAGE_LEVELS):
        for hi, s in enumerate(COVERAGE_STEPS):
            v = fl[:, li, hi]
            frac = float((v == 1).sum()) / float((v >= 0).sum())
            # legacy divides in float32
            _close(frac, new[f"cov{int(p*100)}_{(s+1)//2}s"], 1e-6, "flag aggregation")
            assert int((v >= 0).sum()) == int((yv[:, s] > 0.5).sum())
    # a dedicated generator: reproducible and the global RNG is not touched
    state = torch.get_rng_state()
    a = batch_metrics(out, batch, 64, generator=make_generator("cpu", 7))
    b = batch_metrics(out, batch, 64, generator=make_generator("cpu", 7))
    assert torch.equal(torch.get_rng_state(), state), "generator path drew from the global RNG"
    for k in a:
        if k != "_per_sample":
            assert a[k] == b[k], f"seeded coverage not reproducible: {k}"
    ext_state = torch.get_rng_state()
    extended_metrics(out, batch, make_generator("cpu", 9))
    assert torch.equal(torch.get_rng_state(), ext_state), "extended_metrics drew from the global RNG"


def test_single_gaussian() -> None:
    B = 4000
    s = 0.4                                          # 40 m, isotropic
    t = torch.arange(1, T + 1, dtype=torch.float32)
    mu0 = torch.stack([t * 0.1, t * 0.05], -1)       # [T,2]
    d = torch.tensor([0.3, 0.4])                     # 50 m offset at every step
    out = _const_out(mu0[None], torch.full((1, T, 2), s), torch.zeros(1, T), [1.0], B)
    y = (mu0 + d)[None].expand(B, -1, -1).clone()
    yv = torch.ones(B, T)
    yv[:, 27:] = 0.0                                 # 27 valid steps
    batch = _batch(y, yv, B)
    leg = batch_metrics(out, batch, 16, generator=make_generator("cpu", 1))["_per_sample"]
    ext = extended_metrics(out, batch, make_generator("cpu", 2))
    for k in ("top1ADE", "meanADE", "top1FDE", "meanFDE"):
        _close(float(ext[k].mean()), 50.0, 1e-3, k)
    _close(float(leg["minADE"].mean()), 50.0, 1e-3, "minADE")
    nll_ref = 0.5 * 0.25 / s ** 2 + 2 * math.log(s) + LOG2PI
    _close(float(leg["NLL"][0]), nll_ref, 1e-5, "legacy NLL closed form")
    _close(float(ext["trajNLL"][0]), nll_ref, 1e-5, "trajNLL closed form")
    assert torch.allclose(ext["trajNLL"], leg["NLL"], atol=1e-5), "K=1: trajNLL must equal NLL"
    _close(float(ext["trajNLL_total"][0]), 27 * nll_ref, 1e-3, "trajNLL_total")
    _close(float(ext["n_valid"][0]), 27.0, 0, "n_valid")
    _close(float(ext["H_pi"].abs().max()), 0.0, 1e-6, "H(pi) of one branch")
    _close(float(ext["branch_sep_15s"].abs().max()), 0.0, 0, "separation K=1")
    assert torch.all(ext["n_distinct"] == 1), "K=1 must count one distinct branch"

    # energy score, central: y = mu -> ES = s c_D (1 - 1/sqrt 2), D = 2|V|
    def c_d(D):
        return math.sqrt(2) * math.exp(math.lgamma((D + 1) / 2) - math.lgamma(D / 2))

    batch_c = _batch(mu0[None].expand(B, -1, -1).clone(), torch.ones(B, T), B)
    es = extended_metrics(out, batch_c, make_generator("cpu", 3))["ES"]
    ref = POS_SCALE * s * c_d(2 * T) * (1 - 1 / math.sqrt(2))
    _close(float(es.mean()), ref, 0.01 * ref, "ES central, D=60")
    yv1 = torch.zeros(B, T); yv1[:, 5] = 1.0          # one valid step: D = 2
    es = extended_metrics(out, _batch(batch_c["y"], yv1, B), make_generator("cpu", 4))["ES"]
    ref = POS_SCALE * s * c_d(2) * (1 - 1 / math.sqrt(2))
    _close(float(es.mean()), ref, 0.02 * ref, "ES central, D=2")
    # noncentral, D = 2: E||X-y|| is the Rice mean, E||X-X'|| = s sqrt(pi)
    nu = 0.6
    y_nc = batch_c["y"].clone(); y_nc[:, 5, 0] += nu
    es = extended_metrics(out, _batch(y_nc, yv1, B), make_generator("cpu", 5))["ES"]
    ref = POS_SCALE * (_rice_mean(nu, s) - 0.5 * s * math.sqrt(math.pi))
    _close(float(es.mean()), ref, 0.02 * ref, "ES noncentral (Rice), D=2")


def _two_branch(pi, s0, s1, rho1=0.0, B=8):
    t = np.arange(1, T + 1, dtype=np.float64)
    mu0 = np.stack([t * 0.1, np.zeros(T)], -1)
    mu1 = np.stack([t * 0.1, t * 0.02], -1)            # 0.6 (= 60 m) apart at 15 s
    sig = np.stack([np.broadcast_to(np.asarray(s0, float), (T, 2)),
                    np.broadcast_to(np.asarray(s1, float), (T, 2))])
    rho = np.stack([np.zeros(T), np.full(T, rho1)])
    out = _const_out(np.stack([mu0, mu1]), sig, rho, pi, B)
    return out, mu0, mu1, sig, rho


def test_two_branch() -> None:
    B = 8
    pi = [0.7, 0.3]
    out, mu0, mu1, sig, rho = _two_branch(pi, 0.2, 0.3, B=B)
    y = torch.as_tensor(mu1, dtype=torch.float32)[None].expand(B, -1, -1).clone()
    batch = _batch(y, torch.ones(B, T), B)
    leg = batch_metrics(out, batch, 16, generator=make_generator("cpu", 1))["_per_sample"]
    ext = extended_metrics(out, batch, make_generator("cpu", 2))
    gap = np.hypot(*(mu0 - mu1).T) * POS_SCALE        # [T] metres
    _close(float(leg["minADE"][0]), 0.0, 1e-4, "minADE (branch 1 exact)")
    _close(float(ext["top1ADE"][0]), gap.mean(), 1e-3, "top1ADE = branch 0")
    _close(float(ext["top1FDE"][0]), 60.0, 1e-3, "top1FDE")
    _close(float(ext["meanADE"][0]), 0.3 * 0 + 0.7 * gap.mean(), 1e-3, "meanADE")
    _close(float(ext["meanFDE"][0]), 0.7 * 60.0, 1e-3, "meanFDE")
    # numpy references: per-step marginal (legacy "NLL") vs shared branch
    lp = np.array([[_np_logpdf(mu1[t], [mu0, mu1][k][t], sig[k, t, 0], sig[k, t, 1], rho[k, t])
                    for t in range(T)] for k in range(2)])            # [K,T]
    marg = np.mean([-_logsumexp(np.log(pi) + lp[:, t]) for t in range(T)])
    traj_total = -_logsumexp(np.log(pi) + lp.sum(axis=1))
    _close(float(leg["NLL"][0]), marg, 1e-4, "legacy NLL = per-step marginal mixture NLL")
    _close(float(ext["trajNLL_total"][0]), traj_total, 2e-3, "trajNLL_total")
    _close(float(ext["trajNLL"][0]), traj_total / T, 1e-4, "trajNLL per step")
    assert abs(marg - traj_total / T) > 0.05, "fixture should separate the two NLLs"
    _close(float(ext["H_pi"][0]), -(0.7 * math.log(0.7) + 0.3 * math.log(0.3)), 1e-5, "H(pi)")
    _close(float(ext["branch_sep_15s"][0]), 60.0, 1e-3, "separation at 15 s")
    assert float(ext["n_distinct"][0]) == 2, "0.6 apart > 2*0.2 of the top branch"
    # the HIGHER-weight branch's sigma sets the bar: 2*0.35 = 0.7 > 0.6
    out2 = _two_branch(pi, 0.35, 0.05, B=B)[0]
    assert float(extended_metrics(out2, batch, make_generator("cpu", 3))["n_distinct"][0]) == 1
    # anisotropic sigma_iso = sqrt((0.3^2 + 0.1^2)/2) = 0.2236 -> 0.447 < 0.6
    out3 = _two_branch(pi, [0.3, 0.1], 0.3, B=B)[0]
    assert float(extended_metrics(out3, batch, make_generator("cpu", 3))["n_distinct"][0]) == 2
    # a far branch below the pi threshold does not count
    out4 = _two_branch([0.96, 0.04], 0.2, 0.3, B=B)[0]
    assert float(extended_metrics(out4, batch, make_generator("cpu", 3))["n_distinct"][0]) == 1
    # 3 branches: A(0.5) at 0, B(0.3) at 1.0, C(0.2) at 1.1 on the x axis
    for sB, expect in ((0.2, 2), (0.01, 3)):
        mus = np.zeros((3, T, 2)); mus[1, :, 0] = 1.0; mus[2, :, 0] = 1.1
        sig3 = np.full((3, T, 2), 0.2); sig3[1] = sB
        o = _const_out(mus, sig3, np.zeros((3, T)), [0.5, 0.3, 0.2], B)
        got = float(extended_metrics(o, batch, make_generator("cpu", 3))["n_distinct"][0])
        assert got == expect, f"3-branch chain (sigma_B={sB}): {got} != {expect}"
        sep = np.mean([1.0, 1.1, 0.1]) * POS_SCALE
        _close(float(extended_metrics(o, batch, make_generator("cpu", 3))["branch_sep_15s"][0]),
               sep, 1e-3, "3-branch mean pairwise separation")


def test_energy_mixture() -> None:
    """Estimator (M=64, many replicas) vs an independent numpy MC reference
    for a two-branch mixture with a correlated branch."""
    pi = np.array([0.6, 0.4])
    out, mu0, mu1, sig, rho = _two_branch(list(pi), [0.25, 0.15], [0.2, 0.35], rho1=0.6, B=3000)
    y = 0.5 * (mu0 + mu1)                              # between the branches
    yv = np.ones(T); yv[25:] = 0.0                     # 25 valid steps
    batch = _batch(torch.as_tensor(y, dtype=torch.float32)[None].expand(3000, -1, -1).clone(),
                   torch.as_tensor(yv, dtype=torch.float32)[None].expand(3000, -1).clone(), 3000)
    es = float(extended_metrics(out, batch, make_generator("cpu", 11))["ES"].mean())

    rng = np.random.default_rng(0)
    V = yv > 0.5

    def draw(n):
        k = (rng.random(n) > pi[0]).astype(int)       # P(k=1) = pi[1]
        mus = np.stack([mu0, mu1])[k]                  # [n,T,2]
        s = sig[k]; r = rho[k]
        z1, z2 = rng.standard_normal((n, T)), rng.standard_normal((n, T))
        x1 = mus[..., 0] + s[..., 0] * z1
        x2 = mus[..., 1] + s[..., 1] * (r * z1 + np.sqrt(1 - r * r) * z2)
        return np.stack([x1, x2], -1)[:, V].reshape(n, -1)

    yf = y[V].reshape(-1)
    s1 = s2 = 0.0
    n_chunks, n = 8, 50_000                            # 400k independent pairs
    for _ in range(n_chunks):
        X, X2 = draw(n), draw(n)
        s1 += np.linalg.norm(X - yf, axis=1).sum()
        s2 += np.linalg.norm(X - X2, axis=1).sum()
    ref = POS_SCALE * (s1 - 0.5 * s2) / (n_chunks * n)
    _close(es, ref, 0.01 * ref, "ES two-branch mixture vs numpy MC")


def test_table_strata_io(tmp: str) -> None:
    rng = np.random.default_rng(3)
    S, K = 500, 3
    tab = SampleTable()
    reacq = rng.integers(0, 3, S)
    for lo in range(0, S, 128):
        hi = min(S, lo + 128); b = hi - lo
        cov = rng.integers(-1, 2, (b, 3, 3)).astype(np.int8)
        per = {"minADE": torch.as_tensor(rng.random(b) * 100), "minFDE": torch.as_tensor(rng.random(b)),
               "NLL": torch.as_tensor(rng.random(b)), "type_label": torch.as_tensor(rng.integers(1, 4, b)),
               "intent": torch.as_tensor(rng.integers(0, 3, b)), "type_pred": torch.as_tensor(rng.integers(1, 4, b)),
               "winner": torch.as_tensor(rng.integers(0, K, b)),
               "pi": torch.softmax(torch.as_tensor(rng.random((b, K))), 1),
               "cov": torch.as_tensor(cov), "reacq": torch.as_tensor(reacq[lo:hi])}
        ext = {k: torch.as_tensor(rng.random(b)) for k in
               ("top1ADE", "top1FDE", "meanADE", "meanFDE", "trajNLL", "trajNLL_total",
                "ES", "H_pi", "branch_sep_15s")}
        ext["n_distinct"] = torch.as_tensor(rng.integers(1, K + 1, b)).float()
        ext["n_valid"] = torch.full((b,), 30.0)
        ext["top1"] = torch.as_tensor(rng.integers(0, K, b))
        tab.add(per, ext)
    cols = tab.finalize()
    assert len(cols["minADE"]) == S and cols["cov"].shape == (S, 3, 3)
    st = standard_strata(cols)
    assert sum(v["n"] for v in st["reacq"].values()) == S
    w = sum(v["n"] * v["minADE"] for v in st["reacq"].values()) / S
    _close(w, st["all"]["minADE"], 1e-9, "strata recombine to the pooled mean")
    c = cols["cov"][:, 2, 2]
    _close(st["all"]["cov90_15s"], float((c == 1).sum() / (c >= 0).sum()), 1e-12, "pooled coverage")
    assert abs(sum(st["all"]["n_distinct_frac"].values()) - 1.0) < 1e-9
    assert summarize(cols, np.zeros(S, bool)) == {"n": 0}

    # save / load / join: b is a reversed subset of a
    files = [f"D:/fake/run/episodes/ep_{i:06d}.npz" for i in range(5)]
    fi = rng.integers(0, 5, S).astype(np.int32)
    t0 = rng.integers(19, 90, S).astype(np.int32)
    ag = np.arange(S, dtype=np.int16)                  # unique keys
    pa = save_per_sample(os.path.join(tmp, "a.npz"), files, fi, t0, ag, cols, {"model": "a"})
    sub = np.arange(S)[::-1][:300]
    cols_b = {k: v[sub] for k, v in cols.items()}
    files_b = [f.replace("/", "\\") for f in files]    # other spelling, same files
    pb = save_per_sample(os.path.join(tmp, "b.npz"), files_b, fi[sub], t0[sub], ag[sub],
                         cols_b, {"model": "b"}, extra={"avoid": np.zeros(300, np.int8)})
    A, Bd = load_per_sample(pa), load_per_sample(pb)
    assert A["meta"]["model"] == "a" and "avoid" in Bd
    ia, ib = join_per_sample(A, Bd)
    assert len(ia) == 300, f"join found {len(ia)} of 300"
    assert np.array_equal(A["minADE"][ia], Bd["minADE"][ib])
    assert np.array_equal(A["cov"][ia], Bd["cov"][ib])


def test_avoid_labels(tmp: str) -> None:
    run = os.path.join(tmp, "20990101_000000_test_id_x")
    lab_root = os.path.join(tmp, "labels")
    os.makedirs(os.path.join(run, "episodes")); os.makedirs(os.path.join(lab_root, os.path.basename(run)))
    files = [os.path.join(run, "episodes", f"ep_{i:06d}.npz").replace("\\", "/") for i in range(3)]
    ra = np.zeros((120, 4), bool)
    ra[50, 1] = True                    # agent 1 active only at step 50
    np.savez_compressed(os.path.join(lab_root, os.path.basename(run), "ep_000000.npz"),
                        reaction_active=ra)
    np.savez_compressed(os.path.join(lab_root, os.path.basename(run), "ep_000001.npz"),
                        reaction_active=np.ones((120, 4), bool))
    # samples: (file, t0, agent)
    fi = np.array([0, 0, 0, 0, 1, 2, 2], np.int32)
    t0 = np.array([20, 19, 49, 50, 30, 30, 40], np.int32)
    ag = np.array([1, 1, 1, 1, 0, 0, 3], np.int16)
    lab, st = load_avoid_labels(files, fi, t0, ag, lab_root)
    # t0=20: window 21..50 contains 50 -> 1; t0=19: 20..49 -> 0; t0=49: 50..79 -> 1;
    # t0=50: 51..80 -> 0; file 1 all active -> 1; file 2 missing -> -1
    assert lab.tolist() == [1, 0, 1, 0, 1, -1, -1], lab.tolist()
    assert st["n_episodes"] == 3 and st["n_episodes_missing"] == 1
    assert st["n_samples_unknown"] == 2 and st["n_samples_active"] == 3
    bad = os.path.join(lab_root, os.path.basename(run), "ep_000002.npz")
    np.savez_compressed(bad, reaction_active=np.zeros((60, 4), bool))   # too short
    try:
        load_avoid_labels(files, fi, t0, ag, lab_root)
        raise AssertionError("a too-short label file must raise")
    except ValueError:
        pass


def test_code_fingerprint(tmp: str) -> None:
    for r in ("modelTraining", "modelVal", "baselines"):
        os.makedirs(os.path.join(tmp, r, "pkg", "__pycache__"))
        with open(os.path.join(tmp, r, "pkg", "m.py"), "w") as f:
            f.write(f"X = '{r}'\n")
        with open(os.path.join(tmp, r, "pkg", "__pycache__", "m.cpython-310.py"), "w") as f:
            f.write("ignored\n")
    a = code_fingerprint(tmp)
    assert a == code_fingerprint(tmp) and a["n_files"] == 3, a
    with open(os.path.join(tmp, "modelVal", "pkg", "m.py"), "a") as f:
        f.write("#\n")
    b = code_fingerprint(tmp)
    assert b["sha256"] != a["sha256"]
    assert b["per_root"]["modelTraining"] == a["per_root"]["modelTraining"]
    assert b["per_root"]["modelVal"] != a["per_root"]["modelVal"]


def test_run_evaluation(tmp: str) -> None:
    from data.dataset import build_sample_index
    from data.synthetic import make_synthetic_run
    from model.network import build_model
    from training.job import JobState

    run = make_synthetic_run(os.path.join(tmp, "20990101_000000_synth"), n_episodes=4, seed=11)
    data_cfg = {"max_agents": 6, "t0_stride": 6, "min_obs": 3, "min_future_valid": 27}
    model_cfg = {"d_model": 32, "n_modes": 3, "temporal_layers": 1, "agent_layers": 1,
                 "n_heads": 4, "d_ff": 64, "dropout": 0.0, "cumulative_mu": True}
    torch.manual_seed(0)
    ck = os.path.join(tmp, "model.pt")
    torch.save({"model": build_model(model_cfg).state_dict(), "model_cfg": model_cfg,
                "data_cfg": data_cfg}, ck)               # legacy config: no track_rule
    lab = os.path.join(tmp, "labels", os.path.basename(run))
    os.makedirs(lab)
    ra = np.zeros((120, 5), bool); ra[60:, :] = True
    np.savez_compressed(os.path.join(lab, "ep_000000.npz"), reaction_active=ra)

    from analysis.evaluate import run_evaluation
    reports = {}
    for rule in ("slot", "current"):
        jd = os.path.join(tmp, f"job_{rule}")
        os.makedirs(jd)
        job = JobState(f"test_{rule}", jd, kind="eval")
        out_json = os.path.join(tmp, "out", f"rep_{rule}.json")
        ps = os.path.join(tmp, "out", f"ps_{rule}.npz")
        run_evaluation(job, ck, [run], batch_size=64, index_rule=rule, out_path=out_json,
                       per_sample_path=ps, labels_dir=os.path.join(tmp, "labels"))
        assert job.status == "finished", (job.status, job.error)
        with open(out_json, encoding="utf-8") as f:
            rep = json.load(f)
        with open(os.path.join(jd, "eval_report.json"), encoding="utf-8") as f:
            assert json.load(f)["overall"] == rep["overall"]
        assert rep["index_rule"] == rule and rep["feature_rule"] == "slot"
        n_idx = len(build_sample_index([run], dict(data_cfg, index_track_rule=rule)))
        assert rep["n_samples"] == n_idx == rep["strata"]["all"]["n"], (rep["n_samples"], n_idx)
        assert rep["complete"] and len(rep["code_fp"]["sha256"]) == 64
        for k in ("minADE", "NLL", "cov90_15s", "top1ADE", "meanADE", "trajNLL", "ES",
                  "H_pi", "branch_sep_15s", "n_distinct"):
            assert k in rep["overall"], k
        for k in ("minADE", "minFDE", "NLL", "type_acc", "pi_0", "use_1"):
            ref = rep["overall"][k]            # float32 batch means vs float64 pooled
            _close(rep["strata"]["all"][k], ref, 1e-5 * abs(ref) + 1e-6, f"pooled vs legacy {k}")
        st = rep["avoid_labels"]
        assert st["n_episodes_missing"] == 3 and st["n_samples_unknown"] > 0
        assert sum(v["n"] for v in rep["strata"]["avoid"].values()) == rep["n_samples"]
        assert set(rep["strata"]["reacq"]) <= ({"0", "1", "2"} if rule == "slot" else {"0", "1"})
        d = load_per_sample(ps)
        assert len(d["t0"]) == rep["n_samples"] and d["meta"]["index_rule"] == rule
        reports[rule] = (rep, d)
    # the current index is a subset of the slot index; every shared sample
    # gets the same deterministic metrics (same model, same features)
    (rs, ds), (rc, dc) = reports["slot"], reports["current"]
    ia, ib = join_per_sample(dc, ds)
    assert len(ia) == rc["n_samples"] <= rs["n_samples"]
    assert np.allclose(dc["minADE"][ia], ds["minADE"][ib], rtol=1e-4, atol=1e-3)
    assert np.allclose(dc["trajNLL"][ia], ds["trajNLL"][ib], rtol=1e-4, atol=1e-4)
    # seeded coverage + ES: re-running gives the identical report numbers
    jd = os.path.join(tmp, "job_again"); os.makedirs(jd)
    job = JobState("test_again", jd, kind="eval")
    run_evaluation(job, ck, [run], batch_size=64, index_rule="slot")
    with open(os.path.join(jd, "eval_report.json"), encoding="utf-8") as f:
        again = json.load(f)
    for k, v in rs["overall"].items():
        assert again["overall"][k] == v, f"not reproducible: {k} {v} vs {again['overall'][k]}"
    # never overwrite: a second --out to the same path lands next to it
    job2 = JobState("test_again2", jd, kind="eval")
    run_evaluation(job2, ck, [run], batch_size=64, index_rule="slot",
                   out_path=os.path.join(tmp, "out", "rep_slot.json"))
    assert len([f for f in os.listdir(os.path.join(tmp, "out")) if f.startswith("rep_slot")]) == 2


# ------------------------------------------------------------------- runner

def run_all(verbose: bool = True) -> bool:
    tests = [("legacy_unchanged", test_legacy_unchanged, False),
             ("single_gaussian", test_single_gaussian, False),
             ("two_branch", test_two_branch, False),
             ("energy_mixture", test_energy_mixture, False),
             ("table_strata_io", test_table_strata_io, True),
             ("avoid_labels", test_avoid_labels, True),
             ("code_fingerprint", test_code_fingerprint, True),
             ("run_evaluation", test_run_evaluation, True)]
    ok = True
    for name, fn, needs_tmp in tests:
        tmp = tempfile.mkdtemp(prefix=f"prism_r3_{name}_")
        try:
            fn(tmp) if needs_tmp else fn()
            if verbose:
                print(f"  PASS  {name}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
