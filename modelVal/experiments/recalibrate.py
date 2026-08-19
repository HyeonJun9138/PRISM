"""Post-hoc temperature (variance) recalibration for the PRISM predictor.

Fits ONE scalar T on the OOD-adjacent CALIBRATION runs (never the test set):
sigma -> T * sigma, chosen to minimize mixture NLL on the calibration set.
Reports before/after NLL and HPD coverage on the OOD TEST runs.

Sanity anchor: at T=1 the script must reproduce the official eval numbers
(cov90@15s ~ 0.758, NLL ~ 2.29 for full5x_s4) -- if it does not, the
implementation, not the model, is wrong.

  python -m experiments.recalibrate
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.constants import HIST_STEPS, POS_SCALE  # noqa: E402
from data.dataset import build_sample  # noqa: E402
from model.network import build_model  # noqa: E402

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

MODEL = f"{PRISM_ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"
CAL_RUNS = sorted(glob.glob(rf"{PRISM_ROOT}/Logs/Dataset/*test_ood_cal_*"))
TEST_RUNS = [rf"{PRISM_ROOT}/Logs/Dataset/20260813_023139_test_ood_full5x_a",
             rf"{PRISM_ROOT}/Logs/Dataset/20260813_023139_test_ood_full5x_b"]
T0_STRIDE = 4            # match training-index stride; keeps runtime sane
BATCH = 512
HPD_MC = 96              # MC samples per (sample, step) for the HPD threshold
HORIZON_STEPS = {5: 9, 10: 19, 15: 29}
NOMINALS = (0.5, 0.7, 0.9)
OUT = rf"{PRISM_ROOT}/results/recalibration.json"


def iter_samples(run_dirs, data_cfg):
    min_obs = max(2, int(data_cfg.get("min_obs", 3)))
    for run in run_dirs:
        for f in sorted(glob.glob(os.path.join(run, "episodes", "*.npz"))):
            ep = dict(np.load(f))
            det = ep["obs_detected"].astype(bool)
            gtv = ep["gt_valid"].astype(bool)
            T, N = det.shape
            for t0 in range(HIST_STEPS - 1, T - 30, T0_STRIDE):
                win = det[t0 - HIST_STEPS + 1: t0 + 1]
                n_det = win.sum(axis=0)
                for a in range(N):
                    if n_det[a] < min_obs or not gtv[t0, a]:
                        continue
                    s = build_sample(ep, t0, a, data_cfg, with_labels=True)
                    if s["y_valid"].sum() < 27:
                        continue
                    yield s


def forward_all(model, device, samples):
    """Returns concatenated (mu, sigma, rho, pi, y, y_valid) tensors."""
    outs = {k: [] for k in ("mu", "sigma", "rho", "pi", "y", "yv")}
    buf_x, buf_m, buf_y, buf_yv = [], [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.as_tensor(np.stack(buf_x)).to(device)
        m = torch.as_tensor(np.stack(buf_m)).to(device)
        with torch.no_grad():
            o = model(x, m)
        outs["mu"].append(o["mu"].float().cpu())
        outs["sigma"].append(o["log_sigma"].float().exp().cpu())
        outs["rho"].append(o["rho"].float().cpu())
        outs["pi"].append(torch.softmax(o["mode_logits"].float(), 1).cpu())
        outs["y"].append(torch.as_tensor(np.stack(buf_y)))
        outs["yv"].append(torch.as_tensor(np.stack(buf_yv)))
        buf_x.clear(); buf_m.clear(); buf_y.clear(); buf_yv.clear()

    for s in samples:
        buf_x.append(s["x"]); buf_m.append(s["agent_mask"])
        buf_y.append(s["y"]); buf_yv.append(s["y_valid"])
        if len(buf_x) >= BATCH:
            flush()
    flush()
    return {k: torch.cat(v) for k, v in outs.items()}


def mixture_nll(d, T=1.0):
    """Step-mean mixture NLL (scaled units), masked by y_valid."""
    mu, sig, rho, pi, y, yv = (d["mu"], d["sigma"] * T, d["rho"], d["pi"],
                               d["y"], d["yv"])
    B, K, S, _ = mu.shape
    dx = (y[:, None, :, 0] - mu[..., 0]) / sig[..., 0]
    dy = (y[:, None, :, 1] - mu[..., 1]) / sig[..., 1]
    omr = 1 - rho ** 2
    z = (dx ** 2 - 2 * rho * dx * dy + dy ** 2) / omr
    logN = -(z / 2) - torch.log(2 * math.pi * sig[..., 0] * sig[..., 1]
                                * torch.sqrt(omr))
    w = torch.log(pi[:, :, None].clamp_min(1e-9)) + logN      # [B,K,S]
    step_ll = torch.logsumexp(w, dim=1)                        # [B,S]
    mask = yv.bool()
    return float((-step_ll[mask]).mean())


def coverage(d, T=1.0, gen=None):
    """HPD MC coverage at NOMINALS x HORIZON_STEPS."""
    mu, sig, rho, pi, y, yv = (d["mu"], d["sigma"] * T, d["rho"], d["pi"],
                               d["y"], d["yv"])
    B, K, S, _ = mu.shape
    gen = gen or torch.Generator().manual_seed(7)
    out = {}
    for h_lab, s_idx in HORIZON_STEPS.items():
        m, sg, rh = mu[:, :, s_idx], sig[:, :, s_idx], rho[:, :, s_idx]
        valid = yv[:, s_idx].bool()
        # draw component indices then correlated normals
        comp = torch.multinomial(pi, HPD_MC, replacement=True)     # [B,M]
        idx = comp[..., None]
        mm = torch.gather(m, 1, idx.expand(-1, -1, 2))             # [B,M,2]
        ss = torch.gather(sg, 1, idx.expand(-1, -1, 2))
        rr = torch.gather(rh, 1, comp)
        e1 = torch.randn(B, HPD_MC, generator=gen)
        e2 = torch.randn(B, HPD_MC, generator=gen)
        sx = mm[..., 0] + ss[..., 0] * e1
        sy = mm[..., 1] + ss[..., 1] * (rr * e1
                                        + torch.sqrt(1 - rr ** 2) * e2)

        def dens(px, py):
            dx = (px[:, :, None] - m[:, None, :, 0]) / sg[:, None, :, 0]
            dy = (py[:, :, None] - m[:, None, :, 1]) / sg[:, None, :, 1]
            r = rh[:, None, :]
            omr = 1 - r ** 2
            z = (dx ** 2 - 2 * r * dx * dy + dy ** 2) / omr
            logN = -(z / 2) - torch.log(
                2 * math.pi * sg[:, None, :, 0] * sg[:, None, :, 1]
                * torch.sqrt(omr))
            return torch.logsumexp(
                torch.log(pi[:, None, :].clamp_min(1e-9)) + logN, dim=2)

        samp_d = dens(sx, sy)                                      # [B,M]
        gt_d = dens(y[:, None, s_idx, 0], y[:, None, s_idx, 1])[:, 0]
        for nom in NOMINALS:
            thr = torch.quantile(samp_d, 1 - nom, dim=1)
            cov = (gt_d >= thr)[valid].float().mean()
            out[f"cov{int(nom*100)}_{h_lab}s"] = round(float(cov), 4)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(MODEL, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or ck["cfg"]["model"]
    data_cfg = dict(ck.get("data_cfg") or ck["cfg"]["data"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"cal runs: {len(CAL_RUNS)}, forwarding...", flush=True)
    cal = forward_all(model, device, iter_samples(CAL_RUNS, data_cfg))
    print(f"cal samples: {cal['mu'].shape[0]}", flush=True)

    # golden-section on NLL(T)
    lo, hi = 0.8, 2.6
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = mixture_nll(cal, a), mixture_nll(cal, b)
    for _ in range(24):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = mixture_nll(cal, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = mixture_nll(cal, b)
    T = round((lo + hi) / 2, 3)
    print(f"fitted T = {T}  (cal NLL {mixture_nll(cal, 1.0):.4f} -> "
          f"{mixture_nll(cal, T):.4f})", flush=True)

    print("test runs forwarding...", flush=True)
    test = forward_all(model, device, iter_samples(TEST_RUNS, data_cfg))
    print(f"test samples: {test['mu'].shape[0]}", flush=True)

    res = {"model": MODEL, "T": T,
           "cal_runs": CAL_RUNS, "t0_stride": T0_STRIDE,
           "n_cal": int(cal["mu"].shape[0]),
           "n_test": int(test["mu"].shape[0]),
           "before": {"NLL": round(mixture_nll(test, 1.0), 4),
                      **coverage(test, 1.0)},
           "after": {"NLL": round(mixture_nll(test, T), 4),
                     **coverage(test, T)}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["before"], indent=1))
    print(json.dumps(res["after"], indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
