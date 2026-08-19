"""Score the predictive distribution against branch-rollout TRUE distributions.

For every rollout group (episode, origin step t0, target slot) the generator
re-simulated the SAME past under re-seeded decision randomness B times: the
branch endpoints are an empirical sample of the true conditional future
distribution. We score the model's mixture against that sample:

  * conditional HPD coverage: what fraction of the B true futures fall inside
    the model's q-mass region, PER CONDITION -> distribution over conditions
    (median / IQR), the honest form of "is 90% really 90% here?";
  * density-PIT: rank of each branch's mixture density among mixture MC
    samples -- uniform iff the branches are distributed like the mixture.

Optionally applies the fitted temperature (variance scaling) to show
conditional calibration before/after recalibration on OOD.

  python -m experiments.rollout_score --run <rollout run dir> [--T 1.665]
"""
from __future__ import annotations

import argparse
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
HORIZONS = {"5s": 9, "10s": 19, "15s": 29}
NOMINALS = (0.5, 0.7, 0.9)
HPD_MC = 256


def load_model(device):
    ck = torch.load(MODEL, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or ck["cfg"]["model"]
    data_cfg = dict(ck.get("data_cfg") or ck["cfg"]["data"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, data_cfg


def mixture_logpdf(pts, mu, sig, rho, pi):
    """pts [P,2]; mu/sig [K,2]; rho/pi [K] -> log mixture density [P]."""
    dx = (pts[:, None, 0] - mu[None, :, 0]) / sig[None, :, 0]
    dy = (pts[:, None, 1] - mu[None, :, 1]) / sig[None, :, 1]
    omr = 1 - rho[None, :] ** 2
    z = (dx ** 2 - 2 * rho[None, :] * dx * dy + dy ** 2) / omr
    logN = -(z / 2) - torch.log(2 * math.pi * sig[None, :, 0]
                                * sig[None, :, 1] * torch.sqrt(omr))
    return torch.logsumexp(torch.log(pi[None, :].clamp_min(1e-9)) + logN, 1)


def sample_mixture(mu, sig, rho, pi, n, gen):
    comp = torch.multinomial(pi, n, replacement=True)
    m, s, r = mu[comp], sig[comp], rho[comp]
    e1 = torch.randn(n, generator=gen)
    e2 = torch.randn(n, generator=gen)
    x = m[:, 0] + s[:, 0] * e1
    y = m[:, 1] + s[:, 1] * (r * e1 + torch.sqrt(1 - r ** 2) * e2)
    return torch.stack([x, y], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg = load_model(device)
    min_obs = max(2, int(data_cfg.get("min_obs", 3)))
    gen = torch.Generator().manual_seed(11)

    cov_vals = {h: {n: [] for n in NOMINALS} for h in HORIZONS}
    pit_all = {h: [] for h in HORIZONS}
    n_groups = n_scored = 0

    ep_files = sorted(glob.glob(os.path.join(args.run, "episodes", "*.npz")))
    for epf in ep_files:
        rof = epf.replace(os.sep + "episodes" + os.sep,
                          os.sep + "rollouts" + os.sep) \
                 .replace(".npz", "_rollouts.npz")
        if not os.path.exists(rof):
            continue
        ep = dict(np.load(epf))
        ro = np.load(rof)
        det = ep["obs_detected"].astype(bool)
        gtv = ep["gt_valid"].astype(bool)

        for g in range(len(ro["origin_step"])):
            n_groups += 1
            t0 = int(ro["origin_step"][g])
            a = int(ro["agent_slot"][g])
            if t0 < HIST_STEPS - 1 or not gtv[t0, a]:
                continue
            win = det[t0 - HIST_STEPS + 1: t0 + 1]
            if win[:, a].sum() < min_obs:
                continue
            s = build_sample(ep, t0, a, data_cfg, with_labels=False)
            x = torch.as_tensor(s["x"][None]).to(device)
            m = torch.as_tensor(s["agent_mask"][None]).to(device)
            with torch.no_grad():
                o = model(x, m)
            mu = o["mu"][0].float().cpu() * POS_SCALE          # [K,30,2] m
            sig = o["log_sigma"][0].float().exp().cpu() * POS_SCALE * args.T
            rho = o["rho"][0].float().cpu()
            pi = torch.softmax(o["mode_logits"][0].float(), 0).cpu()

            ox, oy, th = s["frame"]
            c, sn = math.cos(th), math.sin(th)
            R_inv = np.array([[c, -sn], [sn, c]])
            n_scored += 1

            for h_lab, s_idx in HORIZONS.items():
                bp = ro["pos"][g, :, s_idx, :]                 # [B,2] world
                bv = ro["valid"][g, :, s_idx].astype(bool)
                if bv.sum() < 8:
                    continue
                bf = (bp[bv] - np.array([ox, oy])) @ R_inv     # -> frame [m]
                pts = torch.as_tensor(bf, dtype=torch.float32)

                mu_s, sig_s = mu[:, s_idx], sig[:, s_idx]
                rho_s = rho[:, s_idx]
                mc = sample_mixture(mu_s, sig_s, rho_s, pi, HPD_MC, gen)
                d_mc = mixture_logpdf(mc, mu_s, sig_s, rho_s, pi)
                d_br = mixture_logpdf(pts, mu_s, sig_s, rho_s, pi)

                for nom in NOMINALS:
                    thr = torch.quantile(d_mc, 1 - nom)
                    cov_vals[h_lab][nom].append(
                        float((d_br >= thr).float().mean()))
                # density-PIT: rank of each branch among MC densities
                pit = (d_mc[None, :] < d_br[:, None]).float().mean(1)
                pit_all[h_lab].extend(pit.tolist())

    def q(v, p):
        return round(float(np.percentile(v, p)), 3) if v else None

    res = {"run": args.run, "model": MODEL, "T": args.T,
           "n_groups": n_groups, "n_scored": n_scored,
           "conditional_coverage": {
               h: {f"nom{int(n*100)}": {
                   "median": q(cov_vals[h][n], 50),
                   "q25": q(cov_vals[h][n], 25),
                   "q75": q(cov_vals[h][n], 75),
                   "mean": (round(float(np.mean(cov_vals[h][n])), 3)
                            if cov_vals[h][n] else None)}
                   for n in NOMINALS}
               for h in HORIZONS},
           "pit_hist_15s": (np.histogram(pit_all["15s"], bins=10,
                                         range=(0, 1))[0]
                            / max(1, len(pit_all["15s"]))).round(4).tolist()}

    out = args.out or os.path.join(
        rf"{PRISM_ROOT}/results",
        f"rollout_cond_{os.path.basename(args.run)}_T{args.T}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in
                      ("n_groups", "n_scored")}, indent=None))
    print("cov90@15s:", json.dumps(
        res["conditional_coverage"]["15s"]["nom90"]))
    print("PIT(15s):", res["pit_hist_15s"])
    print("saved ->", out)


if __name__ == "__main__":
    main()
