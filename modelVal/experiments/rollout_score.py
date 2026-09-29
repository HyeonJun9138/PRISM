"""Score the predictive distribution against branch-rollout TRUE distributions.

For every rollout group (episode, origin step, target slot) the generator
re-simulated the SAME past under re-seeded decision randomness B times: the
branch endpoints are an empirical sample of the simulator's conditional future
distribution. We score the model's mixture against that sample:

  * conditional HPD coverage: what fraction of the B branch futures fall inside
    the model's q-mass region, PER CONDITION -> distribution over conditions
    (median / IQR), the honest form of "is 90% really 90% here?";
  * density-PIT: rank of each branch's mixture density among mixture MC
    samples -- uniform iff the branches are distributed like the mixture.

Optionally applies a temperature (variance scaling, sigma -> T * sigma).

Timing (review round 1, R3 major 1) -- read before changing anything here
-------------------------------------------------------------------------
* episode arrays: logged index i holds t = (i + 1) * LOG_DT, so t[59] = 30.0 s
  and t[60] = 30.5 s (dataGen/simulation/engine.py logs AFTER advancing);
* rollouts: dataGen/main.py ``_write_branch_rollouts`` forks at branch_t
  (default 30.0 s) and stores ``origin_step = round(branch_t / LOG_DT)`` = 60;
  branch future k is the TRUE position at branch_t + (k + 1) * LOG_DT
  (dataGen/simulation/rollout.py ``branch_rollout``: ``advance_to(branch_t)``,
  then LOG_DECIMATION ticks per recorded step);
* model: a sample built at index t0 predicts step k at t[t0] + (k + 1) * LOG_DT.

The released script used t0 = origin_step = 60, i.e. a history ending at
30.5 s -- one observation AFTER the fork -- so model step k (31.0 + 0.5 k s)
was scored against rollout step k (30.5 + 0.5 k s): one recording step apart.
The default is now TIMESTAMP alignment: t0 is the index whose logged time
equals origin_step * LOG_DT (asserted exact), so model step k and rollout step
k both refer to t[t0] + (k + 1) * LOG_DT. ``--legacy-offset`` restores the
released t0 (same eligibility, same features, same scoring).

What the reference is (R3 major 1): all branches share the full simulator
state at the fork -- hidden manoeuvre mode, intent and reaction-layer state
included -- and only the FUTURE decision streams are re-seeded. The branch
distribution is therefore a SIMULATOR-STATE-conditioned reference, while the
predictor conditions on noisy observations only. Report these numbers as
state-conditioned diagnostics, with the number of valid futures per condition
and the bootstrap intervals written below.

Monte Carlo and reproducibility
-------------------------------
The released script drew mixture components with ``torch.multinomial``
WITHOUT a generator, i.e. from torch's default CPU generator, which on this
Windows build is seeded from the steady clock at process start
(``torch.initial_seed()`` differs on every run). Its coverage / PIT values are
therefore one unseeded Monte-Carlo realisation and cannot be reproduced bit
for bit by any rerun; only the non-MC fields (run, model, T, n_groups,
n_scored) can. Every draw here comes from a seeded torch.Generator (normals:
seed 11 as released; components: ``--comp-seed``). ``--expect`` checks the
non-MC fields EXACTLY and, with ``--mc-replicates R``, checks every MC field of
the released file against R replicate component streams (the only unseeded
input): the released value must lie within ``Z_MAX`` replicate SDs.

Sample eligibility mirrors training: >= min_obs usable detections under
``index_track_rule(data_cfg)`` (``window_detections``) and GT valid at t0; the
features follow the model's own ``track_rule`` (legacy checkpoints: "slot").

  python -m experiments.rollout_score --run <rollout run dir> --out <json>
         [--T 1.665] [--legacy-offset] [--model <ckpt>]
         [--expect <released json> --mc-replicates 100]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.constants import HIST_STEPS, LOG_DT, POS_SCALE  # noqa: E402
from data.dataset import (build_sample, index_track_rule,  # noqa: E402
                          track_rule, window_detections)
from model.network import build_model  # noqa: E402

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

MODEL = f"{PRISM_ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"
HORIZONS = {"5s": 9, "10s": 19, "15s": 29}
NOMINALS = (0.5, 0.7, 0.9)
HPD_MC = 256
#: a horizon of a group is scored only with at least this many valid branches
MIN_VALID_FUTURES = 8
PIT_BINS = 10
N_BOOT = 2000
BOOT_SEED = 20260926
#: torch.Generator seed of the standard-normal draws (the released value)
NORMAL_SEED = 11
#: torch.Generator seed of the mixture-component draws of the main result
COMPONENT_SEED = 12
#: MC-consistency gate: |released - replicate mean| <= Z_MAX replicate SDs
Z_MAX = 4.0
#: the fields of a released rollout_cond file that involve no MC draw
DETERMINISTIC_FIELDS = ("run", "model", "T", "n_groups", "n_scored")

#: never written by any experiment script (CLAUDE.md rule 5 + revision rules)
PROTECTED_DIRS = (f"{PRISM_ROOT}/Paper/draft_work", f"{PRISM_ROOT}/Logs/Dataset",
                  f"{PRISM_ROOT}/Logs/Models")
PROTECTED_JSON_DIRS = (f"{PRISM_ROOT}/Paper/rebuttal",)


# ============================================================ small helpers

def guard_output_path(path: str, overwrite: bool = False) -> str:
    """Absolute output path, refusing protected locations and silent overwrite."""
    p = os.path.abspath(path).replace("\\", "/")
    low = p.lower()
    for d in PROTECTED_DIRS:
        dl = d.lower().rstrip("/")
        if low == dl or low.startswith(dl + "/"):
            raise SystemExit(f"refusing to write {p}: {d} is protected "
                             "(released results / datasets / checkpoints)")
    for d in PROTECTED_JSON_DIRS:
        if os.path.dirname(low) == d.lower().rstrip("/") and low.endswith(".json"):
            raise SystemExit(f"refusing to write {p}: the *.json files in {d} "
                             "are released rebuttal evidence")
    if os.path.exists(p) and not overwrite:
        raise SystemExit(f"{p} already exists; pass --overwrite to replace it")
    return p


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_fields(result, expected, prefix: str = "") -> list[str]:
    """Every field of ``expected`` must exist in ``result`` with the same
    value and JSON type (extra fields in ``result`` are allowed)."""
    out: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(result, dict):
            return [f"{prefix or '/'}: expected an object, got {result!r}"]
        for k, v in expected.items():
            if k not in result:
                out.append(f"{prefix}/{k}: missing")
            else:
                out.extend(compare_fields(result[k], v, f"{prefix}/{k}"))
        return out
    if isinstance(expected, list):
        if not isinstance(result, list) or len(result) != len(expected):
            return [f"{prefix}: {result!r} != {expected!r}"]
        for i, (a, b) in enumerate(zip(result, expected)):
            out.extend(compare_fields(a, b, f"{prefix}[{i}]"))
        return out
    same_type = (type(result) is type(expected)
                 or (isinstance(result, (int, float)) and isinstance(expected, (int, float))
                     and not isinstance(result, bool) and not isinstance(expected, bool)
                     and isinstance(result, int) == isinstance(expected, int)))
    if result != expected or not same_type:
        out.append(f"{prefix}: {result!r} != {expected!r}")
    return out


def json_roundtrip(obj):
    return json.loads(json.dumps(obj))


def flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def mc_consistency(expected: dict, replicates: list[dict], unit_of,
                   z_max: float = Z_MAX, floor_of=None) -> dict:
    """Is each released MC field a plausible draw of the replicate
    distribution?

    ``unit_of(path)`` is the field's rounding unit. The replicate SD used in
    z is floored at ``floor_of(path)`` (default: half the rounding unit, so a
    field every replicate rounds to one value must match within half a unit).
    Quantiles of per-group coverage live on a lattice of step 1/B (B branches
    per group), so for them the caller floors the SD at that step: a raw z
    against a near-degenerate lattice distribution is not a test of anything.
    The raw replicate values are returned so any other criterion can be
    recomputed without rerunning."""
    E = flatten(expected)
    R = [flatten(r) for r in replicates]
    fields, worst, n_out = {}, 0.0, 0
    for path, x in E.items():
        if x is None:
            continue
        ys = np.array([r[path] for r in R], dtype=np.float64)
        unit = float(unit_of(path))
        floor = float(floor_of(path)) if floor_of else unit / 2
        mean = float(ys.mean())
        sd = float(ys.std(ddof=1)) if ys.size > 1 else 0.0
        z = (float(x) - mean) / max(sd, floor)
        outside = not (ys.min() - unit / 2 <= x <= ys.max() + unit / 2)
        tail = 2 * min(float((ys <= x + 1e-12).mean()),
                       float((ys >= x - 1e-12).mean()))
        n_out += int(outside)
        worst = max(worst, abs(z))
        fields[path] = {"released": x, "rep_mean": round(mean, 6),
                        "rep_sd": round(sd, 6), "sd_floor": floor,
                        "rep_min": float(ys.min()), "rep_max": float(ys.max()),
                        "z": round(z, 3), "empirical_two_sided_p": round(min(1.0, tail), 3),
                        "outside_replicate_range": outside,
                        "replicate_values": ys.tolist()}
    return {"n_fields": len(fields), "n_replicates": len(replicates),
            "z_max_allowed": z_max, "max_abs_z": round(worst, 3),
            "n_outside_replicate_range": n_out,
            "pass": worst <= z_max, "fields": fields}


# ========================================================== timing helpers

def aligned_t0(t, origin_step: int, log_dt: float = LOG_DT) -> int:
    """Index of the logged step whose timestamp IS the rollout fork time.

    The fork time is ``origin_step * log_dt`` (dataGen stores
    ``origin_step = round(branch_t / LOG_DT)``). Exactly one logged timestamp
    must equal it; anything else raises AssertionError rather than silently
    scoring shifted futures.
    """
    branch_t = float(origin_step) * float(log_dt)
    tt = np.asarray(t, dtype=np.float64)
    hits = np.flatnonzero(tt == branch_t)
    if hits.size != 1:
        near = tt[np.argmin(np.abs(tt - branch_t))] if tt.size else float("nan")
        raise AssertionError(
            f"rollout fork time {branch_t!r} s (origin_step {origin_step} x "
            f"LOG_DT {log_dt}) matches {hits.size} logged timestamps (nearest "
            f"{near!r} s); expected exactly one -- refusing to guess the "
            "alignment")
    return int(hits[0])


def check_step_alignment(t, t0: int, branch_t: float, steps,
                         log_dt: float = LOG_DT) -> None:
    """Assert model step k (logged index t0+1+k) and rollout step k
    (branch_t + (k+1) log_dt) carry the same timestamp."""
    tt = np.asarray(t, dtype=np.float64)
    for k in steps:
        i = int(t0) + 1 + int(k)
        if i >= tt.shape[0]:
            continue
        want = float(branch_t) + (int(k) + 1) * float(log_dt)
        if tt[i] != want:
            raise AssertionError(
                f"model step {k} is at t[{i}] = {tt[i]!r} s but rollout step "
                f"{k} is at {want!r} s")


def rollout_path(run: str, episode_file: str) -> str:
    name = os.path.basename(episode_file).replace(".npz", "_rollouts.npz")
    return os.path.join(run, "rollouts", name)


# ================================================================== model

def load_model(device, path: str = MODEL):
    ck = torch.load(path, map_location=device, weights_only=False)
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


def sample_mixture(mu, sig, rho, pi, n, gen, comp_gen):
    comp = torch.multinomial(pi, n, replacement=True, generator=comp_gen)
    m, s, r = mu[comp], sig[comp], rho[comp]
    e1 = torch.randn(n, generator=gen)
    e2 = torch.randn(n, generator=gen)
    x = m[:, 0] + s[:, 0] * e1
    y = m[:, 1] + s[:, 1] * (r * e1 + torch.sqrt(1 - r ** 2) * e2)
    return torch.stack([x, y], 1)


# ================================================================ scoring

def forward_pass(run: str, model, data_cfg: dict, device, legacy_offset: bool,
                 log_every: int = 25):
    """Eligibility + one forward per rollout group (released loop order).

    Returns (cache, info); ``cache`` holds per scored group the mixture
    parameters (sigma at T = 1) and the branch points in the target frame, so
    the Monte-Carlo pass can be repeated without the network.
    """
    min_obs = max(2, int(data_cfg.get("min_obs", 3)))
    elig_rule = index_track_rule(data_cfg)
    skips = {"no_full_history": 0, "gt_invalid_at_t0": 0,
             "too_few_detections": 0}
    n_groups = n_scored = n_files = 0
    t0_seen: set[tuple[int, int, float]] = set()
    branch_counts: set[int] = set()
    cache: list[dict] = []

    ep_files = sorted(glob.glob(os.path.join(run, "episodes", "*.npz")))
    t_start = time.time()
    for fi, epf in enumerate(ep_files):
        rof = rollout_path(run, epf)
        if not os.path.exists(rof):
            continue
        n_files += 1
        ep = dict(np.load(epf))
        with np.load(rof) as z:
            ro = {k: z[k] for k in z.files}
        gtv = ep["gt_valid"].astype(bool)
        ep_idx = int(os.path.basename(epf)[3:9])
        branch_counts.add(int(ro["pos"].shape[1]))

        for g in range(len(ro["origin_step"])):
            n_groups += 1
            origin = int(ro["origin_step"][g])
            a = int(ro["agent_slot"][g])
            branch_t = origin * LOG_DT
            if legacy_offset:
                t0 = origin
            else:
                t0 = aligned_t0(ep["t"], origin)
                check_step_alignment(ep["t"], t0, branch_t, HORIZONS.values())
            t0_seen.add((origin, t0, float(ep["t"][t0])))
            if t0 < HIST_STEPS - 1:
                skips["no_full_history"] += 1
                continue
            if not gtv[t0, a]:
                skips["gt_invalid_at_t0"] += 1
                continue
            win = window_detections(ep, t0, elig_rule)
            if win[:, a].sum() < min_obs:
                skips["too_few_detections"] += 1
                continue
            s = build_sample(ep, t0, a, data_cfg, with_labels=False)
            x = torch.as_tensor(s["x"][None]).to(device)
            m = torch.as_tensor(s["agent_mask"][None]).to(device)
            with torch.no_grad():
                o = model(x, m)
            n_scored += 1
            ox, oy, th = s["frame"]
            c, sn = math.cos(th), math.sin(th)
            R_inv = np.array([[c, -sn], [sn, c]])
            entry = {
                "rec": {"episode_index": ep_idx, "group": g, "agent": a,
                        "gt_type": int(ep["gt_type"][a]), "origin_step": origin,
                        "t0": int(t0), "t0_time_s": float(ep["t"][t0]),
                        "reacq": int(s.get("meta_reacq", -1)), "n_valid": {}},
                "mu": o["mu"][0].float().cpu() * POS_SCALE,            # [K,30,2]
                "sig100": o["log_sigma"][0].float().exp().cpu() * POS_SCALE,
                "rho": o["rho"][0].float().cpu(),
                "pi": torch.softmax(o["mode_logits"][0].float(), 0).cpu(),
                "pts": {}}
            for h_lab, s_idx in HORIZONS.items():
                bp = ro["pos"][g, :, s_idx, :]                 # [B,2] world
                bv = ro["valid"][g, :, s_idx].astype(bool)
                nv = int(bv.sum())
                entry["rec"]["n_valid"][h_lab] = nv
                if nv < MIN_VALID_FUTURES:
                    entry["pts"][h_lab] = None
                    continue
                bf = (bp[bv] - np.array([ox, oy])) @ R_inv     # -> frame [m]
                entry["pts"][h_lab] = torch.as_tensor(bf, dtype=torch.float32)
            cache.append(entry)
        if log_every and (fi + 1) % log_every == 0:
            print(f"  {fi + 1}/{len(ep_files)} episodes, {n_scored} groups "
                  f"scored, {time.time() - t_start:.0f}s", flush=True)

    info = {"n_groups": n_groups, "n_scored": n_scored, "skips": skips,
            "n_episode_files": n_files, "t0_seen": sorted(t0_seen),
            "branches_per_group": sorted(branch_counts),
            "elig_rule": elig_rule, "min_obs": min_obs,
            "forward_wall_s": round(time.time() - t_start, 1)}
    return cache, info


def mc_pass(cache: list[dict], T: float, normal_seed: int, comp_seed: int):
    """Conditional coverage + density-PIT for every cached group (released
    arithmetic; seeded generators)."""
    gen = torch.Generator().manual_seed(int(normal_seed))
    comp_gen = torch.Generator().manual_seed(int(comp_seed))
    cov_vals = {h: {n: [] for n in NOMINALS} for h in HORIZONS}
    pit_all = {h: [] for h in HORIZONS}
    groups = []
    for e in cache:
        mu, rho, pi = e["mu"], e["rho"], e["pi"]
        sig = e["sig100"] * T
        rec = dict(e["rec"], cov={}, pit_counts={})
        for h_lab, s_idx in HORIZONS.items():
            pts = e["pts"][h_lab]
            if pts is None:
                rec["cov"][h_lab] = None
                rec["pit_counts"][h_lab] = None
                continue
            mu_s, sig_s = mu[:, s_idx], sig[:, s_idx]
            rho_s = rho[:, s_idx]
            mc = sample_mixture(mu_s, sig_s, rho_s, pi, HPD_MC, gen, comp_gen)
            d_mc = mixture_logpdf(mc, mu_s, sig_s, rho_s, pi)
            d_br = mixture_logpdf(pts, mu_s, sig_s, rho_s, pi)
            cov_g = {}
            for nom in NOMINALS:
                thr = torch.quantile(d_mc, 1 - nom)
                v = float((d_br >= thr).float().mean())
                cov_vals[h_lab][nom].append(v)
                cov_g[f"nom{int(nom * 100)}"] = v
            rec["cov"][h_lab] = cov_g
            # density-PIT: rank of each branch among MC densities
            pit_l = (d_mc[None, :] < d_br[:, None]).float().mean(1).tolist()
            pit_all[h_lab].extend(pit_l)
            rec["pit_counts"][h_lab] = np.histogram(
                pit_l, bins=PIT_BINS, range=(0, 1))[0].astype(int).tolist()
        groups.append(rec)
    return {"cov_vals": cov_vals, "pit_all": pit_all}, groups


def released_mc_fields(pooled: dict) -> dict:
    """The MC-derived fields of the released JSON, computed as released."""
    cov_vals, pit_all = pooled["cov_vals"], pooled["pit_all"]

    def q(v, p):
        return round(float(np.percentile(v, p)), 3) if v else None

    return {"conditional_coverage": {
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


# ============================================================== bootstrap

def _weighted_median(values_sorted: np.ndarray, W_sorted: np.ndarray) -> np.ndarray:
    """Median of the multiset where value i appears W[b, i] times, per row b.
    Equals np.median of the expanded sample (mean of the two middle elements
    when the multiset size is even)."""
    cw = np.cumsum(W_sorted, axis=1)
    M = cw[:, -1]
    k1 = (M - 1) // 2
    k2 = M // 2
    p1 = (cw > k1[:, None]).argmax(axis=1)
    p2 = (cw > k2[:, None]).argmax(axis=1)
    return 0.5 * (values_sorted[p1] + values_sorted[p2])


def bootstrap_summary(groups: list[dict], W: np.ndarray) -> dict:
    """Percentile 95% intervals of mean/median conditional coverage and of
    the PIT histogram bins under replicate weights W [B, G] (group
    multiplicities)."""
    out = {"conditional_coverage": {}, "pit_hist": {}}
    for h in HORIZONS:
        per_nom = {}
        for nom in NOMINALS:
            key = f"nom{int(nom * 100)}"
            v = np.array([g["cov"][h][key] if g["cov"].get(h) else np.nan
                          for g in groups], dtype=np.float64)
            m = np.isfinite(v)
            if not m.any():
                per_nom[key] = None
                continue
            vm, Wm = v[m], W[:, m]
            means = (Wm @ vm) / Wm.sum(axis=1)
            order = np.argsort(vm, kind="stable")
            meds = _weighted_median(vm[order], Wm[:, order])
            per_nom[key] = {
                "n_groups": int(m.sum()),
                "mean": round(float(vm.mean()), 4),
                "mean_ci95": [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])],
                "median": round(float(np.median(vm)), 4),
                "median_ci95": [round(float(x), 4) for x in np.percentile(meds, [2.5, 97.5])],
            }
        out["conditional_coverage"][h] = per_nom
        C = np.array([g["pit_counts"][h] if g["pit_counts"].get(h) else [0] * PIT_BINS
                      for g in groups], dtype=np.float64)          # [G,10]
        n = C.sum(axis=1)
        if n.sum() == 0:
            out["pit_hist"][h] = None
            continue
        frac = (W @ C) / (W @ n)[:, None]
        lo, hi = np.percentile(frac, [2.5, 97.5], axis=0)
        out["pit_hist"][h] = {
            "hist": (C.sum(0) / n.sum()).round(4).tolist(),
            "ci95_lo": lo.round(4).tolist(), "ci95_hi": hi.round(4).tolist(),
            "n_values": int(n.sum())}
    return out


def replicate_weights(groups: list[dict], n_boot: int, seed: int):
    """(group weights, episode-cluster weights), both [B, G]."""
    G = len(groups)
    rng = np.random.default_rng(seed)
    Wg = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)
    eps = sorted({g["episode_index"] for g in groups})
    pos = {e: i for i, e in enumerate(eps)}
    E = len(eps)
    rng_e = np.random.default_rng(seed + 1)
    We = rng_e.multinomial(E, np.full(E, 1.0 / E), size=n_boot)
    col = np.array([pos[g["episode_index"]] for g in groups])
    return Wg, We[:, col], E


def n_valid_stats(groups: list[dict]) -> dict:
    out = {}
    for h in HORIZONS:
        nv = np.array([g["n_valid"][h] for g in groups], dtype=np.int64)
        if nv.size == 0:
            out[h] = None
            continue
        vals, cnt = np.unique(nv, return_counts=True)
        out[h] = {
            "n_groups": int(nv.size),
            "n_scored_at_horizon": int((nv >= MIN_VALID_FUTURES).sum()),
            "n_skipped_lt_min": int((nv < MIN_VALID_FUTURES).sum()),
            "min": int(nv.min()), "q05": float(np.percentile(nv, 5)),
            "q25": float(np.percentile(nv, 25)),
            "median": float(np.median(nv)),
            "q75": float(np.percentile(nv, 75)), "max": int(nv.max()),
            "mean": round(float(nv.mean()), 3),
            "counts": {str(int(v)): int(c) for v, c in zip(vals, cnt)},
        }
    return out


# =================================================================== main

def build_result(args, pooled, groups, info, data_cfg, t_meta):
    res = {"run": args.run, "model": args.model, "T": args.T,
           "n_groups": info["n_groups"], "n_scored": info["n_scored"],
           **released_mc_fields(pooled)}
    res["alignment"] = "legacy_origin_step" if args.legacy_offset else "timestamp"
    res.update(t_meta)
    res["reference_note"] = (
        "state-conditioned diagnostic: every branch shares the full simulator "
        "state at the fork (hidden manoeuvre mode, intent and reaction state "
        "frozen) and re-seeds only future decision/sensor/wind streams; the "
        "predictor conditions on observations only")
    res["eligibility"] = {"index_track_rule": info["elig_rule"],
                          "feature_track_rule": track_rule(data_cfg),
                          "min_obs": info["min_obs"],
                          "min_valid_futures_per_horizon": MIN_VALID_FUTURES,
                          "skips": info["skips"]}
    res["branches_per_group"] = info["branches_per_group"]
    res["n_episode_files"] = info["n_episode_files"]
    res["n_episodes_scored"] = len({g["episode_index"] for g in groups})
    res["n_valid_futures"] = n_valid_stats(groups)
    pit_all = pooled["pit_all"]
    res["pit_hist"] = {h: (np.histogram(pit_all[h], bins=PIT_BINS, range=(0, 1))[0]
                           / max(1, len(pit_all[h]))).round(4).tolist()
                       for h in HORIZONS}
    res["hpd_mc"] = {
        "draws_per_group_step": HPD_MC,
        "normal_draws": f"torch.Generator seeded {NORMAL_SEED} (as released)",
        "component_draws": f"torch.Generator seeded {args.comp_seed} (released: "
                           "unseeded torch default generator)",
        "threshold": "torch.quantile of MC log-densities at 1 - nominal"}
    if groups:
        Wg, We, E = replicate_weights(groups, args.boot, args.boot_seed)
        res["bootstrap"] = {
            "B": int(args.boot), "seed": int(args.boot_seed),
            "interval": "percentile 2.5 / 97.5",
            "group": {"unit": "rollout group (episode, origin step, target slot)",
                      "n_units": len(groups), **bootstrap_summary(groups, Wg)},
            "episode_cluster": {
                "unit": "episode (all groups of an episode share the same "
                        "branch realisations, so they are not independent)",
                "n_units": E, **bootstrap_summary(groups, We)},
        }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--out", required=True,
                    help="output JSON (never inside Paper/draft_work)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--legacy-offset", action="store_true",
                    help="reproduce the released misalignment: t0 = "
                         "origin_step (history ends one step after the fork)")
    ap.add_argument("--comp-seed", type=int, default=COMPONENT_SEED)
    ap.add_argument("--index-track-rule", choices=("current", "slot"),
                    default=None, help="override eligibility rule only")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--expect", default=None,
                    help="released JSON: non-MC fields must match exactly")
    ap.add_argument("--mc-replicates", type=int, default=0,
                    help="with --expect: replicate component streams for the "
                         "MC-consistency gate of the released MC fields")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = guard_output_path(args.out, args.overwrite)
    groups_out = (out[:-5] if out.endswith(".json") else out) + "_groups.jsonl"
    guard_output_path(groups_out, args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg = load_model(device, args.model)
    if args.index_track_rule:
        data_cfg["index_track_rule"] = args.index_track_rule

    cache, info = forward_pass(args.run, model, data_cfg, device,
                               args.legacy_offset)
    t_mc = time.time()
    pooled, groups = mc_pass(cache, args.T, NORMAL_SEED, args.comp_seed)
    mc_wall = time.time() - t_mc
    t0s = sorted(set(map(tuple, info["t0_seen"])))
    if len(t0s) != 1:
        print("WARNING: several (origin_step, t0) pairs:", t0s)
    origin, t0, t0_time = t0s[0] if t0s else (None, None, None)
    branch_t = origin * LOG_DT if origin is not None else None
    t_meta = {
        "branch_time_s": branch_t,
        "origin_step": origin,
        "t0_index": t0,
        "t0_time_s": t0_time,
        "model_step0_time_s": (t0_time + LOG_DT) if t0_time is not None else None,
        "rollout_step0_time_s": (branch_t + LOG_DT) if branch_t is not None else None,
        "model_minus_rollout_time_s": ((t0_time - branch_t)
                                       if t0_time is not None else None),
        "time_convention": "logged index i is t = (i+1)*LOG_DT; rollout step "
                           "k is branch_time + (k+1)*LOG_DT; model step k is "
                           "t[t0] + (k+1)*LOG_DT",
    }
    res = build_result(args, pooled, groups, info, data_cfg, t_meta)

    gate_fail = False
    if args.expect:
        with open(args.expect, encoding="utf-8") as fh:
            expected = json.load(fh)
        det_exp = {k: expected[k] for k in DETERMINISTIC_FIELDS if k in expected}
        det_mism = compare_fields(json_roundtrip(res), det_exp)
        mc_exp = {k: v for k, v in expected.items() if k not in det_exp}
        gate = {"expected_file": args.expect.replace("\\", "/"),
                "deterministic_fields": sorted(det_exp),
                "deterministic_exact": not det_mism,
                "deterministic_mismatches": det_mism,
                "mc_fields_exact_with_this_seed": not compare_fields(
                    json_roundtrip(res), mc_exp),
                "note": "released MC fields came from an unseeded (clock-"
                        "seeded) torch default generator: exact equality is "
                        "not attainable, consistency is tested against "
                        "replicate component streams"}
        if args.mc_replicates > 0:
            t_rep = time.time()
            reps = []
            for r in range(args.mc_replicates):
                p_r, _ = mc_pass(cache, args.T, NORMAL_SEED,
                                 args.comp_seed + 1 + r)
                reps.append(released_mc_fields(p_r))
            n_br = min(info["branches_per_group"]) if info["branches_per_group"] else 1

            def unit_of(p):
                return 1e-4 if p.startswith("/pit_hist") else 1e-3

            def floor_of(p):
                # per-group coverage is k / B: its quantiles move on that lattice
                if p.rsplit("/", 1)[-1] in ("median", "q25", "q75"):
                    return max(unit_of(p) / 2, 1.0 / n_br)
                return unit_of(p) / 2

            cons = mc_consistency(mc_exp, reps, unit_of=unit_of,
                                  floor_of=floor_of)
            cons["sd_floor_rule"] = ("half the rounding unit; quantile fields "
                                     f"(median/q25/q75) at the 1/B lattice step, "
                                     f"B = {n_br} branches")
            cons["replicate_comp_seeds"] = [args.comp_seed + 1,
                                            args.comp_seed + args.mc_replicates]
            cons["wall_s"] = round(time.time() - t_rep, 1)
            gate["mc_consistency"] = cons
            gate["pass"] = (not det_mism) and cons["pass"]
        else:
            gate["pass"] = None if not det_mism else False
        res["gate"] = gate
        gate_fail = gate["pass"] is False

    res["wall_s"] = {"forward": info["forward_wall_s"], "mc": round(mc_wall, 1)}
    res["source"] = {"script": os.path.abspath(__file__).replace("\\", "/"),
                     "script_sha256": file_sha256(os.path.abspath(__file__)),
                     "dataset_py_sha256": file_sha256(
                         sys.modules["data.dataset"].__file__),
                     "groups_sidecar": groups_out,
                     "torch": torch.__version__, "device": str(device)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(groups_out, "w", encoding="utf-8") as fh:
        for g in groups:
            fh.write(json.dumps(g) + "\n")

    print(json.dumps({k: res[k] for k in ("n_groups", "n_scored",
                                          "alignment", "t0_time_s")}))
    for h in HORIZONS:
        print(f"cov@{h}:", json.dumps({k: v["mean"] for k, v in
                                       res["conditional_coverage"][h].items()}))
    print("PIT(15s):", res["pit_hist_15s"])
    if args.expect:
        g = res["gate"]
        print("GATE deterministic:", "EXACT" if g["deterministic_exact"]
              else g["deterministic_mismatches"])
        if "mc_consistency" in g:
            c = g["mc_consistency"]
            print(f"GATE MC consistency: max|z| = {c['max_abs_z']} over "
                  f"{c['n_fields']} fields, {c['n_outside_replicate_range']} "
                  f"outside the {c['n_replicates']}-replicate range -> "
                  f"{'PASS' if c['pass'] else 'FAIL'}")
    print("saved ->", out)
    return 2 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
