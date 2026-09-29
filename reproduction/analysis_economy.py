"""Manoeuvre economy of the closed-loop core conditions (R3 minor 4).

    python analysis_economy.py          # writes planner/economy_r3.json

Reads the same per-episode rows as planner/core_<cond>.json (its "inputs"
paths), so the episodes are exactly those of the headline ledger.

Statistics (per arm):
  omega_p95_ep  per-episode 95th percentile of |omega| (deg/s) over the 2 Hz
                ownship log, then summarised ACROSS episodes: mean (the
                statistic of the submitted Table 11, i.e. "mean of
                per-episode 95th percentiles", not a pooled percentile),
                SD, median, IQR.
  omega_mean_ep per-episode mean |omega| (deg/s), same summary.
  elong_pct     per-episode path elongation vs corridor progress (%).
  v_std         per-episode SD of ownship speed (m/s).
Paired (same episode index in both arms): mean difference a - b with a
paired episode-bootstrap 95 % CI (B = 10,000, seed 20260926, as the ledger)
and the exact two-sided sign test (W = a larger, ties excluded).

Event fraction (the "fraction of episodes with an event" column of the
submitted Table 11): share of episodes with >= 1 event, per-episode counts
from experiments.aggregate_r3.episode_endpoints (the ledger's own counting,
new screen) for union20_152_new, union20_76_new, union20_severe_new.
Paired: exact McNemar test on discordant episodes (a only vs b only) and a
paired bootstrap CI of the fraction difference a - b.
"""
from __future__ import annotations

import os

import json
import math
import os

import sys

import numpy as np

sys.path.insert(0, (os.environ["PRISM_ROOT"] + '/modelVal'))
from experiments.aggregate_r3 import episode_endpoints  # noqa: E402  (same event counting as the ledger)

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PL = f"{HERE}/planner"
CONDS = ("sparse", "ood40", "ood100", "id40", "id24")
ARMS = ("none", "cv", "model")
PAIRS = ("none|model", "cv|model", "none|cv")
METRICS = {"omega_p95_ep": "abs_omega_p95", "omega_mean_ep": "abs_omega_mean",
           "elong_pct": "path_elongation", "v_std": "v_std"}
EVENT_KEYS = ("union20_152_new", "union20_76_new", "union20_severe_new")
B, SEED = 10_000, 20260926


def rows(paths):
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                v = {k: r.get(src) for k, src in METRICS.items()}
                if v["elong_pct"] is not None:
                    v["elong_pct"] = 100.0 * v["elong_pct"]
                v["n_agents"] = r.get("n_agents")
                ee = episode_endpoints(r)
                for k in EVENT_KEYS:
                    v["ev_" + k] = None if ee.get(k) is None else int(ee[k] > 0)
                out[int(r["index"])] = v
    return out


def summary(x):
    x = np.asarray([v for v in x if v is not None], dtype=np.float64)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": round(float(x.mean()), 4), "sd": round(float(x.std(ddof=1)), 4),
            "median": round(float(q50), 4), "q25": round(float(q25), 4), "q75": round(float(q75), 4)}


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail / 2 ** n))


def paired(a, b, rng):
    idx = sorted(set(a) & set(b))
    out = {"n": len(idx)}
    for m in METRICS:
        pa = np.array([a[i][m] for i in idx if a[i][m] is not None and b[i][m] is not None], dtype=np.float64)
        pb = np.array([b[i][m] for i in idx if a[i][m] is not None and b[i][m] is not None], dtype=np.float64)
        d = pa - pb
        bs = d[rng.integers(0, d.size, size=(B, d.size))].mean(axis=1)
        w, l = int((d > 0).sum()), int((d < 0).sum())
        out[m] = {"n": int(d.size), "diff_mean": round(float(d.mean()), 4),
                  "ci": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
                  "rel_pct": round(float(100.0 * d.mean() / pb.mean()), 2) if pb.mean() else None,
                  "wins": w, "losses": l, "ties": int(d.size - w - l), "sign_p": round(sign_p(w, l), 6)}
    return out


def event_fraction(data, rng):
    per_arm = {a: {k: {"n": sum(1 for v in d.values() if v["ev_" + k] is not None),
                       "episodes_with": sum(v["ev_" + k] or 0 for v in d.values()),
                       "fraction": round(float(np.mean([v["ev_" + k] for v in d.values() if v["ev_" + k] is not None])), 4)}
                   for k in EVENT_KEYS} for a, d in data.items()}
    paired = {}
    for pr in PAIRS:
        a, b = (data[x] for x in pr.split("|"))
        idx = sorted(set(a) & set(b))
        paired[pr] = {}
        for k in EVENT_KEYS:
            ia = np.array([a[i]["ev_" + k] for i in idx], dtype=np.float64)
            ib = np.array([b[i]["ev_" + k] for i in idx], dtype=np.float64)
            d = ia - ib
            bs = d[rng.integers(0, d.size, size=(B, d.size))].mean(axis=1)
            only_a, only_b = int((d > 0).sum()), int((d < 0).sum())
            paired[pr][k] = {"n": int(d.size), "diff": round(float(d.mean()), 4),
                             "ci": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
                             "only_a": only_a, "only_b": only_b, "mcnemar_exact_p": round(sign_p(only_a, only_b), 6)}
    return {"per_arm": per_arm, "paired": paired}


def main():
    res = {"definitions": __doc__, "B": B, "seed": SEED, "conditions": {}}
    for c in CONDS:
        core = json.load(open(f"{PL}/core_{c}.json", encoding="utf-8"))
        data = {a: rows(core["inputs"][a]["paths"]) for a in ARMS}
        rng = np.random.default_rng(SEED)
        res["conditions"][c] = {
            "inputs": {a: core["inputs"][a]["paths"] for a in ARMS},
            "per_arm": {a: {m: summary([v[m] for v in data[a].values()]) for m in METRICS} for a in ARMS},
            "paired": {pr: paired(data[pr.split("|")[0]], data[pr.split("|")[1]], rng) for pr in PAIRS},
            "event_fraction": event_fraction(data, np.random.default_rng(SEED)),
            "n_agents": {a: summary([v["n_agents"] for v in data[a].values()]) | {
                "min": int(min(v["n_agents"] for v in data[a].values())),
                "max": int(max(v["n_agents"] for v in data[a].values()))} for a in ARMS}}
    json.dump(res, open(f"{PL}/economy_r3.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for c, d in res["conditions"].items():
        print(c, " ".join(f"{a}:{d['per_arm'][a]['omega_p95_ep']['mean']:.2f}±{d['per_arm'][a]['omega_p95_ep']['sd']:.2f}"
                          for a in ARMS),
              "| cv-model", d["paired"]["cv|model"]["omega_p95_ep"]["diff_mean"], d["paired"]["cv|model"]["omega_p95_ep"]["ci"],
              "| frac152", " ".join(f"{a}:{d['event_fraction']['per_arm'][a]['union20_152_new']['fraction']:.3f}" for a in ARMS))


if __name__ == "__main__":
    main()
