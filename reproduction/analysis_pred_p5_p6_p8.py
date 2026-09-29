"""Pre-registered prediction analyses P5, P6, P8 (DECISIONS.md D6.4) from the
per-sample evaluation files (Logs/Analysis/eval_r3/<tag>_<set>_persample.npz).

    python analysis_pred_p5_p6_p8.py --out pred_p5_p6_p8.json

P5  interaction component: full K=3 (seed 0) vs no-neighbour (seed 0), per-sample
    NLL difference (noneigh - full) on samples whose target's reaction layer is
    active within the horizon (avoid = 1) vs not (avoid = 0).
P6  training recipe: PRISM K=3 (seed 0) vs standard MDN (seed 0): branch usage,
    entropy, distinct branches, accuracy and calibration.
P8  track association: oracle system (s4, slot inputs) vs current-track seeds,
    same current-track sample set, overall and by reacquisition class.
All comparisons are joined on (episode file, t0, agent). CIs: episode-cluster
bootstrap, B = 2000, seed 20260926.
"""
from __future__ import annotations

import os

import argparse
import json
import os

import numpy as np

EV = (os.environ["PRISM_ROOT"] + '/Logs/Analysis/eval_r3')
RNG = np.random.default_rng(20260926)
B = 2000


def load(tag, split):
    p = f"{EV}/{tag}_{split}_persample.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    files = np.array([os.path.basename(os.path.dirname(os.path.dirname(f))) + "/" + os.path.basename(f) for f in z["files"]])
    d = {k: z[k] for k in z.files if z[k].ndim >= 1 and z[k].shape[0] == z["file_idx"].shape[0]}
    d["key_file"] = files[z["file_idx"]]
    return d


def join(a, b):
    ka = {(f, int(t), int(g)): i for i, (f, t, g) in enumerate(zip(a["key_file"], a["t0"], a["agent"]))}
    ia, ib = [], []
    for j, (f, t, g) in enumerate(zip(b["key_file"], b["t0"], b["agent"])):
        i = ka.get((f, int(t), int(g)))
        if i is not None:
            ia.append(i); ib.append(j)
    return np.asarray(ia), np.asarray(ib)


def cluster_boot(values, clusters, stat=np.mean):
    """Episode-cluster bootstrap of stat(values)."""
    uc, inv = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inv == c) for c in range(len(uc))]
    est = float(stat(values))
    bs = []
    for _ in range(B):
        pick = RNG.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        bs.append(stat(values[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": round(est, 5), "ci": [round(float(lo), 5), round(float(hi), 5)], "n": int(len(values))}


def p5(split):
    full, noe = load("r3_k3_s0", split), load("r3_noneigh_s0", split)
    if full is None or noe is None:
        return {"missing": [t for t, x in (("r3_k3_s0", full), ("r3_noneigh_s0", noe)) if x is None]}
    ia, ib = join(full, noe)
    diff = noe["NLL"][ib].astype(np.float64) - full["NLL"][ia].astype(np.float64)
    av = full["avoid"][ia]; cl = full["key_file"][ia]
    out = {"n_joined": int(len(ia))}
    for lab, m in (("avoid_active", av == 1), ("avoid_inactive", av == 0), ("all", av >= -1)):
        out[lab] = cluster_boot(diff[m], cl[m])
    # difference of differences, cluster bootstrap over episodes
    uc, inv = np.unique(cl, return_inverse=True)
    groups = [np.flatnonzero(inv == c) for c in range(len(uc))]
    def dd(idx):
        a, d_ = av[idx], diff[idx]
        return d_[a == 1].mean() - d_[a == 0].mean()
    est = dd(np.arange(len(diff))); bs = []
    for _ in range(B):
        pick = RNG.integers(0, len(groups), len(groups)); bs.append(dd(np.concatenate([groups[p] for p in pick])))
    out["active_minus_inactive"] = {"est": round(float(est), 5), "ci": [round(float(x), 5) for x in np.percentile(bs, [2.5, 97.5])]}
    return out


def recipe_stats(d):
    win = d["top1"] if "top1" in d else np.argmax(d["pi"], axis=1)
    share = np.bincount(win.astype(int), minlength=3) / len(win)
    cov = d["cov"]           # [N, levels(50,70,90), horizons(5,10,15)]
    return {"n": int(len(win)), "top_branch_share": [round(float(x), 4) for x in share],
            "max_top_share": round(float(share.max()), 4),
            "H_pi_mean": round(float(np.mean(d["H_pi"])), 4), "n_distinct_mean": round(float(np.mean(d["n_distinct"])), 4),
            "branch_sep_15s_mean": round(float(np.mean(d["branch_sep_15s"])), 2),
            "minADE": round(float(np.mean(d["minADE"])), 3), "top1ADE": round(float(np.mean(d["top1ADE"])), 3),
            "meanADE": round(float(np.mean(d["meanADE"])), 3), "NLL": round(float(np.mean(d["NLL"])), 4),
            "trajNLL": round(float(np.mean(d["trajNLL"])), 4), "ES": round(float(np.mean(d["ES"])), 2),
            "cov90_15s": round(float(np.mean(cov[:, 2, 2])), 4), "cov50_15s": round(float(np.mean(cov[:, 0, 2])), 4)}


def p6(split):
    k3, mdn = load("r3_k3_s0", split), load("r3_mdn_s0", split)
    if k3 is None or mdn is None:
        return {"missing": [t for t, x in (("r3_k3_s0", k3), ("r3_mdn_s0", mdn)) if x is None]}
    ia, ib = join(k3, mdn)
    sub = lambda d, i: {k: v[i] for k, v in d.items()}
    a, b = sub(k3, ia), sub(mdn, ib)
    out = {"prism_recipe": recipe_stats(a), "mdn": recipe_stats(b)}
    cl = a["key_file"]
    for m in ("NLL", "minADE", "top1ADE", "trajNLL", "ES"):
        out[f"diff_{m}_mdn_minus_prism"] = cluster_boot(b[m].astype(np.float64) - a[m].astype(np.float64), cl)
    return out


def p8(split, seeds):
    orc = load("s4", split + "_indexcurrent") if split == "testid" else load("s4", split + "_indexcurrent")
    if orc is None:
        return {"missing": ["s4 oracle per-sample for " + split]}
    res = {}
    for s in seeds:
        cur = load(f"r3_k3_s{s}", split)
        if cur is None:
            continue
        ia, ib = join(orc, cur)
        rq = cur["reacq"][ib]; cl = cur["key_file"][ib]
        r = {"n_joined": int(len(ia))}
        for m in ("minADE", "NLL"):
            d = cur[m][ib].astype(np.float64) - orc[m][ia].astype(np.float64)
            r[m] = {"all": cluster_boot(d, cl)}
            for c in (0, 1):
                mm = rq == c
                r[m][f"reacq{c}"] = cluster_boot(d[mm], cl[mm])
        res[f"seed{s}"] = r
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="0,1,2,3,4,5,6")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    res = {"definitions": __doc__, "P5": {s: p5(s) for s in ("testid", "testood")},
           "P6": {s: p6(s) for s in ("testid", "testood")}, "P8": {s: p8(s, seeds) for s in ("testid", "testood")}}
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k != "definitions" else "...") for k, v in res.items()}, indent=1)[:6000])
