"""Branch-rollout spread of the held-out rollout sets (revised Table A2 input).

    python analysis_branch_radius.py        # writes branch_radius_heldout.json next to this file

For every rollout group (one target, 60 branch futures re-simulated from the
same simulator state at 30.0 s) the RMS branch radius at horizon h is
sqrt(mean_b |p_b(h) - mean_b p_b(h)|^2) over the branches valid at h. Reported:
median (and quartiles) over groups at 5, 10, 15 s, per set, all groups and
groups with all 60 branches valid at 15 s. Positions are ground truth
(rollouts/*.npz, key 'pos' [groups, 60, 30, 2]); step k is at 30.0 + 0.5 (k+1) s,
so 5/10/15 s are k = 9, 19, 29. numpy + stdlib only; reads only.
"""
from __future__ import annotations

import os

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SETS = {"id_heldout": (os.environ["PRISM_ROOT"] + '/Logs/Dataset/20260926_160253_test_id_rollout_heldout'),
        "ood_heldout": (os.environ["PRISM_ROOT"] + '/Logs/Dataset/20260926_155613_test_ood_rollout_heldout')}
H = {"5s": 9, "10s": 19, "15s": 29}


def group_radii(pos, valid):
    """pos [B,30,2], valid [B,30] -> {h: radius or None}."""
    out = {}
    for h, k in H.items():
        m = valid[:, k]
        if m.sum() < 2:
            out[h] = None
            continue
        p = pos[m, k, :].astype(np.float64)
        out[h] = float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
    return out


def q(v):
    v = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    a, b, c = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": round(float(b), 2), "q25": round(float(a), 2), "q75": round(float(c), 2),
            "mean": round(float(v.mean()), 2)}


def main():
    res = {"definitions": __doc__, "sets": {}}
    for name, d in SETS.items():
        files = sorted(glob.glob(f"{d}/rollouts/*.npz"))
        rad_all = {h: [] for h in H}
        rad_full = {h: [] for h in H}
        n_groups = 0
        for f in files:
            z = np.load(f)
            pos, valid = z["pos"], z["valid"]
            for g in range(pos.shape[0]):
                n_groups += 1
                r = group_radii(pos[g], valid[g])
                full = bool(valid[g, :, H["15s"]].all())
                for h in H:
                    rad_all[h].append(r[h])
                    if full:
                        rad_full[h].append(r[h])
        res["sets"][name] = {"run": d, "episodes": len(files), "groups": n_groups,
                             "all_groups": {h: q(rad_all[h]) for h in H},
                             "groups_all60_valid_15s": {h: q(rad_full[h]) for h in H}}
        print(name, n_groups, "groups;", {h: res["sets"][name]["groups_all60_valid_15s"][h]["median"] for h in H})
    json.dump(res, open(f"{HERE}/branch_radius_heldout.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
