"""Derived "framing" numbers for the manuscript text (percent reductions, gap closure),
each computed from ledger values with its formula and inputs recorded, so no arithmetic is
done by hand in the text.

    python framing_numbers.py      # -> framing_numbers.json (next to this file)

Inputs: r3_numbers.json (make_r3_tables.py), planner/core_<cond>.json (aggregate_r3),
recal_r3_k3_s1.json. Conventions: ratio a/b in the ledger is the paired rate ratio
"a|b" (> 1 = fewer events in b). "fewer with PRISM (%)" = 100 (1 - 1/ratio).
"""
from __future__ import annotations

import os

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
J = lambda p: json.load(open(p, encoding="utf-8"))
N = J(f"{HERE}/r3_numbers.json")
core = {c: J(f"{HERE}/planner/core_{c}.json") for c in ("sparse", "ood40", "ood100", "id40", "id24")}
out = {}


def put(key, value, formula, inputs, note=""):
    out[key] = {"value": round(float(value), 3), "formula": formula, "inputs": inputs, "note": note}


# 1. prediction: gap closure between the observation-only CV baseline and the GT-velocity information bound
for sp in ("ID", "OOD"):
    cvo, cvg, pr = N[f"r3.{sp}.cv-obs.minADE"], N[f"r3.{sp}.cv-gt.minADE"], N[f"r3.{sp}.prism_seeds.minADE.mean"]
    put(f"pred.{sp}.gap_closure_pct", 100 * (cvo - pr) / (cvo - cvg), "100 (cv_obs - prism) / (cv_obs - cv_gt)",
        {"cv_obs": cvo, "cv_gt": cvg, "prism_6seed_mean": pr},
        "share of the minADE gap between CV from detections and CV from the true state that PRISM closes")
    imm = N[f"r3.{sp}.imm-gapfix.minADE"]
    put(f"pred.{sp}.vs_imm_pct", 100 * (1 - pr / imm), "100 (1 - prism/imm)", {"prism": pr, "imm_gapfix": imm},
        "minADE reduction against the corrected, re-tuned IMM on the same samples")
k1 = N["r3.ID.r3_k1_s0.minADE"]; k3 = N["r3.ID.r3_k3_s0.minADE"]
put("pred.ID.vs_k1_pct", 100 * (1 - k3 / k1), "100 (1 - k3_seed0/k1_seed0)", {"k3_seed0": k3, "k1_seed0": k1},
    "value of the three-branch output over a single-branch model of the same size and recipe")
# per-class calibration span on ID (Table 9 revised)
cls = {c: N[f"r3.ID.class.{c}.cov90_15s"] for c in ("Multirotor", "FixedWingUAS", "eVTOL")}
put("pred.ID.class_cov90_min", min(cls.values()), "min over trained classes", cls, "per-class 90 % coverage at 15 s, lowest class")
put("pred.ID.class_cov90_max", max(cls.values()), "max over trained classes", cls, "per-class 90 % coverage at 15 s, highest class")

# 2. planner: severe encounters, time below 45 m, lower tail
for c, d in core.items():
    for pr_key, lab in (("cv|model", "vs_cv"), ("none|model", "vs_static")):
        e = d["paired"][pr_key]["events"]["union20_severe_new"]["rate_ratio"]
        put(f"planner.{c}.severe_fewer_{lab}_pct", 100 * (1 - 1 / e["est"]), "100 (1 - 1/ratio)",
            {"ratio_est": e["est"], "ratio_ci": e["ci"], "sign_p": d["paired"][pr_key]["events"]["union20_severe_new"]["sign_p"]},
            "percent fewer severe (< 45 m) encounters per hour with PRISM; CI and p from the ledger")
    A = d["per_arm"]
    tb_cv, tb_m = A["cv"]["time_below"]["union20_tb_45"]["per_hour_s"], A["model"]["time_below"]["union20_tb_45"]["per_hour_s"]
    put(f"planner.{c}.tb45_less_vs_cv_pct", 100 * (1 - tb_m / tb_cv), "100 (1 - prism/cv)", {"cv_s_per_h": tb_cv, "prism_s_per_h": tb_m},
        "percent less time per exposure hour below 45 m with PRISM than with CV")
    p5_cv, p5_m = A["cv"]["min_sep_20hz"]["p5"], A["model"]["min_sep_20hz"]["p5"]
    put(f"planner.{c}.p5_gain_vs_cv_pct", 100 * (p5_m / p5_cv - 1), "100 (prism/cv - 1)", {"cv_p5_m": p5_cv, "prism_p5_m": p5_m},
        "percent higher 5th percentile of episode minimum separation with PRISM (paired CI of the difference is in the ledger)")
# deadline compliance over all five conditions
miss = sum(core[c]["per_arm"]["model"]["timing"]["deadline_miss"] for c in core)
cyc = sum(core[c]["per_arm"]["model"]["timing"]["cycles"] for c in core)
put("planner.prism_deadline_kept_pct", 100 * (1 - miss / cyc), "100 (1 - misses/cycles)", {"misses": miss, "cycles": cyc},
    "share of PRISM replanning cycles that met the 0.5 s deadline on shared runner hardware")

# 3. recalibration gain on the held-out set
rc = J(f"{HERE}/recal_r3_k3_s1.json")
b, a = rc["before"]["cov90_15s"], rc["after"]["cov90_15s"]
put("recal.cov90_15s_gain", a - b, "after - before", {"before": b, "after": a, "T": rc["T"], "n_test": rc["n_test"]},
    "gain in 90 % coverage at 15 s on the held-out OOD episodes 300-1199")

json.dump(out, open(f"{HERE}/framing_numbers.json", "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    print(f"{k:45s} {v['value']:9.3f}")
