"""Aggregate a batch directory (<arm>.jsonl files) into the paper table.

  python -m experiments.aggregate --dir $PRISM_ROOT/Logs/Planner/<batch>
Writes planner_batch_summary.json into the batch dir AND (if it exists)
into $PRISM_ROOT/Paper/draft_work/, and prints a readable table.
Pairing: episodes are index-matched across arms (identical traffic), so
sign counts (better/tie/worse) are computed on per-index deltas.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

EP_HOURS = 90.0 / 3600.0


def load(path: str) -> dict[int, dict]:
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            rows[r["index"]] = r
    return rows


def summarize(rows: dict[int, dict]) -> dict:
    v = list(rows.values())
    if not v:
        return {}
    arr = lambda k: np.asarray([r[k] for r in v], dtype=float)  # noqa: E731
    n = len(v)
    avoid = arr("nmac_events_avoidable")
    return {
        "episodes": n,
        "sim_hours": round(n * EP_HOURS, 2),
        "nmac_total": int(arr("nmac_events").sum()),
        "nmac_avoidable_total": int(avoid.sum()),
        "nmac_avoidable_per_hour": round(float(avoid.sum())
                                         / (n * EP_HOURS), 2),
        "episodes_with_avoidable_nmac": round(float((avoid > 0).mean()), 3),
        "min_sep_median": round(float(np.median(arr("min_sep"))), 1),
        "min_sep_p5": round(float(np.percentile(arr("min_sep"), 5)), 1),
        "nmac_duration_mean_s": round(float(arr("nmac_duration_s").mean()), 2),
        "lowc_fraction_mean": round(float(arr("lowc_fraction").mean()), 3),
        "abs_omega_mean": round(float(arr("abs_omega_mean").mean()), 2),
        "abs_omega_p95_mean": round(float(arr("abs_omega_p95").mean()), 2),
        "v_std_mean": round(float(arr("v_std").mean()), 2),
        "plan_ms_mean": round(float(arr("plan_ms_mean").mean()), 1),
        "plan_ms_p95_mean": round(float(arr("plan_ms_p95").mean()), 1),
        "infer_ms_mean": (round(float(np.mean([r["infer_ms_mean"] for r in v
                                               if r.get("infer_ms_mean")])), 1)
                          if any(r.get("infer_ms_mean") for r in v) else None),
    }


def paired(a: dict[int, dict], b: dict[int, dict], key: str,
           better: str) -> dict:
    """Sign counts for arm-a vs arm-b on shared indices."""
    idx = sorted(set(a) & set(b))
    da = np.asarray([a[i][key] for i in idx], dtype=float)
    db = np.asarray([b[i][key] for i in idx], dtype=float)
    diff = da - db
    if better == "low":
        win, lose = int((diff < 0).sum()), int((diff > 0).sum())
    else:
        win, lose = int((diff > 0).sum()), int((diff < 0).sum())
    return {"n": len(idx), "win": win, "tie": len(idx) - win - lose,
            "lose": lose, "mean_delta": round(float(diff.mean()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    arms = {}
    for f in glob.glob(os.path.join(args.dir, "*.jsonl")):
        arm = os.path.splitext(os.path.basename(f))[0]
        arms[arm] = load(f)

    out = {"summary": {arm: summarize(rows) for arm, rows in arms.items()},
           "paired": {}}
    if "model" in arms:
        for base in ("none", "cv"):
            if base in arms:
                out["paired"][f"model_vs_{base}"] = {
                    "nmac_avoidable": paired(arms["model"], arms[base],
                                             "nmac_events_avoidable", "low"),
                    "min_sep": paired(arms["model"], arms[base],
                                      "min_sep", "high"),
                }

    dst = os.path.join(args.dir, "planner_batch_summary.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    dw = rf"{PRISM_ROOT}/results"
    if os.path.isdir(dw):
        json.dump(out, open(os.path.join(
            dw, "planner_batch_summary.json"), "w", encoding="utf-8"),
            indent=1, ensure_ascii=False)

    cols = ("episodes", "nmac_avoidable_total", "nmac_avoidable_per_hour",
            "episodes_with_avoidable_nmac", "min_sep_median", "min_sep_p5",
            "abs_omega_p95_mean", "plan_ms_mean")
    hdr = ["arm"] + list(cols)
    print("  ".join(f"{h:>26s}" if i == 0 else f"{h:>22s}"
                    for i, h in enumerate(hdr)))
    for arm in ("none", "cv", "model"):
        if arm not in out["summary"] or not out["summary"][arm]:
            continue
        s = out["summary"][arm]
        print("  ".join([f"{arm:>26s}"] +
                        [f"{s[c]!s:>22s}" for c in cols]))
    for k, v in out["paired"].items():
        print(k, json.dumps(v))
    print("saved ->", dst)


if __name__ == "__main__":
    main()
