# Manuscript v3.0 → code and evidence

`paper/figures/manifest.json` lists the approved captions and image checksums.
No manuscript is included. Reference renderings are exact; regeneration can
change fonts and antialiasing without changing the underlying data.

| Content | Source |
| --- | --- |
| Traffic and sensor model | `dataGen/`, `metadata/Logs/Dataset/` |
| Current-track inputs | `modelTraining/data/dataset.py` |
| Predictor and loss | `modelTraining/model/` |
| CV and corrected IMM | `baselines/`, `metadata/Logs/Models/baselines_r3/` |
| Forecast-to-risk and feasibility | `modelVal/runtime/predictor.py`, `modelVal/planner/` |
| Prediction comparison and ablations | `results/r3_numbers.json`, `reproduction/make_r3_tables.py` |
| Aligned conditional rollouts | `modelVal/experiments/rollout_score.py`, `results/rollout_*.json` |
| Disjoint recalibration | `modelVal/experiments/recalibrate.py`, `results/recal_r3_k3_s1*.json` |
| Five closed-loop settings | `results/planner/core_{sparse,ood40,ood100,id40,id24}.json` |
| Shared hard geometry | `results/planner/shared40_300.json` |
| Calibration and mean/spread controls | `results/planner/cal40_300.json`, `decomp40_300.json` |
| Planning sensitivity | `results/planner/sens300_*.json` |
| Maneuver economy | `results/planner/economy_r3.json`, `reproduction/analysis_economy.py` |
| Joined/reacquisition analysis | `results/pred_p5_p6_p8_final.json`, `reproduction/analysis_pred_p5_p6_p8.py` |
| Parameter inventory | `results/param_appendix.json` |

## Figure sources (`paper/figure_sources/`)

| Figure | Source |
| --- | --- |
| 1, 2, 4, 5 | Explanatory diagrams; exact final images supplied, not numerical experiments |
| 3 | `make_figure03.py` and `make_fig3_headon.py`, simulator seed 4183 |
| 6 | `make_figure06.py`, illustrative schematic |
| 7 | `make_fig_hdr.py`; revised seed-one model, `_v3_s1` suffix; settings in `fig_hdr_manifest_v3_s1.json` |
| 8–11 | `make_r3_figs.py`: `rollout`, `per_class`, `recal_transfer`, `factor_ood` |
| 12 | `make_figure12.py`, episode 319; selected plan versus later repeated-replanning flight |
| 13, 14 | `make_r3_figs.py`: `planner_forest`, `shared_geometry` |
| 15, 17 | `make_r3_figs.py`: `closed_loop_calibration` (main and extended layouts) |
| 16 | `make_fig_case_avoid_r3_sel.py`, same recorded encounter as Figure 12 |
| 18–21 | `make_r3_figs.py`: `reacquisition`, `training`, `sensitivity`, `param_sensitivity` |

Figure 7 images follow manuscript insertion order: a, c, e, b, d, f, legend.
Figure 16's final image can have a different panel arrangement from its
generator. Use the reference images for the exact approved layout.

The risk field is a planning score, not a calibrated collision probability
or certified safety region. Hard feasibility and soft-risk ranking remain
distinct. Figure captions preserve interval availability and display limits.
Real-traffic validation and joint horizontal/vertical prediction are future
work, not claims made by this 2D release.
