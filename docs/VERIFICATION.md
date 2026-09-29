# Release verification — 2026-09-29

This records checks performed on the curated publication tree. It is not a
claim that every training job and closed-loop experiment was rerun.

| Check | Observed result |
| --- | --- |
| Publication contract | 6 tests passed: release scope, final figure bytes, checkpoint identity, aggregate input availability, path relocation and safe archive installation |
| Simulator | 11 self-test files passed |
| Predictor | 7 pipeline checks and 8 revision-metric checks passed |
| CV / IMM baselines | 13 baseline checks and 36 current-track checks passed |
| Planner revision tests | 27 passed |
| Evaluation revision tests, with released datasets | 11 passed, none skipped |
| Model-selection CLI test | 1 passed |
| Headline seed-one checkpoint | End-to-end simulator/predictor smoke run passed on CUDA |
| Numerical figure generator | 12 output entries complete; no missing or pending inputs |
| Explanatory figures | Figures 3, 6, 7, 12 and 16 regenerated successfully |
| Prediction table values | 471 numeric values exactly matched the frozen reference |
| Derived narrative values | 113 numeric values exactly matched |
| Held-out branch-spread analysis | 64 numeric values exactly matched |
| Maneuver/economy analysis | 1,487 numeric values exactly matched |
| Historical training source | 28 file digests and combined fingerprint matched the original manifest |
| Git byte preservation | 30 scene/validation-record files matched the actual staged Git blobs byte-for-byte |
| Fresh Git-index export | Publication tests, workspace preparation and Figure 6 generation passed |
| Credential-pattern scan | No private-key, common access-token or AWS access-key pattern found in the selected text files |

The implementation was compared with the local research source after newline
normalization. Simulator, predictor, baselines and planner changes were path
portability changes, not algorithm changes. One evaluation test was updated
for the now-included held-out dataset: it checks disjointness from the older
corpus and also verifies that the usual guard rejects reusing the released
seeds. The guard and simulator implementation are unchanged.

Plotting changes are limited to portable paths, the final figure layouts and
the approved oracle terminology. Figure 12's validation recomputed safety
costs and hard-collision flags for all 275 recorded candidates. The actual
selected plan remains feasible; later executed flight coordinates are copied
from the recorded run, not extrapolated.

All numerical claims above refer to the released scripts and input records.
Full retraining, every closed-loop rerun, real-traffic validation and 3D
validation were not performed as part of this packaging check.
