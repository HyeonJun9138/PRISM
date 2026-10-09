# Release identity and provenance

> v3.1: corrected evaluation and updated figure references are documented in
> [ROUND2_CORRECTIONS.md](ROUND2_CORRECTIONS.md). The v3.0 analysis and plotting
> commands below are preserved as historical reproduction, not as the corrected
> second-revision result. Model and dataset assets are unchanged.


**PRISM-revision-v3.0** names the frozen release. Its Git commit identifies the
published tree. Historical source fingerprints identify files available at
particular research runs; they are not Git commit numbers.

The training-source fingerprint is
`85a4033b7df0bb8153d3cc13516ae29a0f54b4c0460c8d92b7e7f66de0090863`.
It hashes sorted relative paths and file SHA-256 digests for 28 files. The
optional `historical-training-source` asset contains their exact bytes and
`snapshot_manifest.json`. That snapshot has historical runner paths and is
for audit, not the recommended runnable release. Machine and queue scripts
are excluded.

The released implementation includes subsequent evaluation fixes and portable
paths. It is not byte-identical to the historical training snapshot. Changing
the release name does not retrain a model or change a scientific result.

## Model roles

| Role | Job/checkpoint |
| --- | --- |
| Headline / planner | `20260926_141629_r3_k3_s1/final/model_final.pt` |
| Six-seed summary | `r3_k3_s1` through `r3_k3_s6` |
| Seed-zero ablations | `r3_k3_s0`, `r3_k1_s0`, `r3_noneigh_s0`, `r3_notype_s0` |
| Matched Transformer-MDN | `r3_mdn_s0`, `r3_mdn_s1`, `r3_mdn_s2` |
| Reduced-data comparison | `r3_1x_s0` |
| Point-MSE | `20260926_231702_r3_pointnet_s0/checkpoints/best.pt`, epoch 21 |
| Submitted-model association audit | `20260813_170429_full5x_s4/final/model_final.pt` |

Headline checkpoint SHA-256:
`1bbb693397af69141a563df5eb7b8300138b93c58f2322b4ad1a701dcca841ba`.
Point-MSE checkpoint SHA-256:
`b4128aa51cd3d6a7951445687ca4e23b21294641660da72f7558bf238c943164`.
The resumed Point-MSE run did not improve epoch 21. Its final checkpoint is
not presented as an evaluated model; its log explains the recorded selection.

## Preserved evidence

Reference JSON records retain historical paths and hashes. `prepare` copies
selected metadata and results into an empty workspace and relocates path
strings only; numerical values are unchanged. Recorded scene files remain
byte-identical for provenance checks. Original model and data bytes are
listed in the Release manifest.

Git contains every validation-epoch record, copied verbatim from each original
training log; redundant per-batch training telemetry is omitted. The filter,
original log hashes, retained row counts and published hashes are recorded in
`metadata/validation_record_selection.json`. This retains all points used by
the validation-curve figure. Dataset GUI previews are also omitted; episode
arrays, branching futures, configurations and index records remain available.

The invalid `r3_ood8` duplicate, pilot run, superseded aggregates, preliminary
plots and editorial correspondence are excluded. The sparse headline run is
`r3_sparse`, not `r3_ood8`. Five core conditions and the final matched controls
remain. Original public history and local research files are not rewritten.
