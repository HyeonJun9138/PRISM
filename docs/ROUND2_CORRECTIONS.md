# Corrected evaluation in v3.1

The v3.2 source package retains these endpoint definitions and frozen v3.1
input rows. Duration aggregation and numerical verification are updated as
specified in [NUMERIC_VERIFICATION.md](NUMERIC_VERIFICATION.md). The current
`results/round2/` summaries use integer-tick duration ties, not historical
v3.1 floating-point ties.

PRISM-revision-v3.1 is an additive successor to PRISM-revision-v3.0. The old
tag, original logs, models and dataset assets are unchanged. The second
revision does not require retraining. Model-input construction and planner
behavior are not modified by these evaluation corrections.

## Evaluation definitions

- The event screen uses the latest observation frame at or before entry.
  Only its open track ID is eligible; a missed detection on a still-open
  track is distinct from track closure. First detection belongs to that ID.
  Eligibility uses unrounded lead time, with no future detection lookup.
- Severe encounters are 152.4 m encounters with a raw minimum below 45 m,
  not a count of direct below-45 m runs. The union screen inherits the
  responsible target's 152.4 m pairwise entry clock.
- Actual left-tick intervals determine dwell. The terminal timestamp can
  count an event but adds no extra duration. Complete episodes have 90 s
  exposure, so 300 and 600 episodes correspond to 7.5 and 15 h per arm.
- Legacy rounded minima and last-replan exposure have explicit fallback
  provenance. Loading old rows does not retroactively repair their screen.

The closed-track correction affects no screened endpoint counts in the
verified sample. A separate unrounded-minimum correction changes the static
nominal-OOD severe count from 167 to 168. Corrected CV/PRISM counts remain
126/101; without screening they are 154/131. The latter confidence interval
includes unity. The first 300 pairs yield 50/49 and the additional 300 yield
76/52. Expansion was decided after earlier outcomes had been inspected;
pooled inference is exploratory, not a prospectively fixed 600-pair design.
The centre/width controls do not establish calibration as the causal source
of a planning benefit. Coverage describes generated and logged forecasts,
not a retrospectively recoverable risk-provider-accepted subset. Historical
runtime summaries are not an onboard timing validation.

## Frozen evidence

`results/round2/numbers_manifest_round2.json` and `aggregates/` contain the
corrected numerical ledger and all paired comparisons. `supplement/` contains
the companion S1-S11 tables, coverage flow and timing provenance. Historical
paths in these files identify original sources; portable reproduction resolves
run names without requiring those drives. Old `results/planner/` and old
figure sources remain explicitly historical, not corrected outputs.

The new asset uses `round2/rescored/`, separate from original `Logs/Planner/`.
Its 12,300 rows were either reconstructed and checked against the original
scientific record or certified by the recorded zero-event invariance proof.
They must not all be described as newly GPU-replayed trajectories. Each row
retains its recovery provenance. CPU replay runtime is not the historical
GPU performance measurement. Raw trajectory payloads are not included in this
asset; the public reproduction below re-aggregates the verified rows.

## Reproduce corrected arm and paired statistics

Run from the repository root after installing requirements:

```sh
python -m unittest discover -s tests -v
python modelVal/tests/test_corrected_endpoints.py
python tools/reproduce.py prepare
```

Download `PRISM-revision-v3.1-corrected-endpoints.zip` from the frozen v3.1 release into
`downloads/`, then run:

```sh
python tools/reproduce.py install --bundle corrected-endpoints
python tools/reproduce.py verify --bundle corrected-endpoints
python tools/verify_round2.py
```

The verifier validates all 12,300 rows and hashes, checks paired traffic and
index ranges, then reconstructs every arm summary and paired contrast using
10,000 bootstrap draws and seed 20260926. It compares complete numerical
structures, including unrounded intervals, without overwriting references.
Discrete fields and hashes remain exact; computed floating statistics use
the stated absolute and relative tolerances of 1e-12.
For one comparison use `--only core_ood40`. Reference figures in
`paper/figures/` are the exact final manuscript images; their hashes and
captions are in `manifest.json`. No claim is made that the historical
`tools/reproduce.py figures` command regenerates the corrected images.

For a new simulation use the v3.1 evaluator, an explicitly selected checkpoint
and a new output directory. The v3.0 inputs retain their original checksums;
see `release_assets/README.md` for downloads. Old checkpoint and dataset
assets are intentionally not renamed or uploaded twice.

## Test scope

The 22 endpoint regression tests fail on the old evaluator in 20 cases and
pass on the corrected evaluator. The planner's closed-track test now expects
an ineligible closed track. Tests requiring absent datasets report skips.
An existing fresh-seed test can fail in a populated research workspace when
its default held-out dataset already exists; original logs or seed ranges
must not be deleted or altered to disguise that condition.

AI assistance used during development and revision included Anthropic Claude
and OpenAI Codex, including retained code assistance. The authors remain
responsible for verification and interpretation.
