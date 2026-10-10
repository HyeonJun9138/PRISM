# Numerical verification in v3.2

The v3.2 update changes aggregation and verification, not event screening,
episode membership, checkpoints, or simulation. Frozen inputs are the
12,300 corrected rows in [PRISM-revision-v3.1](https://github.com/HyeonJun9138/PRISM/releases/tag/PRISM-revision-v3.1).
Earlier releases and their hashes are preserved.

## Duration rule

Recorded durations are converted to integer 20 Hz ticks before summation:
one tick is 0.05 seconds. The decoder accepts a residual of at most 1e-8
ticks (5e-10 seconds) from the nearest integer. It rejects off-grid,
negative, nonfinite, and Boolean inputs. The residual is only a record
representation check, not a tie tolerance. Episode wins, losses, and ties
use the canonical tick totals. Reporting and rate calculations use seconds.
The legacy 2 Hz durations also lie on this grid.

For nominal-OOD episode 585, 5.20 + 6.25 + 4.85 seconds and
5.20 + 5.20 + 5.90 seconds both give 326 ticks. Sequential floating-point
addition had incorrectly separated these equal durations. The full audit
also checks other episodes and all 20 analyses, not only this example.

The functions are used here on the released 90-second episode records;
they are not a general arbitrary-magnitude integer arithmetic interface.

## Verification tolerances

The emergency-cycle-weighted penetration mean uses `math.fsum` before
rounding. In the safety-weight-24 CV sensitivity arm, the mean rounds to
11.598 m rather than the historical 11.597 m obtained with sequential
summation. This is a rounding correction to an aggregate diagnostic, not a
change to any recorded penetration or emergency event.

Finite computed floating statistics use `math.isclose` with
`abs_tol=1e-12` and `rel_tol=1e-12`:

`abs(a - b) <= max(1e-12, 1e-12 * max(abs(a), abs(b)))`.

This allows last-bit differences in recomputed probabilities and intervals,
but not changes at the reporting precision. Event counts, episode indices,
wins, losses, ties, bootstrap draw counts, SHA-256 hashes, Boolean values,
nulls, and the nested structure retain exact checks. Nonfinite floating
statistics are rejected; undefined statistics remain explicit nulls. These
tolerances do not modify records or determine scientific ties.

Use the unrounded `sign_p` field or the complete CSV for small p values.
The legacy six-decimal summary field can round a small positive p value to
zero; this does not indicate an exactly zero probability.

## Reproduction

Run from a fresh source directory. Verification needs NumPy, not PyTorch
or a GPU. Tested runtimes are Python 3.10.9 and 3.12.14 with NumPy 2.2.6.

```sh
python -m pip install numpy==2.2.6
python tools/reproduce.py prepare
```

Download the unchanged `PRISM-revision-v3.1-corrected-endpoints.zip` from
the frozen input release into `downloads/`, then run:

```sh
python tools/reproduce.py install --bundle corrected-endpoints
python tools/reproduce.py verify --bundle corrected-endpoints
python tools/verify_round2.py
python -m unittest discover -s tests -v
```

The verifier checks all input hashes, all 12,300 rows, paired traffic and
indices, and all 20 aggregate references. It reconstructs every arm and
paired contrast with 10,000 bootstrap draws and seed 20260926 without
overwriting reference files. Its first line states the runtime and numeric
policy. For a shorter single-analysis check, add `--only core_sparse`.

## Reference summaries

`results/round2/aggregates/` and `numbers_manifest_round2.json` contain the
updated summaries. The complete companion tables are in
`results/round2/supplement/`. `duration_tie_corrections.csv` records the 12
changed duration contrasts across eight analyses, including main-condition,
second-batch, and sensitivity comparisons. Main severe-encounter counts and
their reported intervals are unchanged. No new training or closed-loop
experiment is involved. The standalone supplementary document is supplied
with the journal submission; the repository's CSV files do not replace it.
