# Reproduction guide

> v3.1: corrected evaluation and updated figure references are documented in
> [ROUND2_CORRECTIONS.md](ROUND2_CORRECTIONS.md). The v3.0 analysis and plotting
> commands below are preserved as historical reproduction, not as the corrected
> second-revision result. Model and dataset assets are unchanged.


## Read without downloading

The reference prediction tables (`results/r3_numbers.json`), planner
aggregates (`results/planner/`), evaluation reports (`metadata/`) and final
figures (`paper/figures/`) are small enough to stay in Git.

## Prepare a workspace

```sh
python tools/reproduce.py prepare
```

This creates `outputs/reproduction/`. To choose another empty directory,
pass `--root <directory>` on every command. Preparation refuses a nonempty
destination. The internal `Paper/` layout preserves the analysis scripts'
expected paths; it does not copy the private manuscript folder.

Download selected ZIP assets from the tagged Release into `downloads/`:

| Bundle | Purpose |
| --- | --- |
| `models` | All evaluated checkpoints |
| `prediction-evaluation` | Per-sample evaluations and reaction labels |
| `closed-loop` | Paired episode rows and prediction logs |
| `evaluation-data` | Held-out, calibration, OOD, rollout and sensitivity datasets |
| `training-data` | Full/reduced training and validation datasets; optional for result inspection |
| `historical-training-source` | Optional audit of the historical source fingerprint |
| `rollout-diagnostics` | Optional individual rollout-group diagnostics behind the summaries |

```sh
python tools/reproduce.py install --bundle models --bundle prediction-evaluation --bundle closed-loop --bundle evaluation-data
python tools/reproduce.py tables
python tools/reproduce.py figures
```

Outputs: `outputs/reproduction/Paper/rebuttal/r3/` for tables and working
ledgers, `outputs/reproduction/Paper/figures_r3/` for plots. These do not
overwrite `results/` or the final figure references. The generated ledgers
are not formatted copies of the manuscript tables.

The historical plotting script can emit placeholders if inputs are missing.
A valid reproduction must have **no pending inputs or errors** for the
requested figures in the generated `r3fig_manifest.json`. An exit code alone
does not prove successful reproduction.

```sh
python tools/reproduce.py figure03
python tools/reproduce.py figure06
python tools/reproduce.py figure07
python tools/reproduce.py figure12
python tools/reproduce.py figure16
```

Figure 12 evaluates dense HPD grids and may take several minutes. Its caches
are local to the workspace. Never substitute an untrusted pickle cache.
The 13 s selected plan and subsequent executed flight remain separate.

The Figure 7 command explicitly selects the revised seed-one checkpoint and
`_v3_s1` output suffix rather than the plotting script's legacy defaults.

## Run a new experiment

Set `PRISM_ROOT` to the absolute workspace path. PowerShell:

```powershell
$env:PRISM_ROOT = (Resolve-Path outputs/reproduction).Path
```

POSIX shell:

```sh
export PRISM_ROOT="$(pwd)/outputs/reproduction"
```

Enter `outputs/reproduction/modelVal/`. For a new nominal-load OOD run,
specify the revised checkpoint explicitly; the legacy default is preserved
for compatibility and is not the revised headline model:

```sh
python -m experiments.batch --arm model --episodes 600 --density 40 --preset test_ood --track-rule current --model ../Logs/Models/20260926_141629_r3_k3_s1/final/model_final.pt --out ../Logs/Planner/new_ood40
python -m experiments.batch --arm cv --episodes 600 --density 40 --preset test_ood --track-rule current --out ../Logs/Planner/new_ood40
python -m experiments.batch --arm none --episodes 600 --density 40 --preset test_ood --track-rule current --out ../Logs/Planner/new_ood40
python -m experiments.aggregate_r3 --runs none=../Logs/Planner/new_ood40/none.jsonl cv=../Logs/Planner/new_ood40/cv.jsonl model=../Logs/Planner/new_ood40/model.jsonl --out ../Logs/Planner/new_ood40/aggregate.json
```

Use `--log-preds` for model and CV when reproducing conditional coverage.
The aggregate's `inputs`, `knobs`, episode counts, seed and bootstrap count
define the exact comparison. A density-only sparse run is not equivalent to
the paper's sparse condition: population bounds and designed-encounter
settings also matter. Use `aggregate_r3`, not the legacy aggregator.

Original training configurations and data rosters are under
`metadata/Logs/Models/<job>/config.json`. The headline six-seed summary uses
seeds 1–6; seed 0 pairs the ablations. Full retraining and every closed-loop
episode were not rerun merely to package the release.

For calibration and branching-rollout options, from the modelVal directory:

```sh
python -m experiments.rollout_score --help
python -m experiments.recalibrate --help
python -m experiments.gen_rollout_heldout --help
```

Retain the disjoint episode/seed guards. Submitted-model and legacy-timing
diagnostics are audit controls, not replacements for the revised evaluations.

## Verify installed bytes

```sh
python tools/reproduce.py verify --bundle models --bundle closed-loop
```

The installer checks both archive and member hashes, rejects unsafe paths
and refuses to replace changed local files. Use a new empty workspace for
another preparation rather than deleting an existing experiment.
