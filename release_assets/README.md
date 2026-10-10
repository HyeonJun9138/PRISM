# Versioned downloads

The v3.1 correction is a new asset; unchanged large inputs remain at v3.0.
The v3.2 aggregation update reuses this exact v3.1 corrected-endpoints asset.
No large model, dataset, or episode bundle needs to be duplicated.
Download only the bundles needed for the analysis. Sizes below are compressed decimal MB.

| Bundle | Files | MB | Download |
| --- | ---: | ---: | --- |
| `models` | 16 | 100.1 | [PRISM-revision-v3.0-models.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-models.zip) |
| `prediction-evaluation` | 2,163 | 719.0 | [PRISM-revision-v3.0-prediction-evaluation.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-prediction-evaluation.zip) |
| `closed-loop` | 1,241 | 477.6 | [PRISM-revision-v3.0-closed-loop.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-closed-loop.zip) |
| `training-data` | 8,308 | 612.7 | [PRISM-revision-v3.0-training-data.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-training-data.zip) |
| `evaluation-data` | 6,596 | 785.8 | [PRISM-revision-v3.0-evaluation-data.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-evaluation-data.zip) |
| `historical-training-source` | 29 | 0.1 | [PRISM-revision-v3.0-historical-training-source.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-historical-training-source.zip) |
| `rollout-diagnostics` | 9 | 1.1 | [PRISM-revision-v3.0-rollout-diagnostics.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.0/PRISM-revision-v3.0-rollout-diagnostics.zip) |

The models and records are synthetic research artifacts. GUI dataset previews, interim checkpoints and debug runs are omitted. Full file hashes are in `MANIFEST.json`. Install using `python tools/reproduce.py install`; do not extract over the source tree.

## v3.1 corrected rows

Download [PRISM-revision-v3.1-corrected-endpoints.zip](https://github.com/HyeonJun9138/PRISM/releases/download/PRISM-revision-v3.1/PRISM-revision-v3.1-corrected-endpoints.zip).
This contains 12,300 corrected episode rows, the event audit and frozen status
under `round2/rescored/`. It does not overwrite the historical closed-loop
bundle. File and archive hashes are in `MANIFEST.json`.

```sh
python tools/reproduce.py install --bundle corrected-endpoints
python tools/verify_round2.py
```
