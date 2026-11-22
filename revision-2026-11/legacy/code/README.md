# Fed-EMNU revision experiments

`revision_runner.py` is canonical; `main.py` forwards to it. Older client/server
modules are historical and are not imported. The frozen protocol is
`../experiments/PROTOCOL.md`. It explicitly revises the old single-client event,
partition rounding and equal-client aggregation. Never mix new and old results.

## Environment

Tested: Python 3.12.14, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
RTX 4070 Laptop GPU (8 GB). See `requirements-revision.txt` for pinned packages.
Install the CUDA wheels from the official PyTorch cu124 index. The original
`env.yml` is preserved, but it does not describe the actual revision environment.

The project interpreter is `.venv/Scripts/python.exe`. CIFAR-10 is extracted
under `code/data`; its official archive passed MD5
`c58f30108f718f92721af3b95e74349a`. The download helper supports resumable
official-source download.

## Commands from the project root

```powershell
# Small pipeline check; permanently excluded from scientific summaries.
.\.venv\Scripts\python.exe code\run_revision_suite.py --pilot --seeds 42 --settings iid noniid --controls --output experiments/pilot-suite-new

# Original three-seed core comparison and controls; serial execution on one GPU.
.\.venv\Scripts\python.exe code\run_revision_suite.py --controls

# Protocol and evaluation checks.
.\.venv\Scripts\python.exe -m unittest discover -s code -p test_revision_protocol.py -v
```

The original core suite uses seeds 42, 43, 44 in IID and Non-IID settings.
The current primary tables combine it with seeds 45 and 46 from
`experiments/core-extra-seeds`, totaling 100 stages. To reproduce that additional
suite separately, use:

```powershell
.\.venv\Scripts\python.exe code\run_revision_suite.py --controls --seeds 45 46 --output experiments/core-extra-seeds
```

These commands describe reproduction; no training is required for text review. Each runs
pretraining, independently initialized retained-data retraining, FT, GA, Fed-EMNU,
the documented UNSIR-style adaptation, random/confidence selection at ratio 0.5,
and impair/repair controls. Exact commands and status are in `suite.json`.
Repeating the identical command reuses completed stages and rejects changed
suite specifications. Partially completed training rounds are not resumed.

For an individual run, use `--mode pretrain` first, then pass that run's
`final_model.pt` via `--checkpoint`. Run `--mode retrain` for the reference and
pass its final state via `--reference` to approximate methods. Keep partition,
model and pretraining flags identical. `--iid` selects IID; omitting it selects
Dirichlet alpha 0.5. Default scope is `global_class`; `requesting_client` needs
its own matching reference. There is no `--save_checkpoint` flag.
Use `revision_runner.py --help` for all options.

## Artifacts and interpretation

Each timestamped run saves config/software versions, source hashes and exact
runner/evaluation source copies, original/retained partitions, sample IDs,
client schedule, final model, predictions, metrics, logs and status. Training
also saves latest state and round progress. Selection saves chosen indices.
MIA saves disjoint calibration/test indices, calibrated orientation/threshold,
per-class and target/retained macro measures. Reference checks enforce scope and
partition agreement and record the reference checkpoint hash.
Test posteriors are archived in `test_predictions.npz`. Training posteriors used
for MIA are evaluated in memory and are not separately archived; independently
recomputing attack scores requires the retained local checkpoints and dataset.

`--limit_train`, `--limit_test`, or `--pilot` marks a pipeline check.
Always use `--pilot` for reduced-round timing with full data. Only completed,
audited, full-budget scientific runs belong in paper summaries.

Method time includes confidence, noise, prescribed updates, aggregation and
final serialization, bracketed by GPU synchronization. Post-hoc evaluation and
real-network latency are excluded. Event payload bytes are separate; this
single-process simulation does not measure actual network latency.

On this Windows host the binary FontTools wheel could not load its optional
extension. Installing the same version from source produced a pure-Python wheel
and restored vector PDF export:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-binary=fonttools fonttools==4.68.0
```

Audit completed runs with `code/summarize_revision.py --suite <suite.json>`.
Use it without `--publish` when auditing the present revision. Its `--publish`
option targets the historical three-seed core and can replace current five-seed
tables and create a PDF plot. It is unsuitable for the present text-only review.
The combined primary source is `experiments/combined-core-summary.json`;
intervals are in `experiments/five-seed-summary.json`. See
`../responses/MATERIALS_INDEX.md` for the current result sources.

The author-review ZIP includes configs, metrics, source snapshots, test
predictions, all recorded indices and logs. It excludes model checkpoints and
the dataset. Absolute paths in execution manifests describe this Windows host;
to reproduce elsewhere, run the suite from the new root rather than invoking
an old manifest command verbatim. Existing PDF/ZIP exports are older snapshots and do not include all current
text updates. `finalize_revision_text.py`, `integrate_*` and `refresh_*` are
stage-specific historical editorial helpers, not a current one-command build.
Re-running them can overwrite subsequent manual revisions. Preserve current
sources and review any generated diff before use. The safe current text check is:

```powershell
.\.venv\Scripts\python.exe code\check_text_consistency.py
```

It reads evidence and writes a QA report; it does not train, compile, generate
PDFs or refresh the ZIP. Full attack-score reproduction needs the dataset and
local model checkpoints; these are excluded from the ZIP.
