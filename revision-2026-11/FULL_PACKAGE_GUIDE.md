# Reproducibility guide for the second revision

The old conflicting submission values are not recoverable. The original cause
is unknown; no claim is made that a particular code change explains them.
All revised empirical claims must be tied to the completed-run artifacts.

## Package layout

- `current/`: revised manuscript, current scripts, second-round evidence and protocol.
- `legacy/`: the canonical pre-second-round code, protocols and completed formal
  run artifacts used for the original revised tables. The old runner is preserved
  separately from the current runner. Workstation-specific historical paths in
  manifests are provenance records, not required installation locations.
- `MANIFEST.json`: hashes of every packaged member, plus omitted-file policy.

The package excludes CIFAR datasets, model checkpoints and pilot runs. Saved
test predictions, exact indices, configuration/source snapshots, logs and metrics
permit rechecking utility tables and the new comparisons without GPU training.
Full MIA regeneration requires regenerating the weights and running the supplied
evaluation; aggregate MIA records alone do not independently reconstruct attacks.
No public upload is implied by the package or the original repository URL.

## Environment

Recorded execution: Python 3.12.14, PyTorch 2.6.0+cu124, torchvision 0.21.0,
NVIDIA GeForce RTX 4070 Laptop GPU (8 GB). The second-round session is separate; the current driver is recorded, while
historical driver/thermal/load telemetry is unavailable. Inspect each `config.json` and the
second-round `environment.txt`; do not pool old and new elapsed times.

Create a Python environment, install `legacy/code/requirements-revision.txt`
and a compatible CUDA build of PyTorch/torchvision. Fetch official CIFAR-10
and CIFAR-100 via `legacy/code/download_cifar.py` or torchvision into
`legacy/code/data`. See `legacy/code/README.md` for dataset checksums.

## Regenerate shared primary checkpoints and controls

Run from `legacy/` with the new environment's Python executable:

```sh
python code/run_revision_suite.py --controls --seeds 42 43 44 --output experiments/core-suite
python code/run_revision_suite.py --controls --seeds 45 46 --output experiments/core-extra-seeds
```

Historical manifests refer to omitted weights. Move the packaged evidence-only
`experiments/core-suite` and `experiments/core-extra-seeds` folders to a separate
archive before these commands, so regeneration writes fresh manifests instead
of trying to reuse historical paths. Preserve the evidence archive for checks.
Defaults reproduce the published CIFAR-10/ResNet18 protocol: 50 rounds,
five local epochs, 10 clients, 40% participation, batch 128, class 0 globally.
Approximate methods use each seed's common checkpoint and fresh retained-data
reference. Additional-study recipes are documented in the legacy protocols
and exact historical manifest commands.

## Regenerate second-round controls

Run from `current/` after the legacy checkpoints are available:

```sh
python code/replay_round2.py --legacy-root ../legacy --check-only
python code/replay_round2.py --legacy-root ../legacy
python code/replay_round2.py --legacy-root ../legacy --analyze
```

As above, archive the packaged evidence-only `current/experiments/round2`
folder before GPU regeneration. The runner validates checkpoint metadata,
retains exact commands and hashes, resumes only completed outputs, and runs
serially. It does 36 scientific control events and six technical timing repeats.
Its workstation wrapper `run_round2.py` has a local default; the portable
`replay_round2.py` overrides that default without changing the frozen source.

Use `code/audit_packaged_evidence.py` for evidence-only verification after
unpacking; this check does not require CUDA or model checkpoints. It recomputes
FA/RA from saved test predictions, checks matched optimizer counts and sample
selections, and checks hashes. The training analysis additionally verifies
checkpoint/reference hashes when those weights are present.

## Prepared source/evidence revision with common time caps

The `Fed-EMNU-ready-source-evidence.zip` archive supersedes the old source
content for the further preparation pass. Existing compiled PDFs remain the
previous snapshot by author request. No public deposition has been completed.
The archive includes the matching LaTeX source, tables, figures and templates;
it does not include an updated compiled manuscript or response PDF.

The additional wall-clock study is frozen in `experiments/WALLTIME_PROTOCOL.md`.
To regenerate after rebuilding the legacy checkpoints, archive the packaged
evidence-only `current/experiments/walltime` directory, then from `current/` run:

```sh
python code/replay_walltime.py --legacy-root ../legacy
python code/analyze_walltime.py
```

To check the packaged evidence without training, run `analyze_walltime.py`
against the packaged completed runs. It uses suite-relative paths, recomputes
FA/RA from saved posteriors, checks stage accounting, source hashes and paired
selection/attack splits, and generates numerical summaries and LaTeX tables.
It does not regenerate the manually reviewed interpretation file or claim to
rerun MIA from the weights. All over-budget events, if any, are listed explicitly.
Compare regenerated `analysis.json` and `seed-level.csv` to the supplied files.

## Interpretation limits

Matched update counts are not equal wall-clock budgets. Gaussian noise saves
server optimization steps. The event-weight control changes only request-time
weights, not pretraining. Timing repeats are technical replicates. One loss-based
MIA, three new scientific seeds and limited simulated client/dataset coverage
do not establish deletion certification or general superiority. Read the
manuscript's applicability and failure-condition section before interpreting results.
