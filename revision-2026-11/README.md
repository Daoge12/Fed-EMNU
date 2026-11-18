# November 2026 reproducibility revision

Source and numerical summaries for the empirical class-suppression revision.
The original implementation remains in `../Fed-EMNU-main/`.
The manuscript source includes the final four-section revision, concise main-body control results and Appendix D diagnostics. Compiled manuscripts and response Word files are in the separate local submission bundle.

## Rebuild checkpoints

Create the recorded Python/CUDA environment from `legacy/code/requirements-revision.txt`.
Run from `legacy/`:

```sh
python code/run_revision_suite.py --controls --seeds 42 43 44 --output experiments/core-suite
python code/run_revision_suite.py --controls --seeds 45 46 --output experiments/core-extra-seeds
```

From `current/`, run:

```sh
python code/replay_round2.py --legacy-root ../legacy
python code/replay_walltime.py --legacy-root ../legacy
python code/analyze_walltime.py
```

Data download and model regeneration require public CIFAR access and a compatible
GPU. The local default paths in historical scripts are overridden by the replay
entry points. Checkpoint/source/configuration hashes are recorded per event.
See `protocols/` for exact budgets, stopping rules and exclusions.

## Evidence scope

`results/` contains numerical summaries and seed-level metrics. It is a concise
source release, not the full prediction/log archive. The separately prepared
source/evidence supplement contains saved posteriors, attack/sample indices,
logs and per-run source snapshots; this repository upload alone does not make
that entire archive publicly available. `FULL_PACKAGE_GUIDE.md` describes that
archive layout. Run the evidence-only audits after obtaining the supplement.

36 component controls, six technical timing repeats and 48 common-time-cap
events supplement the 322 earlier formal stages. Three scientific seeds per
new cell are paired, not independent across methods. No result certifies
deletion. The Fed-EMNU objective is not consistently superior; retained utility
under the tested time caps trades off against target suppression and attack AUC.
Historical conflicting-result provenance and PAR full-paper verification remain
unresolved. These limitations are retained in manuscript source.
