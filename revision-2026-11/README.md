# Fed-EMNU revision reproduction guide

This release provides an implementation-oriented federated adaptation and empirical characterization. It does not claim a superior optimization principle over UNSIR or PAR, or verified deletion.

## Public files and exclusions

`protocols/replay-catalog.json` maps all **412 formal recorded stages/events** to their exact command flags and dependencies. `results/records/<suite>/<event>/` contains the original `config.json`, numerical `metrics.json`, and the frozen runner/evaluation/adapter Python snapshots saved by that event. Historical absolute paths in configuration records are provenance only: the replay entry point remaps executable paths and generated checkpoints. `PUBLIC_MANIFEST.json` hashes the release files. `results/legacy/`, `results/round2/`, and `results/walltime/` contain published numerical summaries; the latter two also include seed-level CSV files. Pilots are excluded.

The source release omits datasets, checkpoints, posterior arrays and raw execution logs. Their historical values cannot be independently recovered from metrics alone; running the recipes regenerates new outputs. `FULL_PACKAGE_GUIDE.md` points to these public instructions and makes the raw-evidence exclusion explicit; no separate evidence archive is needed to run the documented code.

## Fresh checkout and environment

```sh
git clone --branch codex/revision-2026-11 https://github.com/Daoge12/Fed-EMNU.git
cd Fed-EMNU/revision-2026-11
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\activate` on Windows; `source .venv/bin/activate` on Linux). Use Python 3.12. The recorded GPU execution used PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and an RTX 4070 Laptop GPU. Install compatible wheels:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-replay.txt
```

For CPU unit/pipeline checks, use the same torch/torchvision versions from `https://download.pytorch.org/whl/cpu`. Actual scientific timing and cap stopping require the recorded GPU class or an explicitly disclosed new hardware setting. Historical full environment pins are preserved in `legacy/code/requirements-revision.txt`; the minimal replay file avoids requiring unrelated plotting packages. Record the environment used for a new reproduction.

Download official CIFAR data using torchvision (checks the official extracted-file checksums):

```sh
python -c "from torchvision.datasets import CIFAR10,CIFAR100; [CIFAR10('data',train=t,download=True) for t in (True,False)]; [CIFAR100('data',train=t,download=True) for t in (True,False)]"
```

## Checks that do not run scientific training

```sh
python check_release.py
python reproduce.py --suite all
python -m unittest discover -s current/code -p "test_*.py" -v
python smoke_pipeline.py --data-root data --output-root pilot-output --device cpu
```

The first command checks release hashes, all 412 records and source snapshots, dependency coverage, all 90 control/timing CSV rows, cap compliance in saved metrics, and five-seed primary retained-accuracy mean/SD. It does **not** inspect excluded posteriors or independently regenerate MIA. The second prints executable plans only. The last runs a tiny pretrain → independently initialized retained-data retrain → approximate update/evaluation pipeline, permanently `pilot=true`, and must never be pooled with scientific results.

## Regenerate reported experiment families

All commands below run from this directory. Without `--execute`, they are dry runs. Use a fresh output directory for an independent rerun; completed stages resume using `replay-state.json`. Dependencies are automatically included, so extension events receive newly generated checkpoints/references rather than workstation-specific historical paths.

| Manuscript experiment | Suite | Formal recorded stages/events |
|---|---|---:|
| Five-seed primary comparison, selection and stage controls; Appendix A | `core-suite core-extra-seeds` | 60 + 40 |
| ResNet34 extension | `extension-resnet34` | 36 |
| Dirichlet alpha 0.1 | `extension-alpha01` | 30 |
| CIFAR-100 extension | `extension-cifar100` | 36 |
| Twenty clients | `extension-clients20` | 36 |
| PAR code-based adaptation | `par-comparison` | 6 |
| Two requests | `request-sequence` | 54 |
| Retained-only continuation | `continuation` | 24 |
| Matched-update controls and technical repeats; Appendix D | `round2` | 42 (36 controls + 6 technical repeats) |
| Common maximum time caps; Appendix D | `walltime` | 48 |

```sh
python reproduce.py --suite core-suite core-extra-seeds --data-root data --output-root replay-output --execute
python reproduce.py --suite extension-resnet34 extension-alpha01 extension-cifar100 extension-clients20 --data-root data --output-root replay-output --execute
python reproduce.py --suite par-comparison request-sequence continuation --data-root data --output-root replay-output --execute
python reproduce.py --suite round2 walltime --data-root data --output-root replay-output --execute
python summarize_replay.py --output-root replay-output
```

Alternatively use `--suite all`. Full reproduction requires substantial GPU time; no full scientific rerun is implied by release checks. Outputs are `replay-output/<suite>/<event>/runs/<timestamp>/`, including metrics, config, checkpoints, predictions and evaluation records. `execution.log` records each subprocess; `source/` contains that event's frozen code. `summarize_replay.py` produces seed-level metrics, mean/sample SD and paired control differences from regenerated runs. It keeps pretrain/retrain stages, technical repeats and scientific methods distinct. The historical published metrics remain unchanged under `results/records/`.

## Computational interpretation

Matched update budgets fix the stated server noise and local update counts within objective pairs, not elapsed time. Common maximum wall-clock caps bound the declared compute/serialization interval; recipes may finish early. Actual execution seconds are measured separately, exclude post-hoc evaluation and real network transport, and are not padded. Numerical results may differ across libraries/devices; synchronized minibatch stopping makes time-cap trajectories hardware dependent. These procedures support rerunning documented recipes, not a promise of identical cross-hardware times, cap trajectories, deletion guarantees or an explanation of the unresolved historical conflicting values.
