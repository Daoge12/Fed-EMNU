# Second-round protocol — 2026-11-18

Frozen before inspecting new results. Source project is preserved unchanged at
`C:/Users/le/Documents/Codex/2026-11-09/ni/outputs/Fed-EMNU-revision`.
New outputs and code live in the current `Fed-EMNU-round2` directory.

## Scientific controls (36 events)

CIFAR-10, ResNet18, ten clients, global class-0 request, IID and Dirichlet
alpha=0.5, seeds 42–44, original saved checkpoints and independently trained
references. Retain all examples (ratio=1); use random ordering consistently
in every new scientific control. This avoids the mode-dependent automatic
selection/order difference. No test-based tuning, altered initialization,
new pretraining or changed pretraining aggregation.

| Variant | Noise updates | Repair epochs | Noise penalty | Event aggregation |
|---|---:|---:|---|---|
| fed_high | 80 | 3 | Unsquared full-tensor L2 | Retained sample count |
| unsir_high | 80 | 3 | Mean squared per-image L2 | Retained sample count |
| fed_low | 40 | 1 | Unsquared full-tensor L2 | Retained sample count |
| unsir_low | 40 | 1 | Mean squared per-image L2 | Retained sample count |
| gaussian | 0 | 3 | No optimization, same initial Gaussian tensor | Retained sample count |
| equal_event | 80 | 3 | Unsquared full-tensor L2 | Equal affected-client weight |

The high and low within-budget comparisons match server noise-step counts,
client optimizer-step counts, retained examples/order and local learning rates.
They do not impose equal wall-clock latency: norm computation costs can differ.
Measured wall time and realized optimizer steps must be reported. Comparing
high versus low changes two budgets together and cannot isolate either one.
The Gaussian control isolates optimization availability, but saves server steps;
it is not equal compute. The event-weight control changes only request aggregation,
not the saved checkpoint policy. Existing stage-removal controls remain relevant
to local impair/repair roles but do not prove a new local optimizer.

## Timing investigation (6 repeat events)

Original Non-IID full recipe, confidence ordering, same request checkpoints and
unlearning seeds 42 and 45, three exact repeats each. Seed 45 is the prior timing
outlier. Repeats are technical replicates, never six new independent seeds.
Preserve the old outlier. A new GPU driver/session cannot establish its historical
cause. Record current environment; compare predictions as well as time.

All runs retain MIA, reference distances, partitions, schedules, selection indices,
source snapshots, configuration hashes, predictions and full log files. Failed
or partial runs are excluded. No pilots enter scientific tables. Report mean/SD
for seed-level outcomes, paired differences, and median/IQR for timing. New
timings are a separate session, not pooled with the old primary measurements.

## Execution

Use the original project's `.venv/Scripts/python.exe` to run `code/run_round2.py`.
The script serializes GPU use, records exact commands, and resumes completed runs.
Dataset and original reference paths are read-only dependencies. Full reproduction
from public data additionally requires the retained original protocol and runner;
local checkpoint paths do not constitute a public artifact URL.
