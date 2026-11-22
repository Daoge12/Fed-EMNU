# PAR-code-based comparison protocol, version 1

Frozen before any PAR experiment on 2026-11-10. Public source:
https://github.com/lizouss/PAR commit
`c587e11e56c2be71b99ed716f6045e5d92346c19`.
The repository describes itself as accompanying the exact Pang title. The
paper full text is still unavailable; code inspection is not full-paper
verification. Report this baseline as a **PAR-code-based adaptation**.

## Matched conditions

- Use the six saved CIFAR-10/ResNet18 core pretrained checkpoints, their
  partitions and corresponding retained-data references; seeds 42/43/44,
  IID and Dirichlet alpha=0.5, global class 0 request, 10 clients.
- Retain 32x32 images, [-1,1] normalization and the canonical model architecture
  to share exact starting weights. Public PAR CIFAR preprocessing instead
  resizes real examples to 128x128 and applies its own normalization. This
  difference must be disclosed; published PAR numbers are not reproduced.
- Identical utility test set, class-conditioned MIA splits/calibration rules,
  prediction distance and timing boundary. Time includes generator learning,
  synthetic extraction, local stages, aggregation, serialization and the
  additional generator/provenance serialization; excludes post-hoc evaluation
  and real transport. Thus artifact overhead is included rather than hidden.

## Preserved public-code event choices

- Learnable generator: latent dimension 128; 128-channel 8x8 stem, two upsample
  stages with 128 then 64 channels, three-channel Conv/BatchNorm output.
- Adam lr .001; 5 groups of 8 generator updates; negative target CE plus .01
  times mean per-image squared L2; fixed latent batch for each eight calls;
  ReduceLROnPlateau factor .5, patience 2, stepped with final loss per group.
- Freeze request model parameters and evaluation-mode BN. Public code calls
  server evaluation immediately before generation, yielding the same mode.
- Extract eight synthetic batches in generator train mode, same new latent
  batch for those eight calls, target labels; no clipping.
- All clients participate, even those with no target records. One impair epoch
  on local retained plus synthetic inputs; aggregate equally; ten repair rounds
  with one local retained epoch per client and aggregation after every round.
  Adam lr .001, fresh optimizer per client call, as in public defaults.
- Equal aggregation of floating states follows PAR. Use max BN integer counters
  to avoid public code's float averaging and implicit integer truncation; this
  is a disclosed compatibility change.
- Use the final fixed repair round. Public code selects a best model using
  retained test accuracy; the revision does not use the evaluation test set to
  choose the checkpoint. Save stage records but make no post-hoc best selection.

## Interpretation and checks

Differences from Fed-EMNU include generator optimization, local learning rates,
all-client participation, event aggregation and multi-round restoration. Their
combined effect cannot be assigned to a single component. Preserve native
budgets and report compute, local steps and tensor payload explicitly. Do not
claim that this is an equal-compute ablation or exact original reproduction.

Before GPU runs, use a tiny CPU pipeline check with a separate pilot checkpoint
and reference, confirm finite outputs, original-model preservation during
generation, final evaluation/artifacts, and the eleven aggregation stages.
Pilot values never enter scientific tables. Formal GPU runs are serial after
the active ResNet34 suite completes to avoid timing contention. No old core
runner/evaluator files are modified. The wrapper has its own frozen hash.
