# Time-cap protocol, fixed before new outcomes

48 scientific events: CIFAR-10/ResNet18, global class 0, IID and alpha=0.5,
seeds 42–44, budgets 10/20/30/40 seconds, two original recipe adaptations.
Shared legacy checkpoints/retraining references, random retained-data ordering,
ratio=1, frozen hyperparameters. Mode order alternates across seed/budget.

Noise generation keeps the original Fed 80-step/three-repair-epoch recipe and
UNSIR-style 40-step/one-repair-epoch recipe. After noise, subtract a predefined
reserve max(0.5 seconds, 15% total cap). Divide remaining local-training time
equally among all target-holding clients. Divide each client's allocation in
ratio 1:repair_epochs between impair and repair. Model copies, selection,
aggregation and final serialization count toward total elapsed time.

Local updates stop at synchronized minibatch boundaries, reserving the maximum
previous minibatch time before starting another batch. The first batch and
noninterruptible overhead can overrun; record EVERY total, report violations,
never relabel an over-budget run as on time. If no time is available, a stage
can perform zero steps, disclosed per client. Finish early if the original
recipe completes; no padding, extra epochs, test-based early stopping or
choosing best test checkpoint. These are equally capped adaptations, not
claims that the unmodified recipes are anytime algorithms or globally optimal
under the cap. Small caps may damage stage balance. Seed-level comparisons
use identical requested maximum time, with actual times and achieved steps.

Post-hoc utility/MIA/reference evaluation is excluded, as in prior timing.
Save full predictions, attack indices, stage traces, configs, source hashes,
logs and weights. Independent seeds=3 per cell, NOT 48 independent seeds.
No changes to existing PDF files or export scripts during this work.
