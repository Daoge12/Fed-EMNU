# Frozen revision protocol, version 2

Canonical executable: `code/revision_runner.py` (`code/main.py` forwards to it).
The older module set is preserved as implementation history; its import-time
argument parsing is not used by the revision runner.

Primary scientific question: global class 0 revocation across all clients.
Requester-only revocation is available as an explicit alternative and must use
its own matching retraining reference. Metrics from the two scopes are never
merged.

- CIFAR-10, standard torchvision ResNet18 stem, normalized to [-1,1], no
  augmentation, 10 clients, 40% participation, 50 FL rounds, 5 local epochs,
  batch size 128, Adam learning rate 0.01.
- IID: class-stratified equal counts. Non-IID: per-class Dirichlet alpha 0.5,
  largest-remainder integer allocation, disjoint exhaustive partitions.
- Sample-count FedAvg for floating states; maximum for integer BatchNorm
  counters. This explicitly changes the old equal-client aggregation.
- Global request: all clients with target records independently update the same
  original checkpoint, followed by retained-count-weighted aggregation. All
  target records are excluded from the global-class Retrain reference.
- Requester-only request: one client updates; Retrain excludes only that
  client's target examples and retains other clients' target examples.
- Every approximate method and its Retrain reference share original partitions,
  sample IDs and round schedules. Retrain is initialized independently and uses
  the full retained training budget; no pretrained parameters are reused.
- FT: five full-network epochs on all accessible retained examples, lr 0.001.
  GA: five ascent epochs on forgotten examples, lr 0.001, no repair.
- Fed-EMNU: 80 noise Adam steps at lr 0.1, lambda 0.1 unsquared full-tensor norm,
  20 repeats of one noise batch, one impair epoch at lr 0.02 per target, three
  repair epochs at lr 0.01. Retained ratio 1 for the primary comparison; ratio
  0.5 for the matched confidence-vs-random control.
- UNSIR-style adaptation: 40 noise steps with mean squared per-image norm,
  random class-stratified retained subset, one impair/one repair epoch. This
  changes the centralized access/aggregation protocol and is labeled accordingly.
- Targets for independent runs: seeds 42, 43, 44, with split/pretraining seeds
  equal to run seed, unlearning seed equal to run seed, reference seed 1000+seed.
  Actual completed run counts must be reported, with sample SD (ddof=1).

Timing includes confidence, noise, updates, aggregation and final serialization;
post-hoc evaluations and real-network latency are excluded. GPU synchronization
brackets timing. Checkpoint writes during full training are included. Byte counts
report payload only, not transport framing.
Event payload is recorded for approximate updates. Empty communication records
for full pretraining/retraining mean unmeasured, not zero network cost. The
`optimizer_steps` field counts local model-update steps; server noise steps are
specified separately by noise_epochs times noise_steps per target class.

MIA: class-conditioned log true-class posterior; historical members from the
original training request scope versus held-out test nonmembers. Equal group
counts within each class; disjoint calibration/test groups; seed 90210. Score
orientation and threshold maximize calibration TPR-FPR only. Report held-out
classwise and macro target/retained AUC and signed advantage and differences
from the same retained-data retraining reference. This is one declared attack,
not a certificate.

Numerical score: log(max(true-class posterior, 1e-30)); AUC uses average ranks
for ties. The floor is fixed for all models, not tuned using attack test labels.

Pilot runs are permanently labeled `pilot=true`, use reduced samples or rounds,
and are excluded from scientific aggregation. No old published number is used
to fill a missing new result.
