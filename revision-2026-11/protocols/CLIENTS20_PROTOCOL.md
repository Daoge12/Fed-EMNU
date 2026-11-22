# Twenty-client extension v1

Frozen 2026-11-12 before runs. Responds to R1's larger-client recommendation.
CIFAR-10/standard ResNet18, twenty clients, IID and Dirichlet alpha0.5,
seeds42/43/44, global class0 request. Original dataset size, normalization,
50 FL rounds, five local epochs, batch128, Adam0.01 and 40% participation
retained. Eight clients per training round rather than four; each client's
partition is smaller. Rounding/minibatch counts and partitions change, so this
is a scale extension, not an isolated causal test of client count or equal-time
benchmark. No test-driven hyperparameter tuning.

Six stages per setting/seed: pretrain, freshly initialized retained retrain,
FT, GA, Fed-EMNU, UNSIR-style adaptation. Total36 full-budget stages. Each
method shares the twenty-client starting checkpoint/reference within seed.
All target holders participate in approximate events; record their actual
counts. Same event weighting, noise/update budgets, attack and evaluation as
canonical core. No larger-client PAR or new selection control claim.

Before formal runs: separate pilot seed42, both partitions, six stages each,
1000train/500test, one FL round/one local epoch (12stages). Gate formal launch
on twenty nonempty disjoint client partitions, eight-client round schedule,
retained class0 exclusion, exact parent/reference hashes, finite normalized
predictions/accuracy/reference distance and ten disjoint class attack splits.
Reduced-data pilot values never enter scientific tables.

Formal audit: canonical saved-artifact audit plus twenty-client partition,
schedule, actual target-holder participation and payload checks. Report mean/SD
n3 and paired 10-versus20 seed differences descriptively; no monotonic scalability
claim from two client counts. Compute is serial on one GPU; no actual transport
delay/framing measurement. Include time/local steps/payload and before utility.
Retain unfavorable runs; no selective removal or tuning. Frozen runner/evaluator
remain unchanged. No PDF generation in this batch.
