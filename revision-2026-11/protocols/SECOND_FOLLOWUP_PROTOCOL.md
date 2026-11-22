# Second follow-up protocol

Frozen before these runs. Preserve all outcomes without test-based tuning.

1. CIFAR-10/ResNet18 additional core seeds 45 and 46, IID and alpha=0.5,
   all ten original stages including matched selection/stage controls: 40 stages.
   Frozen runner/evaluator and original training budgets remain unchanged.
   Combine with seeds 42–44 only after verifying matching source/protocol/data.
2. CIFAR-100/ResNet18, seeds 42/43/44, IID and alpha=0.5, ten clients,
   global class 0 request. Six stages (Original/Retrain/FT/GA/Fed-EMNU/UNSIR):
   36 full-budget stages. No claim of reproduced PAR performance on this dataset.
   Same 50-round/five-local-epoch recipe, standard stem, batch 128, normalization
   and fixed update budgets; no dataset-specific test tuning.
3. Before full CIFAR-100 training, run a separate reduced-data pilot for both
   partitions; verify 100 logits, all 100 attack classes, disjoint calibration/test,
   one target plus 99 retained classes, matched before/reference and provenance.
   Pilot uses at least four test examples per class; exclude all pilot metrics.
   Full dataset has 100 test examples per class, giving 50 calibration and 50
   test examples per member/nonmember group for the declared class attack.
4. GPU work is serial. Additional core seeds first; CIFAR-100 pilot/full runs
   follow only after data checksum validation and pilot artifact verification.
5. Report mean, sample SD and two-sided 95% Student-t intervals for five-seed
   core principal metrics and paired differences; df=4, normal independent seed
   differences assumed. Intervals are descriptive, unadjusted for multiplicity;
   no universal significance or privacy claim. Keep three-seed earlier exports
   traceable and regenerate all affected tables/replies together after completion.

Poor initial utility is a result, not grounds to omit a run or tune on the test
set. No larger-client or repeated-request claim follows from this batch.
