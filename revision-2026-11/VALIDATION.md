# Minor revision validation

- All 412 recorded commands resolve their checkpoint/reference dependency graph.
- Release file hashes, original source hashes and public config/metric coverage checked.
- All 90 published round2/walltime CSV rows match the public event metrics; all saved time-cap metrics comply.
- Five-seed primary Fed-EMNU retained-accuracy mean/SD recomputed from original metrics.
- Public summary generator exercised over all 412 metrics; technical repeats kept separate.
- Eight protocol/evaluation/event-weight tests passed.
- Tiny CPU pretrain, independent retained-data retrain and noise-update/evaluation pipeline passed with pilot=true. No pilot results enter scientific tables.
- Existing printed mean/SD numerical entries preserved by manuscript table reformatting.
- Full GPU reproduction was not rerun for this editorial revision. Raw posteriors/logs and weights remain excluded; checks do not independently recompute attacks. Actual times and time-cap stopping are hardware dependent.
