# Response to the further review comments

We thank the reviewers for identifying the remaining gaps between the stated contribution and the experimental evidence. We have narrowed the title and contribution statements, added controlled comparisons and robust timing analyses, and made the failure conditions and reproducibility boundary explicit. The response below follows the order of the two supplied review images. Comment headings below summarize the supplied review; reviewer groups follow the image order.

## Reviewer 1

### Comment 1: Scientific benefit of integration and controlled component experiments

**Response:** We agree that integrating known mechanisms does not itself establish scientific novelty. We now attribute the core error-maximizing-noise and impair--repair mechanisms to UNSIR and the overlapping federated perturb/restore design to PAR. The introduction describes our contribution as a specified adaptation and empirical investigation. The title now uses “Approximate Class Suppression.”

**Action:** We added objective comparisons at two common update budgets, a Gaussian-noise control and a request-event weighting control. The objective pairs use identical checkpoints, retained-data indices/order, local update steps, learning rates and attack splits. The event-weight control changes request aggregation without changing pretraining. The existing five-seed impair-only/repair-only controls remain as evidence about stage roles. We distinguish those controls from equal-compute experiments, and we do not claim that the controls establish a new optimization principle. Numerical results and their limits are given in “Controlled Noise Objectives and Event Aggregation.”

**Result:** The new controls do not establish a consistent benefit of the Fed-EMNU penalty. At the high budget, Fed-minus-UNSIR retained-accuracy differences are 0.063 ± 0.371 percentage points (IID) and -0.463 ± 0.361 (Non-IID); at the low budget they are -0.004 ± 0.306 and -0.356 ± 0.222. Optimized noise reduces mean target accuracy relative to Gaussian inputs under IID but does not show a consistent benefit under Non-IID. Retained-count event weighting increases Non-IID retained accuracy by 0.896 ± 0.224 points and target accuracy by 0.600 ± 0.557 relative to equal weighting. These are three-seed descriptive paired differences. We have incorporated these negative and mixed results into the abstract, component discussion and conclusion.

### Comment 2: Comparison with UNSIR and practical circumstances for use

**Response:** We agree that the original recipe comparison confounds the noise penalty with optimization and repair budgets. We added both objectives at 80 noise steps/three repair epochs and at 40 noise steps/one repair epoch, with all retained examples and the same ordering. Realized local optimizer-step counts are checked for equality within each budget; elapsed time is also reported. These are matched update budgets, not enforced equal wall-clock latency.

**Action:** We revised “Unlearning Effectiveness” to quantify the primary retained-accuracy advantage (0.44 percentage points IID, 1.43 Non-IID) alongside higher compute cost and residual target accuracy. The new controlled results are interpreted separately from the original five-seed recipes. We explain that a potential user would need to value the retained-utility gain and tolerate the measured residual behavior; the results establish no uniformly preferable method or certified-deletion use case.

**Result:** The retained-accuracy advantage of the original full recipe cannot be attributed to its norm penalty alone. The matched-budget penalty differences are small or favor the squared-penalty adaptation; we do not present the original recipe ranking as evidence of a superior objective.

The optimized-versus-Gaussian control also provides a favorable attack-specific result: target AUC decreases by 0.029 ± 0.012 under IID and 0.022 ± 0.018 under Non-IID, with decreases in all three pairs in each partition. We report this alongside the mixed accuracy outcomes and do not interpret it as a privacy guarantee or evidence of superiority over the UNSIR-style objective.

**Further common-time-cap comparison:** We additionally completed 48 events at maximum compute/serialization budgets of 10, 20, 30 and 40 seconds (three paired seeds per cell, two partitions, two recipes). All completed within their caps. We use a predefined client-balanced stage scheduler and report actual consumed time and realized steps; methods finishing early are not padded. Fed-EMNU's mean RA advantage is 1.41/0.49/0.48/0.39 percentage points under IID and 4.50/1.81/1.29/1.56 under Non-IID across those caps. Its target accuracy and target attack AUC are also higher at every cap. This supports a utility–suppression trade-off, not dominance or a privacy advantage. The added “Common Wall-Clock Caps” section describes truncation, synchronization overhead, allocation and limited cross-hardware reproducibility. These are time-capped adaptations, not claims that unmodified recipes are optimal anytime algorithms.

### Comment 3: Residual membership information and the meaning of deletion

**Response:** We agree. Class suppression is not equivalent to verified removal of historical training influence. In the primary experiments, the target-class attack AUC is 0.620 (IID) and 0.596 (Non-IID), compared with 0.508 and 0.487 for the retained-data retraining references. These residual differences are retained in the paper.

**Action:** The abstract, title, contribution statements, practical interpretation and conclusion explicitly restrict the outcome to approximate class suppression. “Class-Conditioned Membership Inference” explains the attacker, calibration/test split, score orientation, ties and evaluation limits. “Applicability and Failure Conditions” states that the present evidence does not qualify the method for requirements to verifiably remove individual training influence. Neither low FA nor a single chance-level attack result is treated as a deletion certificate.

### Comment 4: Generalization limitations across datasets and client settings

**Response:** We agree that these unfavorable results should determine the applicability claims. They remain in the main discussion and appendices.

**Action:** We added “Applicability and Failure Conditions.” It highlights CIFAR-100 residual target accuracy (33.00% IID; 24.67% Non-IID), membership differences, the twenty-client retained-accuracy decline (3.88 and 5.46 percentage points), and sparse-owner aggregation under stronger heterogeneity. ResNet34 evidence is limited to the ResNet family. The paper no longer interprets additional experiment count as broad robustness or scalability evidence.

### Comment 5: Confidence-based selection does not outperform random selection

**Response:** We agree. The five-seed matched half-subset controls show confidence-minus-random retained-accuracy differences of -1.50 percentage points (IID) and -0.84 (Non-IID). The full-retained-data main comparison cannot establish a ranking benefit.

**Action:** We retain the negative findings, describe confidence selection as an optional mechanism rather than a demonstrated improvement, and state this boundary in the abstract and contributions. The new objective and weighting controls use the same random ordering to remove the previous mode-dependent ordering difference. Alternative selection criteria remain future work; no untested criterion is presented as effective.

### Comment 6: Timing outlier, robust summaries and communication efficiency

**Response:** We agree. We preserve the original outlier, add median/quartile/range summaries for the original five seeds, and separately repeat the original Non-IID recipe three times on each of the seed-42 and seed-45 checkpoints. These are technical replicates, not additional scientific seeds. The repeated timings and comparison with original predictions are reported in “Robust Timing Summaries and Outlier Investigation.”

**Action:** We record the current driver and separate execution session, and avoid pooling new repeats with old timings. Saved logs lack the historical telemetry needed to identify a specific cause of the original delay; we therefore do not claim that a new timing result explains it. The timing boundary still includes computation and serialization while excluding post-hoc evaluation and real transport. Payload accounting is distinct from end-to-end federated latency, energy and communication efficiency.

**Historical investigation:** The seed-45 slowdown also occurs in retraining (2310.15 s, versus 420.10–429.82 s for the other Non-IID seeds) and the UNSIR-style event (63.40 s). It is therefore a cross-method anomaly in that recorded session, not an observation unique to Fed-EMNU. This pattern does not identify its historical cause.

**Repeat results:** Seed 42 took a median 23.03 s (range 22.99–24.20) across three exact-recipe repeats, versus 22.44 s originally. Seed 45 took a median 25.76 s (25.48–26.13), versus 114.20 s originally. In all six repeats, the saved posterior arrays were exactly equal to the corresponding original prediction arrays. The large historical delay was not reproduced in this session, but its cause remains unresolved. The original observation remains in the primary mean/SD and robust timing summaries.

### Comment 7: Resource-constrained edge deployment and MEC-related future work

**Response:** We added a bounded future-work discussion of resource-constrained edge deployment, including energy, memory, uplink/downlink latency, request completion time and placement of unlearning computation.

**Action:** We verified the metadata and abstract of Salman et al. (2025), “Hybrid CDN Architecture Integrating Edge Caching, MEC Offloading, and Q-Learning-Based Adaptive Routing,” Computers 14(10), 433, DOI: 10.3390/computers14100433, and cite it as an application context. The paper makes clear that this reference does not evaluate Fed-EMNU or establish its efficiency. Offloading would require explicit data-access and trust assumptions.

### Closing request: Make the revised method and supporting materials available

**Response:** We prepared an accompanying reproducibility package containing the current and prior canonical runners, fixed protocols, exact configurations/commands, source snapshots, saved test predictions, indices, logs, numerical summaries and analysis scripts. A file manifest and an evidence-only audit script permit verification without running GPU training.

**Action:** “Data and Code Availability” now links the public revision branch and identifies source snapshot 8fa5b727b820eac8c69a8c85e83840ba1951904d. That concise release contains source, protocols and numerical summaries; the full prediction/log evidence supplement is supplied separately. CIFAR data and checkpoint weights are excluded; regeneration instructions are supplied. The last sentence of the supplied review image is cut off; no missing text has been inferred.

## Reviewer 2

### Comment 1: Technical contribution relative to UNSIR and practical trade-off

**Response:** We clarified the adaptation's concrete protocol choices and attributed the underlying mechanism to prior work. We added objective comparisons at two matched update budgets, with identical retained-data ordering, and separate noise-optimization and event-weight controls. The results are presented as a limited empirical assessment rather than evidence that integration alone establishes novelty.

**Action:** See the revised introduction, closest-method comparison, “Controlled Noise Objectives and Event Aggregation,” and the practical interpretation in “Unlearning Effectiveness.” The original recipes favor Fed-EMNU on mean retained accuracy, while UNSIR-style is faster and suppresses target accuracy more strongly. The new objective-controlled results are reported separately; no universal superiority is asserted.

**Result:** Common-budget controls do not support a consistent retained-utility advantage of the Fed-EMNU penalty. In all three Non-IID seed pairs, the squared-penalty adaptation has higher retained accuracy at both tested budgets. The revised contribution is consequently a specified adaptation and empirical characterization, including unfavorable evidence, rather than a newly established superior mechanism.

### Comment 2: Practical meaning of membership-inference differences

**Response:** We strengthened the interpretation beyond acknowledging a generic limitation. Residual target-class membership distinguishability and prediction differences mean the evidence cannot establish that the requested data's training influence has been fully removed.

**Action:** The abstract, practical interpretation, “Applicability and Failure Conditions” and conclusion explicitly state the restriction to approximate class suppression. The evaluated black-box loss attack does not exhaust attackers, and class suppression cannot support a verified-deletion claim. The retraining comparator does not itself certify privacy of the full deployment.

### Comment 3: Reproducibility and the unexplained original inconsistency

**Response:** The historical cause of the conflicting original values remains unresolved. We have not invented an explanation. The revised evidence is a documented replacement evaluation, with its changed implementation and environment disclosed, rather than a reconstruction of the original results.

**Action:** We added “Reproducibility and Revision Provenance” and a table distinguishing known changes from unknown historical provenance. It covers explicit request sets, fresh retained-data retraining, aggregation, selection, optimizer budgets, saved predictions, attack splits, environment and synchronized timing. The accompanying package preserves old canonical source separately from the new runner, records source/configuration/checkpoint hashes and exact commands, and provides portable replay instructions and an evidence-only checker. Pilots/failed runs and technical timing repeats are not counted as independent scientific evidence. Original conflicting numbers are excluded from current quantitative conclusions.
