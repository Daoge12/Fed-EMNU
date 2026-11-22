Authors Reply to Reviewer Comments

Title of the Paper: Fed-EMNU: Federated Error-Maximizing Noise for Approximate Class Suppression

Authors: Jiale Wang; Zixiao Liu; Jianhua Wang

Summary of Changes

We thank the editor and reviewer for the positive assessment and the four remaining suggestions. This revision clarifies the empirical and implementation-oriented contribution, completes the public reproduction materials, distinguishes computational comparison conditions, and improves the appendix tables and notation. The scientific protocols and reported numerical results are unchanged.

We made the implementation-oriented federated adaptation and empirical characterization explicit in both the abstract and conclusion, without claiming superiority over UNSIR or PAR.

We expanded the public release to include configurations, numerical metrics and frozen source snapshots for all 412 formal recorded stages/events, a dependency-aware replay entry point and documented environment/data/output instructions.

We defined matched update budgets, common maximum wall-clock caps and actual execution time separately, and clarified the table headers and timing boundaries.

We enlarged appendix table text, split the widest control and continuation tables, standardized partition terminology, and added a shared guide to abbreviations, units, statistics and missing entries.

The original Introduction, Methodology, Evaluation and Conclusion structure and the manuscript title are retained. The highlighted manuscript marks the present minor-revision text relative to the preceding clean revision. Formatting changes and table rearrangements should be assessed in the clean PDF. Each reviewer action below is followed by an excerpt from the current manuscript.

Reply to the Editor

We appreciate the opportunity to address the remaining points. We have kept the contribution bounded to a reproducible federated adaptation and empirical characterization of existing mechanisms. We respectfully submit the clean and highlighted manuscripts, LaTeX source, the point-by-point response below and an updated cover letter. The public revision branch supplies the documented source, configurations and numerical summaries. The revised text retains the unfavorable results and the stated limits on class suppression and reproducibility.

Reply to Reviewer Comments

Concern 1: Contribution statements in the abstract and conclusion

Comment: The authors should ensure that the abstract and conclusion consistently mention the empirical and implementation-oriented contribution, without claiming superiority over UNSIR or PAR.

Author response:

We agree. Both sections now identify the contribution as an implementation-oriented federated adaptation and empirical characterization, rather than a new optimization principle or general superiority over UNSIR or PAR. The specific observed utility differences remain paired with their computational costs, residual target behavior and membership limitations.

Author action: Revised the opening contribution statement in the abstract and the opening of Section 4. The introduction retains attribution to UNSIR and PAR, and the conclusion retains the negative objective-control results and utility–suppression trade-offs. No new experimental result or stronger deletion claim is introduced.

Revised manuscript excerpt — page 1, Abstract.

Concern 2: Public materials and reproducible execution

Comment: The authors should ensure that all configurations, scripts, and numerical summaries publicly described are available in the stated repository and that the documented instructions allow for the reproduction of the reported experiments.

Author response:

We agree. The revision branch now includes original per-event configurations, numerical metrics and frozen runner/evaluation/adapter snapshots for 412 formal recorded stages/events across all experiment families. The replay catalog preserves recorded scientific flags and resolves checkpoint/reference dependencies into newly generated output paths. The README documents environment installation, official dataset preparation, dry runs, full suite execution, output locations and summary regeneration.

Author action: Expanded Data and Code Availability and Section 3.7; added reproduce.py, check_release.py, smoke_pipeline.py, summarize_replay.py, requirements-replay.txt, protocols/replay-catalog.json, PUBLIC_MANIFEST.json and results/records/. Release integrity/coverage checks, agreement with all 90 second-round/time-cap CSV records, eight protocol tests and a small CPU pretrain–fresh-retrain–update/evaluation pipeline passed. The small run is permanently pilot=true and excluded from scientific summaries. All 412 dependency plans were checked; the full GPU experiment matrix was not rerun for this editorial revision. Raw predictions/logs, datasets and weights remain excluded, with regeneration commands provided. Cross-hardware actual times and time-cap trajectories are not promised to match exactly.

Revised manuscript excerpt — page 18, Data and Code Availability.

Concern 3: Update budgets, maximum time caps and actual times

Comment: Computational evaluation is important to maintain the distinction between matched update budgets, common maximum time caps, and actual execution times throughout the manuscript.

Author response:

We agree. Matched update budgets fix the declared server noise-step and local-update counts within objective pairs. Common maximum wall-clock caps bound the declared compute/serialization interval under the specified scheduler. Actual execution time is measured separately; a recipe can finish before its cap. Neither matched steps nor a common cap imply identical elapsed time.

Author action: Added Computational comparison terminology at the start of Section 3 and aligned the abstract, conclusion and Appendix D terminology. Time columns now read Exec. (s); time-cap tables label the upper bound as T_max and explain that execution seconds are not padded. The existing exclusion of post-hoc evaluation and real network transport remains explicit, as do hardware-dependent minibatch stopping and native-recipe budget differences.

Revised manuscript excerpt — page 8, Computational comparison terminology.

Concern 4: Appendix readability, terminology and notation

Comment: Editorial improvements are recommended to enhance the readability of the appendix tables and to maintain consistency in terminology and mathematical notation.

Author response:

We agree. The appendix now provides a common reading guide for FA, RA, AUC, TV, percentage points, execution seconds, sample standard deviation, seed counts and unavailable entries. Partition names use IID and Non-IID consistently. The widest continuation and objective-control tables have been split into related tables so their text can be enlarged without scaling to fit.

Author action: Revised Appendix A’s introductory notation guide and the Appendix C/D table layout and captions. Control behavior is separated from realized local update counts and measured time; continuation accuracy changes are separated from membership/reference-distance diagnostics. Original numerical entries are preserved, and no missing metric is presented as zero. Tables were recompiled and visually inspected.

Revised manuscript excerpt — page 19, Reading the appendix tables.

Closing Remarks

We thank the editor and reviewer for the constructive final suggestions. The changes align the contribution statements, make the public reproduction path explicit, clarify computational comparisons and improve appendix readability. The measured results, limitations and historical-provenance uncertainty are retained.
