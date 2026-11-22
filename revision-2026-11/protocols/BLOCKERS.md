# Evidence status and remaining gaps

Updated 2026-11-12. Author-review evidence; not a submission-ready claim.

Completed and audited: primary core 100 stages (five seeds); ResNet34 36 stages; stronger
heterogeneity alpha=0.1 30 stages; six PAR-code-based events; CIFAR-100 36 stages (three seeds). Numerical
tables use saved predictions, reference hashes, attack accounting and frozen
source snapshots. PAR has an additional method/protocol/payload audit.
Seven canonical protocol tests and the CPU PAR preservation check passed.

The public PAR code was found at https://github.com/lizouss/PAR, commit
c587e11e56c2be71b99ed716f6045e5d92346c19. Full paper text remains unavailable.
The comparison is a code-based adaptation: matched 32x32 preprocessing,
max integer BN counters, fixed final checkpoint instead of test-based best
selection. It is not an exact reproduction of the published paper.

Remaining evidence gaps:

0. Central novelty risk: UNSIR supplies the core mechanism and PAR overlaps
   federated coordination. Paired controls do not support confidence ranking;
   method budget/objective differences do not isolate a new effective component.
   Requester-level effectiveness is untested. See
   `responses/CONTRIBUTION_EVIDENCE_AUDIT.md`.

1. Readable PAR full-paper verification and exact published-protocol comparison.
2. A fixed two-target/two-request study (54 formal stages) is now audited and integrated. Twenty-client simulation is complete and audited (36 stages); longer/arbitrary request streams remain untested and five-round retained-only post-event FL is complete and audited (24 formal stages), with no observed Fed-EMNU FA rebound but persistent target MIA differences. CIFAR-100 is now evaluated; Two ResNets remain one architecture family.
3. Primary study now has five seeds and descriptive t intervals; other studies have three. Small samples and multiplicity still limit inference; no universal significance is claimed.
4. Alpha=0.1 has poor original utility and only 4/5/7 target-holding clients;
   target-holder-only state replacement can omit unaffected-client contributions.
5. Original conflicting table/figure provenance is unrecoverable; no cause is invented.
6. Some inherited NeurIPS records remain independently unverified after TLS failures.
7. Author review of attribution, affiliations and journal requirements is required.

One loss-based MIA and one independent retraining draw per seed do not certify
deletion or privacy. See manuscript Section 5.6 and the frozen PAR protocol.

Text/table sources now integrate the second follow-up. Existing PDFs and ZIP remain the previous snapshot at user request; no PDF generated in this pass.

Sequence results integrated in Section5.9; no class0 accuracy rebound in six Fed-EMNU paths, but new-class residual behavior remains. No new PDF/ZIP generated.

Continuation integrated in Section5.11. Remaining central risks are contribution/novelty, PAR full-paper verification, sparse-owner event replacement, unavailable old provenance and limited inference beyond tested conditions. No further experimental success is inferred from low FA.
