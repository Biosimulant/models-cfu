# Model Requirements Specification: CFU Counting

## 1. Document Control
MRS identifier: CFU-MRS-001
Version: 1.1
Status: Approved
Model owner: implementation agent, under Demi’s direction
Scientific owner: Demi; release claims limited to the annotated source domain
Validation owner: Demi; automated prespecified validation against independent held-out human annotations; no external-lab claim
Intended-use owner: Demi
Created: 2026-10-01
Last revised: 2026-10-03
Related biological brief: user-supplied implementation plan, pasted-text-1.txt, 2026-10-01
Related technical specification: mts.md

### Approval
Demi explicitly approved the proposed numerical count/detection gates and training recreation tolerance on 2026-10-01 via the requirements approval question. Approval receipt: “Approve these gates and recreation tolerance”. Demi also authorized implementation, a maximum $30 Modal spend, and autonomous completion of all requested operations. This approves requirements and execution, not unexecuted validation results. No final test outcomes have been opened by this implementation workflow.

Demi explicitly increased the total Modal cap to $50 on 2026-10-03, answering “$50 total (recommended)” after allowing more compute if needed. This budget-only amendment supersedes the original $30 limit; all scientific gates, split, intended use, recreation tolerance, and release approval boundaries remain unchanged.

### Revision History
0.1 — 2026-10-01 — implementation agent — initial draft.
1.0 — 2026-10-01 — implementation agent — Demi approved numerical gates and recreation tolerance; requirements frozen before final outcomes.

1.1 — 2026-10-03 — implementation agent — Demi approved a $50 total Modal cap; budget-only amendment, no scientific acceptance change.

## 2. Executive Summary
Count visible bacterial colonies in one plate photograph for research, with a marker on every detection so a scientist can inspect and correct mistakes. Success requires an actual trained model, untouched evaluation, repeatable execution, and a live hosted Hub Lab with inspectable lineage.

## 3. Biological Objective and Intended Use
Research assistance for clear, separated colonies in photographs compatible with the documented evaluation domain. Human review is required before using counts. A predicted count is an estimate of visible colonies, not a direct viable-cell measurement. Do not infer species, clinical conclusions, sterility, or CFU/mL. Dense merged growth, touching colonies, reflections, and new acquisition conditions require caution and separate evaluation. No claim of validation at unrelated laboratories is permitted without their independent data.

## 4. Model Definition and Scope
Selected type or types: Predictive / ML
Unit of prediction: one whole plate photograph.
One invocation is finite: one uploaded image produces a result or explicit failure. No simulated time evolution is required.
In scope: plate counts, colony locations, annotated images, metadata, data preparation/training/evaluation/recreation, and hosting provenance.
Out of scope: species identification, dilution calculations, automated scientific decisions, annotation editing, general billing, broad datalake UI, drift monitoring, and arbitrary adaptive campaigns.

## 5. Inputs
MRS-IN-001: One account-authorized retained JPEG or PNG, SHA-256 and byte length verified before execution. Operational bounds: 20 MiB and 25 million pixels; corrupt, unsupported, oversized, missing, or unauthorized inputs fail explicitly. These bounds are engineering limits, not biological acceptance thresholds.
MRS-IN-002: Exact model release and its inference configuration; user images may not silently change model or thresholds.
Acquisition conditions not represented in the held-out evaluation must be displayed as unvalidated. No experimental metadata may be fabricated.

## 6. Outputs
MRS-OUT-001: Nonnegative integer visible-colony count, unit colonies.
MRS-OUT-002: Bounding boxes in original-image pixel coordinates, detection scores, and annotated image. Scores must not be described as calibrated count uncertainty.
MRS-OUT-003: Input digest, exact model/release digest, configuration, execution ID, timing, warnings, and links to accessible evidence.
MRS-OUT-004: Failure must produce a machine-readable status and clear message, never a plausible default count. The displayed count must equal the length of the displayed deduplicated detections.

## 7. Data, Assay, and Evidence Specification
Primary source: ADBC, Figshare article 22022540, version 3, photographs and COCO labels. The source license and bytes must be verified and attribution retained. Dataset availability, image inventory, and annotation compatibility remain subject to data-readiness checks.
MRS-DQ-001: Preserve originals, annotation version, source/license, byte checksums, exclusions, and group-aware split manifest. Changes create a new dataset version.
MRS-DQ-002: Whole plates and byte-identical images cannot cross train/validation/test partitions. Tiles retain their parent plate’s split. Use acquisition groups where available; do not invent missing groups. Near-duplicate review must be recorded.
Train/validation/test allocation must be frozen before tiling. Test annotations cannot guide candidate selection, confidence tuning, or exclusions. All attempted plates and failures remain in evaluation denominators.
Blank-plate and external-lab evidence must be distinguished from annotated ADBC evidence. Missing control data is an evidence gap, not permission to fabricate biological validation fixtures.

## 8. Ground Truth or Estimand
Target: count of annotated visible colonies per whole image, compared with documented human bounding-box annotations. Labels are a reference standard with possible omission and ambiguity; they do not prove individual viable-cell counts. Ambiguous or merged colonies must be documented rather than silently redefined.

## 9. Generalization Domain
First release claims only performance on the independently held-out source-domain photographs and explicitly described compatible conditions. Species, density, acquisition groups, and imaging variation must be reported where metadata supports them. External and prospective laboratory generalization requires new independent photographs and human review.

## 10. Performance Requirements
Approved project gates, frozen before final evaluation:
MRS-PERF-001: Whole-plate weighted absolute count error, sum(abs(predicted-reference))/sum(reference), at most 10%; report MAE, signed bias, and bootstrap interval too. This is an approved project gate, not a published result.
MRS-PERF-002: Report the 95th percentile of per-plate symmetric absolute percentage error, 2*abs(predicted-reference)/(predicted+reference), treating two zeros as zero. Approved maximum: 20%. Preserve all severe errors and density strata.
MRS-PERF-003: Report one-to-one bounding-box precision and recall at IoU 0.5, false positives, misses, tiling duplicates, blank-plate behavior where real controls exist, and failures. Approved precision/recall floor: 90% each. Metric thresholds must not be weakened after test outcomes are opened.
MRS-OPS-001: Saved-model evaluation replay must produce the same counts and identical reported aggregate metrics in the pinned environment. Runtime/cold-start/warm-start measurements must be recorded; no latency promise is made before benchmarking.
MRS-OPS-002: Training recreation must actually execute from pinned inputs and yield aggregate count error within two percentage points of the selected model, with both models passing the approved gates. This tolerance was approved before recreation outcomes; weights need not be byte-identical.

## 11. Validation Plan
Freeze weights, configuration, code, environment, exclusion rules, metric definitions, candidate-selection rule, and test partition. Select using validation data only. Execute the locked test once, retain per-plate predictions and error images, and derive every summary from those predictions. Perform evaluation replay and separate training recreation. External lab validation is mandatory for an external-lab performance claim; it is not replaced by software tests. Include corrupt/oversized input, permissions, duplicate tiling, deterministic replay, and difficult-image checks. Synthetic software fixtures must be labeled as such.

## 12. Acceptance and Traceability
MRS-001: Public hosted Lab counts a new image with a real trained checkpoint; separate-account staging and production invocations are required evidence.
MRS-002: Every counting result traces to input, weights, training run, immutable dataset, code, configuration, environment, evaluation, release approval, and deployment. Public views omit inaccessible upstream records.
MRS-003: Another authorized workspace/account can retrieve verified retained artifacts, replay evaluation, and recreate the training workflow.
MRS-004: Existing scalar-input Labs, historical Runs, status meanings, and issued Passports remain compatible. Additive migrations and independent UI/host/deployment rollback are required.
All requirements are Not tested. Completion requires gates and evidence, not package compilation or isolated tests.

## 13. Constraints, Risks, and Governance
Modal spending cap: $50 total for this workflow, explicitly authorized by Demi on 2026-10-03, superseding the original $30 cap approved on 2026-10-01. Start with a 10-minute benchmark and estimate full training before launch. Retain failed and cancelled runs. Scale to zero, bounded concurrency, no unattended paid resources.
Software and data must permit intended redistribution/commercial reuse with notices; no Ultralytics Enterprise purchase is authorized. Production remains Modal; final production promotion follows review of the exact qualified release and deployment. Preserve the presentation deployment until that step.
Leakage, label errors, domain shift, merged growth, missing control data, and overinterpreted detection scores must appear in the model card. Public sharing must not implicitly expose private data or credentials.
Changes to intended use, ground truth, thresholds, or supported domain require a new version and approval. Never loosen acceptance because training under budget failed.

## 14. Reproducibility, Deliverables, and Handoff
Deliver retained original data/labels, immutable split and preparation manifests, code revision, pinned dependencies/environment, seeds, configuration, checkpoint, all training outcomes, actual predictions, metrics, failure examples, recreated-workflow evidence, Lab archives, evidence card, deployment history, and verified live invocation. Imported execution evidence, managed evidence, and independently reproduced evidence must be visibly distinct. A signed Passport authenticates its recorded statement; it does not automatically prove biological validity.
Rollback: disable new UI independently, pause CFU hosting, restore previous serving release. Do not alter historical Passports.

## 15A. Predictive / ML Module
MRS-ML-001: Training examples and tiling preserve sample/group identity. Feature availability is one photograph at inference; labels and outcome-derived information are unavailable.
MRS-ML-002: Compare raw whole-image and tiled predictions using validation data, and preserve the decision rationale. Human corrections are review evidence, not automatic retraining labels. Display count with inspectable detections rather than claiming calibrated confidence in a whole count.

## 16. Milestones and Responsibilities
A: retained data and recorded training checkpoint.
B: approved evaluation and executed reproduction.
C: staging image hosting, readable lineage and compatibility verification.
D: exact public Hub release, Modal hosting, external-user execution and production smoke.
Demi owns intended-use and spending decisions; implementation agent owns engineering and evidence capture. Demi is the release decision owner. Automated held-out evaluation uses independent human annotations; it is not described as an independent human scientific review.

## 17. Open Questions and Decisions
DEC-003: Demi approved count error ≤10%, 95th-percentile symmetric error ≤20%, detection precision and recall ≥90% at IoU 0.5 on 2026-10-01.
DEC-004: Demi approved recreation aggregate count error within two percentage points, with both models passing gates, on 2026-10-01.
Future external-domain claims: real blank controls and external-lab review images are not yet verified available. These claims are excluded from this release unless separate independent evidence is acquired; Demi owns any future scope expansion.
DEC-001: Use permissively licensed small detector and tiled inference, authorized implementation scope from pasted plan; architecture to be recorded in the MTS.
DEC-002: Maximum total Modal spend $50, explicitly approved by Demi on 2026-10-03 (“$50 total (recommended)”), superseding the original $30 cap. Optimize usage and stop before exceeding it.

## 18. References and Glossary
ADBC primary dataset: https://figshare.com/articles/dataset/Annotated_dataset_for_deep-learning-based_bacterial_colony_detection/22022540/3
ADBC primary description: https://pmc.ncbi.nlm.nih.gov/articles/PMC10382471/
YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
SAHI: https://github.com/obss/sahi
CFU: colony-forming unit; this model estimates visible colonies in an image, not CFU/mL.
