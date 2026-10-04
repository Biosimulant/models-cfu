# Does GPT vision improve colony counting for the money?

Specification **CFU-GPT-COMP-001, draft 0.1.0**, 4 October 2026.
Status: comparison setup and incremental budget pending agreement. No GPT
inference has run. This is a new comparison protocol, not an amendment of the
historical approved MRS/MTS or scientific gates.

## Decision and scope

Compare the frozen specialist's retained counts with GPT's own visual counts.
Primary GPT identifier: **`gpt-6.1-sol`**, `reasoning.effort: high`, selected by
Demi on 4 October 2026. An agent invoking the counter is detector evidence.
No training, detector changes, production deployment, provider switch or public
publication is included. Production remains on Modal.

Separate count performance, localization, value, and human review utility. The
hypothesis that the specialist is better at dense localization and GPT helps
sparse/contextual review remains untested. One GPT configuration cannot establish
which of all GPT models offers the best value.

## Verified baseline and provenance

The staging repository was clean at `e9cd591e2bd057e71a3a0e5fd322a305c202f793`.
After fetching origin, both staging refs matched. Remote heads contain staging
only; there is no main baseline in this dedicated repository to merge. No
staging-only commits were discarded.

Frozen release: `demi/plate-colony-counter@0.1.1`, LabVersion
`b6e2cd4a-e6c1-4ea2-b529-abb530f30f05`, package SHA-256
`77cb99f47f032e9921ce653875a38d5dbd2b0e3757b2d322d61631a670c4d57d`.
YOLOX-Tiny EMA step 18,601, epoch 30/cursor 1,456, weight SHA-256
`27068af634b80c51d3c318d8fd7e8c1d79269fa00ce62b557f5b9cf3e5d9cd90`,
20,372,446 bytes. Keep the actual inference contract and all source hashes.
The tiled configuration is `tiled-GREEDYNMM-IOS-0.50-conf0.40-relative-area0.25`:
640px tiles, overlap .2, batch 8, confidence .4, GREEDYNMM/IOS .5. The filter
uses the median area of predictions scoring at least .7 when there are at least
three anchors; keep area >= .25 times that median. With fewer anchors, do not
filter. These are retained rules, not a proposed reconstruction or new tuning.

Read-only authenticated MCP inspection confirms production hosting is public,
ready, revision 6, approved, same package, T4; it does not establish new inference
or scientific qualification. Hub inspection still reports runtime qualification
unestablished and domain validation false. Hosting readiness and these flags
are distinct. The original evaluation's READY Passport does not override the
failed scientific gate.

Original managed run `76c52ab5-f41c-4a47-ba48-83eeb07f5265`, immutable revision
`e935877b-5cdd-4be0-971a-dd6d06dd8010`, completed on Modal T4. Original split
run `17ae012f-bb0f-4c3d-96f5-ab376aa94d62` is retained. Live MCP lists the
original artifacts. Their local retained copies were independently rechecked
against those live sizes/hashes; a fresh owner-bound download attempt returned
HTTP 403. No direct REST substitution was used. This audit verified cached bytes
against live identities rather than claiming a successful fresh download.

`audit_baseline.py` verifies the five artifact identities, all original COCO
references in source order, image dimensions, groups, and frozen-test membership.
It re-runs the unchanged, checksum-verified metric code on saved records, without
model execution. All metric fields, match lists and the seeded grouped bootstrap
are exactly equal to the retained summary.

| Metric | Recovered detector result |
| --- | ---: |
| Original plates / reference colonies | 37 / 6,409 |
| WAPE / absolute count error | 2.9021688251% / 186 |
| MAE | 5.0270270270 colonies |
| Signed bias | +1.8378378378 colonies per plate |
| p95 symmetric count error | 22.2222222222% — original 20% gate failed |
| IoU .5 precision / recall | 91.4466574031% / 92.4169137151% |
| TP / FP / FN | 5,923 / 554 / 486 |
| Group-bootstrap WAPE 95% interval | 1.8369902865%–4.3526160758% |
| Failed detector outcomes | 0 |

Runtime: Python 3.12.3, Biosimulant 0.0.34, Torch
2.6.0+cu124, torchvision 0.21.0+cu124, NumPy
1.26.4, CUDA 12.4, cuDNN 90100, T4. Training cost is a sunk cost, reported
separately; independent training recreation is still incomplete.

## Input inventory and blinding

The actual 37 test IDs come from the checksum-pinned split and are checked
against the original run, not `validation-plan.json`. The recovered private
`test-inventory.json` holds every sample/group/File ID, source digest/size,
original width/height, density stratum and deterministic subset selection.
`per-plate-results.csv` contains all detector outcomes and explicit `not_run`
GPT fields; those fields are not failures or benchmark measurements.

Proposed density strata, fixed before GPT outcomes: sparse 0–30 source colonies,
moderate 31–300, dense >300. They contain 10/15/12 test plates. The thresholds
are design choices made after the detector test was opened; they are not a
previously preregistered detector analysis. Source counts determine sampling
strata only; never place them in GPT requests.

Use six validation images, two per stratum, from separate related-image groups,
chosen by SHA-256 rank of `cfu-gpt-dev-v1:<sample_id>`. No test photograph,
reference, detector prediction or test-error example is used for prompt choice.
Historical validation annotations may score development responses but remain
outside requests. This is new GPT development data, not a fresh external dataset.

Every request uses one isolated conversation and a neutral input label. No
filenames, source counts, annotation overlays, detector outputs, URLs identifying
the dataset, tools, search, example answers or prior responses enter GPT-alone.
Persist the exact rendered prompt/schema/settings and SHA-256 before requests.

Original image bytes remain to be retrieved/verified for all 43 selected source
photographs (37 test + 6 development). The present audit verifies manifests and
annotations, not every image byte. Before execution check File ownership,
source byte length/SHA-256, decoding, EXIF orientation and dimensions. Preserve
originals; normalize with pinned Pillow 12.1.0, EXIF transpose and RGB conversion,
then retain a lossless PNG derivative, its digest and original-to-input transform.
Compare normalized dimensions with the source coordinate frame and explicitly
transform annotations for nonidentity EXIF; do not assume they agree.

Preferred proposed image setting is **`detail: original`**, independent of high
reasoning. Current vision documentation has explicit sizing rules for Astra but
omits Sol's exact patch/resizing rules. Sol image input and structured output are
documented; original-detail support and its precise preprocessing must be
confirmed before final freeze. Do not borrow Astra's multiplier for Sol or
silently substitute high detail. If support cannot be established, amend this
draft before test execution. Record any unknown provider preprocessing.

Two originals, `sp21_img38.jpg` (5824×5919) and `sp21_img39.jpg` (5535×5504),
exceed the deployed detector's 25M-pixel input bound. Their original offline
predictions remain in the 37-plate accuracy comparison. Do not resend them to
the hosted counter or call a resized serving result equivalent to the original.
If GPT requires downsampling, specify its deterministic rule and transform before
freeze, report the resized subgroup, and keep all outcomes. A whole-image GPT
with different effective resolution is a system comparison, not equal-resolution
evidence about model architecture. Native-resolution GPT remains unestablished
where preprocessing is unknown. No crops or GPT tiling in the primary arm.

## Bounded arms, settings and development

Use Responses API, Standard tier, one image/request, no tools, `store: false`,
high reasoning, sequential calls, strict JSON schema, no ensemble. Omit
temperature/top_p/seed unless explicitly supported and added before freeze;
do not claim deterministic sampling. Record response model ID, request ID, SDK
version, SDK retry configuration, usage including reasoning/cache tokens,
provider/tier and timestamps. Model page lists `gpt-6.1-sol` as the selectable
snapshot; no dated alternative was documented. Do not invent a dated identifier
or promise immutable provider weights.

1. **Detector saved baseline:** reuse all 37 verified records. Accuracy scoring
   costs no new GPU inference. The runtime/evaluator stay frozen.
2. **GPT-alone count:** development tests both checked-in count prompts once on
   each of six images: at most 12 calls. Choose lower failure-penalized development
   WAPE, then fewer failures, then lower p95, then prompt v1 for an exact tie.
   Validate schema on development only and freeze chosen bytes before testing.
   Run all 37 test plates once with the fixed prompt, `max_output_tokens: 16384`
   including reasoning. High reasoning may exhaust the output budget; incomplete
   results are explicit failures, not a reason to increase the cap on test images.
3. **Repeatability:** nine preselected test plates (three per stratum, hash rank
   `cfu-gpt-repeat-v1:<sample_id>`), two additional calls each, with identical
   frozen settings. Total three attempts/plate including the primary response.
   Primary accuracy uses only the first call; do not pick a best answer or average
   responses into that result. Report count range, sample SD, exact agreement,
   failure frequency and box variability where repeated boxes exist.
4. **GPT-alone localization:** a separate count-and-box prompt/configuration.
   Six development checks (one per development plate), then one call on each of
   nine independently preselected test plates (three per stratum, hash rank
   `cfu-gpt-localize-v1:<sample_id>`), `max_output_tokens: 32768`. Do not use its
   counts to replace the primary count arm. No reference or detector information
   enters these requests. The localization prompt can be revised from development
   checks before freeze; no further development calls are in this allowance.
   Report localization on this subset only, paired against the same detector
   subset. Full 37-plate localization parity and localization repeatability are
   unvalidated by this bounded phase.
5. **Optional hybrid:** excluded from the proposed paid allowance. A future
   separately frozen arm sees the same photograph plus indexed detector boxes,
   scores and actual overlay, returns review flags/regions/reasons, and retains
   the detector count. Judge flag sensitivity, false-alert rate, review burden and
   expert-confirmed usefulness. Without blinded expert review and paired outcomes,
   agreement, disagreement and persuasive explanations do not prove benefit.

The private inventory materializes the exact six/nine/nine IDs; selection never
uses detector error or GPT response quality. Development counts and boxes are
stored separately from frozen test outputs. No test tuning, alternate models,
retries or replacement results are authorized by this plan.

## Scoring and uncertainty

Preserve original `cfu-whole-plate-metrics-v2` at SHA-256
`c05d24fcf8fdcec0748dd231b984b5d6c63cf9377216ea2005ad732837bbdb2b`.
Use its deterministic maximum-cardinality one-to-one IoU >=.5 matching and
unmodified COCO XYWH→XYXY reference order; out-of-frame references retain their
full area in IoU and zero-area references remain count/FN entries.

A new explicitly versioned count adapter is required because the old evaluator
requires count == number of boxes. Do not create fictional boxes for count-only
responses. For a valid integer count C and reference R, use |C−R|, C−R, and
2|C−R|/(C+R), both-zero =0. WAPE=sum absolute errors/sum R;
MAE=sum absolute errors/37; signed bias=mean signed error only when all 37
counts succeed (otherwise null, with labeled successful-only bias separately).
p95 uses linear interpolation at (n−1)×.95. Refusal, timeout, transport failure,
schema/parse error, missing/incomplete output or cannot_count is a failed primary
attempt: count=null, absolute-error penalty R, symmetric penalty 2; blocks
acceptance. Keep error type, raw response, request ID, spend and elapsed time.
Also report unpenalized success-only metrics with their smaller denominator and
failure rate, so failure penalties cannot be mistaken for predictions.

Count and localization statuses are independent. Invalid or absent boxes never
discard an otherwise valid primary count. Coordinates from the localization
arm must be finite positive-area input-frame boxes, mapped by the retained
transform to EXIF-normalized original pixels. No clipping or silent repairs.
Duplicate predicted boxes remain predictions and can become false positives.
For complete localization require box count == reported count; contradictions
are localization failures. Partial boxes are labeled partial and evaluated as
the emitted set, with unlocalized references as misses; no complete-overlay claim.
Use constant score 1.0 only as a documented deterministic matcher tie-break,
never as confidence. Refused/invalid localization on a scheduled plate has
TP=0, FP=0, FN=R, explicit failure and blocked completeness. Count-only arm
localization is N/A, never perfect or zero precision. Undefined denominators
remain null. Point outputs need a separate prespecified point metric and are
outside this phase; do not convert them into IoU boxes.

Grouped paired bootstrap: 2000 replicates, seed 20261001, sample frozen groups
with replacement and preserve both arms' records within each drawn group. Report
95% percentile intervals for each WAPE and paired GPT-minus-detector WAPE, MAE,
p95 and supported localization metrics. Preserve zero-denominator/null replicate
counts. The recovered test has 37 screening groups for 37 images; these are
conservative duplicate-screening groups, not proven independent acquisitions.
Tail intervals at n=37, and especially localization at n=9, are exploratory.
Do not infer laboratory independence or precise tail risk from them.

Keep all original detector gates visible; GPT count-only can be assessed against
count gates but cannot pass a full detection-output requirement. Report strata,
oversized/source-preprocessing subgroup and all failures. Report paired error
plots and precision/recall only at their actually evaluated scopes.

## Latency, costs and useful overlays

Historical hosted worker 22.14s, hosted elapsed ~117.68s and ordinary elapsed
~135.93s use different boundaries and are not comparison benchmarks.
New instrumentation records input preparation/upload, dispatch-to-first-output,
dispatch-to-terminal, queue/cold setup, actual detector preprocessing/forward/
merge/render time, collection and durable-result/visual availability separately.
Use monotonic clocks for durations and UTC timestamps for audit. OpenAI server
processing/queue timing is null unless exposed; client elapsed does not establish
model-processing time. Report comparable complete-result elapsed separately.

If authorized and a staging/private harness is available, run the nine eligible
repeatability originals on an isolated Modal T4 harness built from staging with
the exact frozen package: one call each, then two repeats each (27 image
inferences), in at most two 1800s GPU assignments, concurrency one. First call
per assignment is cold; others are labeled warm in actual order. Never interpret
an offline batched timing as hosted end-to-end latency. If per-image serving
latency cannot be collected within these bounds, report the gap rather than
benchmark all 37 again. Two independent assignment starts provide limited cold
evidence. Any new staging hosting/deployment would need separate authorization.

All new scientific runs/evidence use authenticated Biosimulant MCP; implement
from staging, use an isolated private harness or staging MCP without changing
production. OpenAI credentials must be supplied through an authorized secret
binding, never written into manifests, Git, logs, artifacts or prompts. Confirm
that the managed runner permits the needed network/secret binding; the existing
hosted counter has blocked network and cannot itself make GPT calls. Do not
route API inference through Studio Chat or confuse ChatGPT/Codex usage with
billed API cost. No new run plan/digest has been prepared in this inspection.

Track actual API input/output/reasoning/cache usage, price version, returned tier,
failed-call spend, unknown timeout billing and invoice reconciliation. Count
reasoning as part of output usage, not an extra double-counted fee. Conservative
reservation uses uncached input; never assume cache savings. Detector report
includes allocated CPU/memory/GPU cost and full allocation times, failures and
collector overhead; managed user_charge=0/free_beta is distinct from provider
spend. Historical cumulative Modal invoices and the old $50 allowance remain
unreconciled; no remaining balance is assumed. Report dollars per attempted
plate, per successful count, and per complete annotated result separately.

Render actual emitted coordinates programmatically over the source photograph,
with retained transform and readable index labels. Retain detector TP/FP and
source-FN overlays and GPT disagreements; unlocated count differences cannot
identify which colonies GPT missed. Predefine all-nine localization panels plus
largest absolute count disagreements per stratum, labeling that retrospective
illustration rule. Optional expert corrections are separate review records and
cannot overwrite the benchmark reference. Scientific outputs stay typed and
checksum-pinned. Write under relative `outputs/`, keep files until collection,
use the same file paths in typed outputs and image VisualSpecs, retrieve through
authorized run context, verify bytes and reopening. A visual count is insufficient.

## Incremental budget proposed for agreement

Official model prices checked 4 October 2026: Sol input **$2/M**, output **$10/M**
on Standard short context. A documented cache write can cost $2.50/M; reserve
that upper input rate. Astra is $10/$50; Luna $0.10/$0.50. Only Sol is selected.
No result establishes their colony accuracy, latency or best value.

| Included stage | Maximum API calls | Output ceiling per call |
| --- | ---: | ---: |
| Count development: 6 images × 2 prompts | 12 | 16,384 including reasoning |
| Frozen primary test | 37 | 16,384 including reasoning |
| Repeatability: 9 images × 2 more calls | 18 | 16,384 including reasoning |
| Localization development | 6 | 32,768 including reasoning |
| Localization test subset | 9 | 32,768 including reasoning |
| Total | **82** | 67 count calls + 15 localization calls |

Require at most **40,000 billable input tokens/request**, verified by supported
preflight token accounting before dispatch; this is a cost bound, not a guess at
Sol's image multiplier. If a supported bound is unavailable, do not make a paid
call until the cost envelope is resolved. Each count reserves
$0.10000 + $0.16384 = **$0.26384**; each localization reserves
$0.10000 + $0.32768 = **$0.42768**. Total maximum modeled API spend:
67×.26384 + 15×.42768 = **$24.09248**.

Propose **$25 new OpenAI API cap + $3 new Modal provider cap = $28 incremental
maximum**. These are ceilings, not predictions of actual bills. No retries are
included; SDK automatic retries must be disabled. A timeout remains reserved
until usage/billing is reconciled. Before each call reserve its entire bounded
cost; stop before exceeding either sub-cap. No Fast/Ultrafast, Batch, Flex, regional
surcharge, paid tool or long-context pricing is included; a price/tier change
requires recomputing the envelope before execution. Two T4 allocations must fit
the $3 provider ceiling under the live exact compute plan; if not, reduce scope
before agreement, never silently consume the historical $50 allowance.

Count-only alternative: 67 bounded calls, modeled maximum **$17.67728**, propose
**$18 API** plus $3 optional Modal, **$21 total**. Excludes new GPT boxes and
localization examples. Hybrid and other GPT models require their own budget.

## Freeze, blockers and decision report

Before test calls: agree this setup/budget; confirm API account model access,
image-detail/preprocessing support and input-token bound; verify all source
image bytes; build and verify the independent count/localization adapters on
staging; complete bounded development; freeze prompt/schema/settings, actual
input derivatives/transforms, evaluator version and inventory hash. Any
unresolved item remains a named blocker. Then prepare one bounded Biosimulant
experiment with separately named arms and exact revisions/requests/bounds, retain
its digest and obtain any required exact compute approval. Budget agreement
does not fabricate a grant or approve an unknown MCP digest. No publication
is included.

Final results: all-37 per-plate table including every attempted failure; separate
count/localization metrics and grouped intervals; all attempt-level raw responses,
repeatability and timing; actual billed and provider costs; retained overlays and
reopening evidence. Recommend best observed count accuracy, best supported
localization and cost/accuracy tradeoff separately. Call a result inconclusive
when uncertainty or missing costs prevents a ranking. Full localization and
hybrid benefit remain unvalidated under this first bounded phase. Any model or
prompt improvement after test inspection requires separate development and
fresh untouched confirmation, preferably independent acquisition. Public ADBC
pretraining exposure is unknown; detector-held-out is not necessarily GPT-unseen.

Environment report distinguishes: branch synchronization (staging-only remote,
clean baseline), local saved-evidence audit (Python 3.14, staging; no inference),
isolated adapter/software checks (not yet implemented), live staging end-to-end
tests (not run), deployment (none). No compute resources were created in this
inspection. No provider was switched; Modal remains selected; no rollback action
was necessary. The source commit containing this draft is recorded by Git.

## Reproduce the recovered baseline

Run on staging with account-authorized local copies matching the listed hashes:

```sh
python3 specifications/colony-vision-comparison/audit_baseline.py \
  --evidence-dir /path/to/retained-private-evidence \
  --output-dir /path/outside/git/comparison-evidence
```

Do not commit raw photographs, generated predictions, per-plate evidence,
credentials or signed URLs. New development outputs do not alter the frozen model.

## Sources

- [Sol model, settings and pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
- [Current models](https://developers.openai.com/api/docs/models)
- [API pricing](https://developers.openai.com/api/docs/pricing)
- [Image preprocessing, counting/localization limitations](https://developers.openai.com/api/docs/guides/images-vision)
- [ADBC source dataset description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10382471/)
- [Object counting study](https://arxiv.org/html/2512.03233v1): FSC-147/CARPK,
  not colonies; crowded failures and extreme-error exclusions prevent borrowing
  headline metrics as our baseline.
- [Hybrid pharmaceutical quality-control preprint](https://arxiv.org/abs/2602.20543):
  motivates a separately evaluated review workflow, not GPT-alone superiority.
- Historical repository MRS/MTS, frozen inference contract and original evaluator;
  private operational verification `cfu-fix-verification-20261004.md` and retained
  bytes are recorded in the baseline audit.
