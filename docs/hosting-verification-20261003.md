# Public hosted verification

Checkpoint: 2026-10-03, after the owner explicitly approved hosting and the
administrator approval step. This records observed state, not a live monitor.

## Release and deployment

Public Hub: https://hub.biosimulant.com/labs/fb09b12a-975a-43af-8a70-652ed2c9d362

Release: `demi/plate-colony-counter@0.1.1`.
Package SHA-256:
`77cb99f47f032e9921ce653875a38d5dbd2b0e3757b2d322d61631a670c4d57d`.
Hosting record: `5b389cf7-1ab2-4515-b83d-32acbe4c7304`, revision 3,
public, approved, ready. Specification:
`77aa8bc4-bf73-46b0-84ee-d5a2dbe6ef1e`, SHA-256
`94fbb33032f0ff8a9aba71681d0a96fc34a5242c16130641e152170691c876ee`.

Administrator approval used the authenticated Biosimulant Admin interface
because the native MCP connection exposes no administrator review action.
The approval audit is dated 20:23:44 UTC. Deployment and scientific checks
were read or executed through Biosimulant MCP. No backend production
deployment or compute-provider change was performed.

Production provider is Modal: one T4 container maximum, 4 CPU, 16 GiB memory,
1800-second invocation timeout, zero warm containers, 300-second idle
scaledown, blocked runtime network. Archive, runtime image, archive-cache
and smoke checks all passed. The smoke check took 24.8813 seconds; its empty
values do not establish trained inference. The actual image invocation below
does establish execution of the packaged detector.

## Actual hosted inference

Invocation: `eda64ed7-847a-4b3c-b0d2-c47301dc4474`, completed, attempt 1.
Prepared plan: `a9312221-9494-49e6-a479-82665517ac52`, SHA-256
`1b84399acfb3713f60664c3c020542a6a557c247f489397d06f91a96ac13abaf`.
Input: owned retained `sp01_img04.jpg`, 1,285,071 bytes, SHA-256
`d24c04909ca046506c19f3a7ed747530a0e120a75400fe7e0bd64aa274cbd79f`.
The stored JPEG is 2813 x 2804 pixels, EXIF orientation 1.

Accepted at 20:33:37.783113 UTC; completed at 20:34:04.456949 UTC.
Hosted result reports 21.6209 seconds execution and 16.788194 seconds for
detector setup, inference and annotation. These are single observations,
not a latency benchmark or a cold-start guarantee.

The full retained result contains count **7**, exactly seven detections,
original-image boxes, detector scores and the provenance/environment receipt.
Every box and score exactly equals the previously verified managed result
for this release. Its annotated PNG is also byte-identical.

Raw JSON: 703,101 bytes, SHA-256
`5f1f3e18a1a2fcc55bcbaeb03080cf4ffecedd350a3948d8306f36682b89cb93`.
Retained PNG: 522,399 bytes, SHA-256
`f4670ae7ab30bc945f27891b7d69cd286ac2b8efbbb0ca48e8367820b549c21a`.
Both raw artifacts were downloaded using MCP-returned owner-bound links,
then independently checked for length and SHA-256 before interpretation.
The PNG is 716 x 778, including its warning footer.

Runtime receipt: Python 3.12.3, Biosimulant 0.0.34, Torch 2.6.0+cu124,
Torchvision 0.21.0+cu124, NumPy 1.26.4, CUDA build 12.4, cuDNN 90100.
Frozen weights SHA-256 remains
`27068af634b80c51d3c318d8fd7e8c1d79269fa00ce62b557f5b9cf3e5d9cd90`;
inference contract SHA-256 remains
`18deb9b8a8982f01cf4db749c5bbcdd4ec34af7dc681d9724017cb9ffd3cd794`.
No weights, inference settings or model source changed during hosting.

The compact MCP status includes an empty `values` mapping. The full retained
result contains `outputs.detector.count.value = 7` and the count text visual.
Do not interpret the compact summary as absence of detections. The public
frontend renders the text visual and offers the PNG download; browser result
rendering itself was not exercised in this checkpoint.

## Hub interface and remaining limits

Read-only browser inspection verified the public page displays **Hosted** and
**Try hosted Lab** opens the exact release's authenticated image-upload form.
The form accepts one JPEG or PNG, up to 20 MiB and 25 million pixels. Uploads
and results remain private to the Studio account. A fresh browser file-picker
upload was not submitted: actual scientific inference used the MCP-owned
stored-image input above, following the owner's MCP workflow instruction.

The immutable About text still records hosting as pending at publication time.
Its current Hosted badge and this later checkpoint record the new state.
Standard public Results/Experiments tabs do not automatically expose private
managed runs or hosted invocations. No run-visibility change was made.

This is operational verification of the experimental release, not scientific
qualification. The original p95 symmetric count error remains 22.2222%, above
the approved 20% gate; independent training recreation remains incomplete.
The sparse plate has five reference annotations, so seven detections must not
be presented as seven ground-truth colonies. Human review remains required.

The owner-approved total Modal cap remains $50. One bounded hosted verification
was submitted; provider invoices, image-build and idle costs were not audited
here. The platform's 1800-second timeout bounds an invocation, not total account
spending. No further training, repeated fixtures or new providers were used.

## Source and rollback

Source extraction commit: `models-cfu@779fd128093e0facdbfc3e2438a8e9a72272191b`
on staging. This checkpoint adds documentation only, also on staging.
The repository has no main branch to synchronize. Backend extraction remains
on staging at `4b1dfd3529920db7bc3eafb52c9d78c3c9de40da`; it was not deployed
by this task. Scientific execution occurred in production against the exact
authorized Modal hosted release, not as a staging end-to-end test.

No temporary compute harness was created. Completed invocation artifacts are
retained; the hosted container may scale to zero after the configured idle
window. Rollback is to pause or withdraw hosting through Biosimulant while
retaining the immutable public package and evidence. Hosting has been left
ready as requested; no rollback or provider switch was performed.
