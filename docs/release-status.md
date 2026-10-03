# Release status at repository migration

Checkpoint: 2026-10-03. These are recorded observations, not a live status page.

Later checkpoint: public hosting was approved and became ready, all four
deployment checks passed, and an actual hosted image invocation reproduced
the seven saved detections and identical annotated PNG. See
[public hosted verification](hosting-verification-20261003.md) for exact
identities, artifact hashes, runtime, interface checks and limitations.
The pending approval description below records the earlier migration state.

Counter: `demi/plate-colony-counter@0.1.1`, public, package SHA-256
`77cb99f47f032e9921ce653875a38d5dbd2b0e3757b2d322d61631a670c4d57d`.
Lab: `fb09b12a-975a-43af-8a70-652ed2c9d362`.

Exact immutable source revision: `6761a1d6-76bd-443c-a818-1ef75edaf983`,
SHA-256 `1670ebe07bd519cabd742c54bde28cb12e9f0dcac17a0b5cf6e357a6d7c9c31d`.
Its managed Modal T4 run `576fa4d8-168f-4550-9d18-daa870bbd62a` completed:
count 7, all saved boxes/scores equal, result and PNG bytes independently
verified, required image/text visuals delivered. Its execution Passport was
READY for the declared experimental intended use. This does not erase the
failed scientific accuracy gate.

Public hosting request was applied for the exact counter release, one Modal T4
container, zero warm containers and 300-second idle scaledown. Hosting record
`5b389cf7-1ab2-4515-b83d-32acbe4c7304` was awaiting administrator approval.
No deployment checks had run, and no ready hosted invocation was available.
The native MCP connection did not expose an administrator approval action.

Public Results/Experiments visibility is separate from publication. Do not
claim those tabs are populated merely because the package is public.

The repository move changes source ownership/layout only. It does not replace
an immutable Hub package, redeploy the backend, change providers, submit
training or retune the detector. Git pushes are restricted to staging.
