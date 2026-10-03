# Repository extraction verification

Date: 2026-10-03. Source baseline:
`Pledre/biosimulant-backend@cd1b8312b2db5c29ec494d3f063bd8ac2ebfbcc2`.
The existing staging worktree was fast-forwarded from the current main baseline;
that baseline already contains the prior staging commits. No staging-only work
was discarded. This was branch synchronization, not a production deployment.

The migration manifest covers 174 moved files: 169 are byte-identical and five
have relocation-only test/helper changes. Changes remove backend dependencies
from model fixtures, update repository-relative paths and replace one personal
source-fixture path with `CFU_VERIFIED_YOLOX_SOURCE`. All 14 frozen inference
source files and the inference contract are unchanged. Sixteen `model.yaml`
files and the single-image Lab contract pass repository checks.

Standalone software suite: 417 passed, one skipped. The skipped optional
source-import test requires a retained checksum-pinned source archive. Four
warnings came from intentional duplicate-member ZIP rejection fixtures.
`git diff --cached --check` passed.

Fixture environment: macOS arm64, Python 3.11.16, Biosimulant 0.0.34,
NumPy 2.4.6, Pillow 12.1.0, PyYAML 6.0.3 and pytest 9.0.2. This is the existing
software-only fixture environment; it is distinct from the pinned scientific
inference environment. No checkpoint was deserialized, no trained detector was
executed locally, and no training, hosted deployment or managed compute was
requested by this extraction.

Platform collector and hosted-image-limit checks remain with the backend and
do not import the new model repository. Historical private platform audit
records remain in the backend. The new public repository contains source,
contracts, specifications, synthetic tests and provenance, not backend history,
private operational evidence, credentials or large model/dataset assets.

All new commits and pushes use staging. Existing Hub releases and Modal hosting
are independent immutable resources and are not changed by the repository move.
Rollback: restore the moved backend paths from the recorded baseline or revert
the backend staging extraction commit; the dedicated repository remains intact.
