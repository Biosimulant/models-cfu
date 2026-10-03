# Workspace workflow

Develop, fix and test on `staging`, not `main`. Before new work, synchronize
staging from any current main baseline while preserving staging-only commits.
Commit and push changes to staging only. This repository initially has only a
staging branch.

Production remains on Modal. Do not deploy changes or switch production
providers without a new explicit user instruction. Scientific runs, immutable
artifacts, publication and hosting use Biosimulant MCP. Local software fixtures
do not establish scientific accuracy or hosted readiness.

Preserve the frozen weights, source contract, inference settings and original
failed accuracy gate. Do not retune against the opened final test. Large assets
remain outside Git and must match the retained contract before execution.
