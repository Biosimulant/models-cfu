# models-cfu

Biosimulant colony-counting models and Labs, extracted from the backend into a
dedicated repository following the `Biosimulant/models-*` collection layout.
The runnable image-counting Lab is self-contained under
[`labs/plate-colony-counter`](labs/plate-colony-counter).

| Directory | Purpose |
|---|---|
| `labs/plate-colony-counter/` | One-image visible-colony counter, model source, immutable inference contract, MRS/MTS, caveats and portable tests. |
| `labs/acquisition/` | Source acquisition components and source-capture fixtures. |
| `labs/preparation/` | Duplicate grouping, frozen split, image tiling and transport verification components. |
| `labs/training/` | Scratch-trained YOLOX-Tiny training and checkpoint/resume components. |
| `labs/evaluation/` | Validation, pretest locking, original-image evaluation and software fixtures. |
| `research/` | Acquisition utilities and independent artifact/geometry/history verification scripts. |
| `specifications/cfu-counting/` | Original approved requirements and technical specification, retained as historical records. |
| `docs/` | Migration provenance, artifact requirements and release status. |

Acquisition, preparation, training and evaluation directories retain development
components assembled through immutable Biosimulant workspaces. They do not each
contain a standalone `lab.yaml`; do not describe them as independently runnable
Hub releases. The published research Lab remains the retained benchmark record.

## Published Labs

- [Plate Colony Counter](https://hub.biosimulant.com/labs/fb09b12a-975a-43af-8a70-652ed2c9d362):
  `demi/plate-colony-counter@0.1.1`, public experimental single-image counting.
- [Recorded Training and Evaluation](https://hub.biosimulant.com/labs/70c5eb8d-6822-4ebe-8247-0619ee05fd29):
  `demi/cfu-counting-recorded-training-and-evaluation@0.2.0`.

The counter accepts one JPEG/PNG and returns an integer visible-colony count,
an annotated PNG and an audit receipt. Mandatory human review applies. On the
37-plate held-out test, aggregate count error was 2.9022%, precision 91.4467%
and recall 92.4169%. The p95 symmetric per-plate error was 22.2222%, exceeding
the approved 20% limit. Independent training recreation remains incomplete.
No species identification, CFU/mL, sterility or clinical claims are supported.

At the migration checkpoint, public Modal T4 hosting was requested and was
awaiting Biosimulant administrator approval. Hosted execution is not yet
verified. Three managed Modal fixtures returned 7, 71 and 620 detections with
exact saved box/score parity; the exact published source revision also passed
the sparse fixture and retained its preview. These checks establish serving
parity, not fresh scientific qualification. See [release status](docs/release-status.md).

## Artifacts and runtime

Large model weights, upstream source archives and dataset images are not Git
source files. The immutable Hub counter package includes the required weights
and source assets. A raw Git checkout needs these exact retained assets before
inference; see [artifact requirements](docs/artifacts.md). Do not replace them
with unrelated weights or weaken the source/runtime guards.

Inference is fixed to the saved YOLOX-Tiny EMA checkpoint and pretest settings.
The scientific runtime is Python 3.12.3, Biosimulant 0.0.34, Torch 2.6.0+cu124,
torchvision 0.21.0+cu124, NumPy 1.26.4, CUDA 12.4 and a Modal T4. The BioModule
uses `ExecutionPolicy.ONCE_BEFORE_RUN`; visualization reads the computed result
and does not run the detector again.

## Development

Development, fixes and checks use `staging`. This newly created repository uses
`staging` as its default branch; no changes are pushed to `main`. No publication
or deployment happens merely by moving these files or pushing this repository.
Managed scientific execution, artifacts, publication and hosting continue
through Biosimulant MCP.

For software fixtures in a separate development environment:

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_repository.py
python -m pytest -q
```

These fixtures use synthetic inputs and stub inference. They do not load a
trained checkpoint, train a model or establish detector accuracy. An optional
source-import fixture requires `CFU_VERIFIED_YOLOX_SOURCE` to point to its exact
checksum-pinned source ZIP; otherwise it is skipped.

## Provenance and license

Extracted from backend commit `cd1b8312b2db5c29ec494d3f063bd8ac2ebfbcc2` after
synchronizing the staging worktree from current main. See
[`docs/migration-manifest.json`](docs/migration-manifest.json) for every source
path, destination and original/current byte identity. Model implementation and
the frozen inference contract are unchanged. Platform services and their tests
remain in the backend, along with private operational audit records.

Authored adapter code: MIT. YOLOX source: Apache-2.0. SAHI: MIT. ADBC dataset
evidence: CC BY 4.0. See [third-party notices](THIRD_PARTY_NOTICES.md).
