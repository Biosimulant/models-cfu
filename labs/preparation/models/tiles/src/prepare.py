"""Finite retained-original training/validation preparation, never held-out evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import tempfile
import zipfile
from pathlib import Path

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .geometry import index_coco, prepare_plate

MAX_BATCH = 16
MAX_DERIVED_BYTES = 128 * 1024 * 1024


def read_pinned(value, pin, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must resolve to an authorized retained file path")
    path = Path(value)
    if not path.is_file() or path.stat().st_size != pin["size_bytes"]:
        raise ValueError(f"{label}: retained file length changed")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError(f"{label}: retained file digest changed")
    return raw


def prepare_batch(config, batch_index, inputs, root):
    inputs = {k: v.value if isinstance(v, BioSignal) else v for k, v in inputs.items()}
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or not 0 <= batch_index < len(config["batches"]):
        raise ValueError("Preparation batch index is outside the immutable plan")
    names = config["batches"][batch_index]
    if not 1 <= len(names) <= MAX_BATCH or len(names) != len(set(names)):
        raise ValueError("Preparation batch must contain 1–16 distinct originals")
    frozen_bytes = read_pinned(inputs.get("frozen_split"), config["frozen_split"], "Frozen split")
    frozen = json.loads(frozen_bytes)
    manifest = frozen["native_dataset_manifest"]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != config["frozen_dataset_sha256"]:
        raise ValueError("Frozen dataset differs from its native registration")
    originals = {m["sample_id"]: m for m in manifest["members"] if m["role"] == "image"}
    if any(n not in originals or originals[n]["split"] not in {"train", "validation"} for n in names):
        raise ValueError("Preparation cannot consume held-out test targets")
    selected_groups = {originals[n]["group_id"] for n in names}
    if any(n not in names for n, m in originals.items() if m["group_id"] in selected_groups):
        raise ValueError("Transformation must consume every original in its selected parent groups")
    for i in range(len(names), MAX_BATCH):
        if inputs.get(f"image_{i:02d}") not in (None, ""):
            raise ValueError("Unused image input is not allowed in this batch")
    annotation_bytes = read_pinned(inputs.get("annotations"), config["annotations"], "Original annotations")
    document = json.loads(annotation_bytes)
    images, annotations = index_coco(document, set(originals), selected_names=set(names))
    # Verify every consumed original before writing any transformed data.
    contents = [read_pinned(inputs.get(f"image_{i:02d}"), originals[n], n) for i, n in enumerate(names)]
    plates, total = [], 0
    for name, content in zip(names, contents):
        plate = prepare_plate(content, originals[name], images[name], annotations[name], root / "tiles")
        total += sum(t["size_bytes"] for t in plate["tiles"])
        if total > MAX_DERIVED_BYTES:
            raise ValueError("Derived batch exceeds its fixed byte budget; no plate is silently excluded")
        plates.append(plate)
    report = {
        "schema_version": 1, "stage": "training_validation_tile_preparation",
        "batch_index": batch_index, "frozen_dataset_id": config["frozen_dataset_id"],
        "frozen_dataset_sha256": digest, "frozen_split_input": config["frozen_split"],
        "original_annotation_input": config["annotations"], "source_categories": document["categories"],
        "consumed_original_file_ids": [originals[n]["file_id"] for n in names], "plates": plates,
        "derived_image_bytes": total, "derived_image_count": sum(len(p["tiles"]) for p in plates),
        "test_targets_inspected_or_emitted": False, "source_annotations_modified": False,
        "training_executed": False, "scientific_acceptance_established": False,
        "effective_environment": {"python": platform.python_version(),
                                  "biosimulant": importlib.metadata.version("biosimulant"),
                                  "Pillow": importlib.metadata.version("Pillow")},
    }
    manifest_path = root / "prepared-batch.json"
    manifest_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    manifest_path.chmod(0o600)
    if manifest_path.stat().st_size + total > MAX_DERIVED_BYTES:
        raise ValueError("Derived manifest and images exceed their fixed byte budget")
    archive_path = root / "prepared-tiles.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(manifest_path, "prepared-batch.json")
        for plate in plates:
            for tile in plate["tiles"]:
                archive.write(root / "tiles" / tile["file_name"], "tiles/" + tile["file_name"])
    archive_path.chmod(0o600)
    return archive_path, manifest_path


class PrepareADBCTiles(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, batch_index=0):
        self.batch_index = batch_index

    def inputs(self):
        signals = {"frozen_split": SignalSpec.scalar(dtype="str", value_type="file", format="json"),
                   "annotations": SignalSpec.scalar(dtype="str", value_type="file", format="json")}
        signals.update({f"image_{i:02d}": SignalSpec.scalar(dtype="str", value_type="file", format="jpg", required=False)
                        for i in range(MAX_BATCH)})
        return signals

    def outputs(self):
        return {"tiles_archive_path": SignalSpec.scalar(dtype="str", value_type="file", format="zip"),
                "prepared_manifest_path": SignalSpec.scalar(dtype="str", value_type="file", format="json")}

    def execute(self, inputs, *, context):
        config = json.loads((Path(__file__).resolve().parent.parent / "preparation-plan.json").read_text())
        root = Path(tempfile.mkdtemp(prefix="cfu-preparation-", dir=Path.cwd()))
        root.chmod(0o700)
        archive, manifest = prepare_batch(config, self.batch_index, inputs, root)
        return {"tiles_archive_path": str(archive), "prepared_manifest_path": str(manifest)}
