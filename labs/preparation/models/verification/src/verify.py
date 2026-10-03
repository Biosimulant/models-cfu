"""Read retained archives directly and emit bounded managed verification receipts."""

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .core import verify

SCHEMA = {"stage": "str", "batches": "json", "artifact_bytes_verified": "bool",
          "test_outcomes_inspected": "bool", "training_executed": "bool", "scientific_acceptance_established": "bool"}


def pinned(path, metadata):
    value = path.value if isinstance(path, BioSignal) else path
    if not isinstance(value, str) or not value:
        raise ValueError("Verification requires an authorized retained File path")
    p = Path(value)
    if p.stat().st_size != metadata["size_bytes"]:
        raise ValueError("Retained verification input length changed")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("Retained verification input digest changed")
    return raw


def check(inputs, config, batch, root):
    if not __debug__:
        raise RuntimeError("Archive verification requires enabled runtime assertions")
    if isinstance(batch, bool) or not isinstance(batch, int) or not 0 <= batch < len(config["verification_batches"]):
        raise ValueError("Verification batch is outside the immutable plan")
    (root / "verified-frozen-original-split.json").write_bytes(pinned(inputs.get("frozen_split"), config["frozen_split"]))
    (root / "verified-original-annotations.json").write_bytes(pinned(inputs.get("annotations"), config["annotations"]))
    selected = config["verification_batches"][batch]
    if len(selected) > 4 or sum(config["archives"][str(i)]["size_bytes"] for i in selected) > 320 * 1024 * 1024:
        raise ValueError("Verification archive inputs exceed the fixed batch bound")
    envelopes = []
    for slot, original_batch in enumerate(selected):
        meta = config["archives"][str(original_batch)]
        value = inputs.get(f"archive_{slot:02d}")
        value = value.value if isinstance(value, BioSignal) else value
        if not isinstance(value, str):
            raise ValueError("Missing retained crop archive")
        envelopes.append({"run_id": meta["run_id"], "batch_index": original_batch, "artifacts": {
            "tiles_archive_path": {"path": value, "sha256": meta["sha256"], "size_bytes": meta["size_bytes"]},
            "prepared_manifest_path": {"path": value, "member": "prepared-batch.json", **meta["manifest"]}}})
    for slot in range(len(selected), 4):
        value = inputs.get(f"archive_{slot:02d}")
        value = value.value if isinstance(value, BioSignal) else value
        if value not in (None, ""):
            raise ValueError("Unused verification archive input is forbidden")

    def reader(meta):
        if "member" not in meta:
            return pinned(meta["path"], meta)
        with zipfile.ZipFile(meta["path"]) as z:
            info = z.getinfo(meta["member"])
            if info.file_size != meta["size_bytes"]:
                raise ValueError("Pinned archive manifest length changed")
            raw = z.read(meta["member"])
        if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError("Pinned archive manifest digest changed")
        return raw

    summaries = verify(envelopes, root, config, reader=reader)
    index = []
    for original_batch in selected:
        tiles = json.loads((root / f"verified-prepared-tile-identities-{original_batch:02d}.json").read_text())
        index.extend({"preparation_batch": original_batch, **{k: t[k] for k in
                     ["file_name", "sha256", "size_bytes", "sample_id", "group_id", "split", "parent_file_id", "source_window_xyxy"]}} for t in tiles)
    output = root / "verified-tile-hash-index.json"
    output.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n")
    output.chmod(0o600)
    return {"stage": "independent_prepared_archive_verification", "batches": summaries,
            "artifact_bytes_verified": True, "test_outcomes_inspected": False, "training_executed": False,
            "scientific_acceptance_established": False}, output


class VerifyPreparedArchives(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, verification_batch=0):
        self.verification_batch = verification_batch

    def inputs(self):
        values = {"frozen_split": SignalSpec.scalar(dtype="str", value_type="file", format="json"),
                  "annotations": SignalSpec.scalar(dtype="str", value_type="file", format="json")}
        values.update({f"archive_{i:02d}": SignalSpec.scalar(dtype="str", value_type="file", format="zip", required=False)
                       for i in range(4)})
        return values

    def outputs(self):
        return {"receipt": SignalSpec.record(schema=SCHEMA, emitted_unit="1"),
                "tile_hash_index_path": SignalSpec.scalar(dtype="str", value_type="file", format="json")}

    def execute(self, inputs, *, context):
        config = json.loads((Path(__file__).resolve().parent.parent / "verification-plan.json").read_text())
        root = Path(tempfile.mkdtemp(prefix="cfu-verification-", dir=Path.cwd()))
        root.chmod(0o700)
        receipt, path = check(inputs, config, self.verification_batch, root)
        return {"receipt": receipt, "tile_hash_index_path": str(path)}
