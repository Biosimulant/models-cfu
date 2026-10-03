"""Freeze whole-original membership without reading colony annotation outcomes.

Every byte-duplicate, supplied source group and review candidate component stays
together. Groups are screening constraints, not asserted acquisition batches.
This task registers no dataset itself; its declared output is the proposal to be
verified and registered through the authenticated Biosimulant MCP afterward.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

MAX_DOCUMENT_BYTES = 512 * 1024
METHOD = "candidate-connected-groups-hash-allocation-v1"


def canonical_digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_split(config: dict, review_bytes: bytes) -> dict:
    if len(review_bytes) != config["review_size_bytes"] or hashlib.sha256(
        review_bytes
    ).hexdigest() != config["review_sha256"]:
        raise ValueError("Duplicate-review artifact identity differs from the approved Run")
    manifest = config["manifest"]
    if canonical_digest(manifest) != config["dataset_sha256"]:
        raise ValueError("Original dataset manifest differs from its native registration")
    members = manifest["members"]
    if any(m["split"] != "unassigned" for m in members):
        raise ValueError("Split freezing requires an entirely unassigned original registration")
    images = [m for m in members if m["role"] == "image"]
    if not images:
        raise ValueError("No original images are available")
    by_name = {m["sample_id"]: m for m in images}
    if len(by_name) != len(images) or len({m["file_id"] for m in members}) != len(members):
        raise ValueError("Repeated original sample or file identity")
    review = json.loads(review_bytes)
    if review.get("stage") != "duplicate_review" or review.get("candidate_count") != len(
        review.get("candidates", [])
    ):
        raise ValueError("Unexpected duplicate-review contract")
    reviewed = {m["file_name"]: m for m in review["images"]}
    if len(review["images"]) != len(images) or set(reviewed) != set(by_name):
        raise ValueError("Review does not cover every original image exactly once")
    for name, original in by_name.items():
        for key in ("sha256", "size_bytes"):
            if reviewed[name].get(key) != original[key]:
                raise ValueError("Reviewed image identity differs from retained originals")
    annotations = {m["sha256"]: m["size_bytes"] for m in members if m["role"] == "annotation"}
    reviewed_annotations = {m["sha256"]: m["size_bytes"] for m in review["annotation_identities"]}
    if annotations != reviewed_annotations:
        raise ValueError("Reviewed annotation identity differs from retained originals")

    parents = {name: name for name in by_name}

    def root(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def connect(left, right):
        if left not in parents or right not in parents:
            raise ValueError("Review candidate names an unknown original")
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    # Preserve provided groups and exact byte duplicates before perceptual edges.
    for key in ("group_id", "sha256"):
        first = {}
        for name in sorted(by_name):
            identity = by_name[name][key]
            if identity in first:
                connect(first[identity], name)
            else:
                first[identity] = name
    for pair in review["candidates"]:
        connect(pair["left"], pair["right"])
    groups = {}
    for name in sorted(by_name):
        groups.setdefault(root(name), []).append(name)
    grouped = []
    for names in groups.values():
        identity = "\n".join(f"{n}:{by_name[n]['sha256']}" for n in names)
        group_id = "screening:" + hashlib.sha256(identity.encode()).hexdigest()
        rank = hashlib.sha256(f"{METHOD}:{config['seed']}:{group_id}".encode()).hexdigest()
        grouped.append({"group_id": group_id, "samples": names, "rank": rank})

    count = len(images)
    targets = {"train": (8 * count + 5) // 10, "validation": (count + 5) // 10}
    targets["test"] = count - sum(targets.values())
    remaining = dict(targets)
    assigned = {}
    for group in sorted(grouped, key=lambda g: (-len(g["samples"]), g["rank"])):
        # The rank chooses a prespecified 80/10/10 bucket independently of labels.
        selector = int(group["rank"][:16], 16) / 2**64
        preferred = "train" if selector < 0.8 else "validation" if selector < 0.9 else "test"
        fallback = sorted(remaining, key=lambda s: (-remaining[s], s))
        choices = [preferred] + [s for s in fallback if s != preferred]
        split = next((s for s in choices if remaining[s] >= len(group["samples"])), None)
        if split is None:
            raise ValueError("A related-image group cannot fit the prespecified split sizes")
        remaining[split] -= len(group["samples"])
        group["split"] = split
        for name in group["samples"]:
            assigned[name] = (group["group_id"], split)
    if any(remaining.values()):
        raise ValueError("The prespecified split sizes were not attained")

    frozen = []
    for member in members:
        item = dict(member)
        if item["role"] == "image":
            item["group_id"], item["split"] = assigned[item["sample_id"]]
        frozen.append(item)
    frozen.sort(key=lambda m: (m["role"], m["sample_id"], m["file_id"]))
    native_manifest = {
        **{k: manifest[k] for k in ("schema_version", "license", "attribution", "annotation_version")},
        "name": "ADBC v3 — frozen original photographs; conservative candidate groups",
        "source": manifest["source"] + "; original_registration=" + config["dataset_id"],
        "members": frozen,
        "parent_dataset_id": None,
        "transformation_run_id": None,
    }
    return {
        "schema_version": 1,
        "stage": "original_split_frozen",
        "method": METHOD,
        "seed": config["seed"],
        "original_dataset_id": config["dataset_id"],
        "original_dataset_sha256": config["dataset_sha256"],
        "review_run_id": config["review_run_id"],
        "review_sha256": config["review_sha256"],
        "groups": sorted(grouped, key=lambda g: g["group_id"]),
        "image_counts": dict(Counter(m["split"] for m in frozen if m["role"] == "image")),
        "target_image_counts": targets,
        "original_member_count": len(frozen),
        "candidate_count": review["candidate_count"],
        "grouping_decision": config["grouping_decision"],
        "native_dataset_manifest": native_manifest,
        "native_dataset_manifest_sha256": canonical_digest(native_manifest),
        "annotation_outcomes_opened": False,
        "training_executed": False,
        "registration_semantics": (
            "New immutable original-file partition registration, not a pixel transformation. "
            "Native parent_dataset_id is reserved for derived files preserving existing splits. "
            "Original registration and managed review/freezing artifacts supply explicit provenance."
        ),
        "limits": (
            "Perceptual grouping may miss relationships. Acquisition independence is unproved; "
            "visual screening is not independent human scientific review. No originals excluded."
        ),
    }


class FreezeADBCSplit(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {"review_report": SignalSpec.scalar(dtype="str", value_type="file", format="json")}

    def outputs(self):
        return {"split_manifest_path": SignalSpec.scalar(dtype="str", value_type="file", format="json")}

    def execute(self, inputs, *, context):
        value = inputs.get("review_report")
        value = value.value if isinstance(value, BioSignal) else value
        if not isinstance(value, str) or not value:
            raise ValueError("review_report must resolve to a retained file path")
        path = Path(value)
        if not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Review report is missing or exceeds its read bound")
        config_path = Path(__file__).resolve().parents[1] / "source-dataset.json"
        if config_path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Source registration snapshot exceeds its read bound")
        config = json.loads(config_path.read_text())
        report = freeze_split(config, path.read_bytes())
        scratch = Path(tempfile.mkdtemp(prefix="cfu-freeze-split-", dir=Path.cwd()))
        scratch.chmod(0o700)
        output = scratch / "frozen-original-split.json"
        output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        output.chmod(0o600)
        return {"split_manifest_path": str(output)}
