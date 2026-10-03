"""Synthetic identity and transport fixtures; not ADBC scientific evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/split/src/split.py"
spec = importlib.util.spec_from_file_location("cfu_split", SOURCE)
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def fixture_documents(count=30):
    images = []
    for i in range(count):
        name = f"fixture-{i:03d}.jpg"
        content = f"synthetic metadata fixture {i}".encode()
        images.append({"file_id": f"fixture-id-{i}", "sample_id": name,
                       "group_id": f"source-group-{i}", "role": "image",
                       "split": "unassigned", "sha256": hashlib.sha256(content).hexdigest(),
                       "size_bytes": len(content)})
    annotation = {"file_id": "fixture-annotation", "sample_id": "labels.json",
                  "group_id": "annotation", "role": "annotation", "split": "unassigned",
                  "sha256": hashlib.sha256(b"deliberately unparsed labels").hexdigest(),
                  "size_bytes": len(b"deliberately unparsed labels")}
    manifest = {"schema_version": 1, "name": "synthetic fixtures",
                "source": "software fixtures, not a real plate dataset", "license": "MIT",
                "attribution": "software test", "annotation_version": "fixture",
                "members": images + [annotation], "parent_dataset_id": None,
                "transformation_run_id": None}
    review = {"stage": "duplicate_review", "images": [{"file_name": m["sample_id"],
               "sha256": m["sha256"], "size_bytes": m["size_bytes"]} for m in images],
               "annotation_identities": [{"sha256": annotation["sha256"],
                                           "size_bytes": annotation["size_bytes"]}],
               "candidates": [], "candidate_count": 0}
    config = {"manifest": manifest, "dataset_sha256": split.canonical_digest(manifest),
              "dataset_id": "synthetic-fixture-dataset", "review_run_id": "synthetic-review",
              "seed": 20261001, "grouping_decision": "retain all candidate edges"}
    return config, review


def seal(config, review):
    config["dataset_sha256"] = split.canonical_digest(config["manifest"])
    review["candidate_count"] = len(review["candidates"])
    raw = json.dumps(review).encode()
    config["review_sha256"] = hashlib.sha256(raw).hexdigest()
    config["review_size_bytes"] = len(raw)
    return raw


def test_all_related_views_stay_together_and_every_original_is_preserved():
    config, review = fixture_documents()
    members = config["manifest"]["members"]
    review["candidates"] = [{"left": members[0]["sample_id"], "right": members[1]["sample_id"]},
                            {"left": members[1]["sample_id"], "right": members[2]["sample_id"]}]
    members[5]["sha256"] = members[4]["sha256"]
    members[5]["size_bytes"] = members[4]["size_bytes"]
    review["images"][5]["sha256"] = members[4]["sha256"]
    review["images"][5]["size_bytes"] = members[4]["size_bytes"]
    members[8]["group_id"] = members[7]["group_id"]
    report = split.freeze_split(config, seal(config, review))
    output = report["native_dataset_manifest"]["members"]
    by_name = {m["sample_id"]: m for m in output}
    for indices in [(0, 1, 2), (4, 5), (7, 8)]:
        assert len({by_name[members[i]["sample_id"]]["group_id"] for i in indices}) == 1
        assert len({by_name[members[i]["sample_id"]]["split"] for i in indices}) == 1
    assert {m["file_id"] for m in output} == {m["file_id"] for m in members}
    assert report["image_counts"] == {"train": 24, "validation": 3, "test": 3}
    assert report["native_dataset_manifest_sha256"] == split.canonical_digest(report["native_dataset_manifest"])
    assert by_name["labels.json"]["split"] == "unassigned"
    assert report["annotation_outcomes_opened"] is False
    assert report["training_executed"] is False


def test_allocation_replays_and_does_not_depend_on_input_order():
    config, review = fixture_documents()
    first = split.freeze_split(config, seal(config, review))
    assert first == split.freeze_split(config, seal(config, review))
    config["manifest"]["members"].reverse()
    review["images"].reverse()
    second = split.freeze_split(config, seal(config, review))
    assert first["native_dataset_manifest"]["members"] == second["native_dataset_manifest"]["members"]
    assert first["groups"] == second["groups"]


@pytest.mark.parametrize("tamper", ["bytes", "source_manifest", "missing_image", "extra_image",
                                   "image_identity", "annotation_identity", "unknown_pair", "already_frozen"])
def test_missing_and_changed_identities_fail_closed(tamper):
    config, review = fixture_documents()
    raw = seal(config, review)
    if tamper == "bytes":
        raw += b" "
    elif tamper == "source_manifest":
        config["manifest"]["name"] = "tampered"
    else:
        if tamper == "missing_image":
            review["images"].pop()
        elif tamper == "extra_image":
            review["images"].append(dict(review["images"][0]))
        elif tamper == "image_identity":
            review["images"][0]["sha256"] = "f" * 64
        elif tamper == "annotation_identity":
            review["annotation_identities"] = []
        elif tamper == "unknown_pair":
            review["candidates"] = [{"left": "unknown.jpg", "right": "fixture-000.jpg"}]
        else:
            config["manifest"]["members"][0]["split"] = "test"
        raw = seal(config, review)
    with pytest.raises(ValueError):
        split.freeze_split(config, raw)


def test_a_group_that_cannot_fit_is_not_split_to_force_acceptance():
    config, review = fixture_documents(count=10)
    for member in config["manifest"]["members"][:9]:
        member["group_id"] = "one-related-group"
    with pytest.raises(ValueError, match="cannot fit"):
        split.freeze_split(config, seal(config, review))


def test_freeze_task_consumes_a_connected_signal_and_emits_a_retained_manifest(tmp_path, monkeypatch):
    from biosimulant import BioModule, BioSignal, BioWorld, ExecutionPolicy

    config, review = fixture_documents()
    raw = seal(config, review)
    review_path = tmp_path / "review.json"
    review_path.write_bytes(raw)
    model = tmp_path / "model"
    (model / "src").mkdir(parents=True)
    (model / "source-dataset.json").write_text(json.dumps(config))
    monkeypatch.setattr(split, "__file__", str(model / "src/split.py"))
    monkeypatch.chdir(tmp_path)
    component = split.FreezeADBCSplit()

    class Source(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

        def outputs(self):
            return component.inputs()

        def execute(self, inputs, *, context):
            return {"review_report": str(review_path)}

    world = BioWorld(communication_step=0.01)
    world.add_biomodule("source", Source())
    world.add_biomodule("split", component)
    world.connect("source.review_report", "split.review_report")
    world.run(duration=0.01)
    signal = world.get_outputs("split")["split_manifest_path"]
    assert isinstance(signal, BioSignal) and signal.source == "split"
    output = Path(signal.value)
    report = json.loads(output.read_text())
    assert Counter(m["split"] for m in report["native_dataset_manifest"]["members"] if m["role"] == "image") == {"train": 24, "validation": 3, "test": 3}
    assert report["original_member_count"] == 31
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("value", [None, "", {"kind": "stored_file", "file_id": "unresolved"}])
def test_unresolved_inputs_fail_before_creating_outputs(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must resolve"):
        split.FreezeADBCSplit().execute({"review_report": value}, context=None)
    assert not list(tmp_path.glob("cfu-freeze-split-*"))
