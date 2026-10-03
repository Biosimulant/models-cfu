"""Preparation software fixtures; no observed source-label or accuracy results."""

import hashlib
import importlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "models/tiles/src"
spec = importlib.util.spec_from_file_location("cfu_preparation_fixture", SRC / "__init__.py", submodule_search_locations=[str(SRC)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
prepare = importlib.import_module("cfu_preparation_fixture.prepare")


def pin(path, raw):
    path.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def fixture(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (900, 700), "white").save(stream, format="JPEG")
    raw = stream.getvalue()
    image_pin = pin(tmp_path / "fixture.jpg", raw)
    originals = [{**image_pin, "file_id": "synthetic-train", "sample_id": "fixture.jpg", "group_id": "synthetic-group",
                  "role": "image", "split": "train"},
                 {**image_pin, "file_id": "synthetic-test", "sample_id": "held-out.jpg", "group_id": "synthetic-held-out",
                  "role": "image", "split": "test"}]
    manifest = {"members": originals, "source": "synthetic software fixture"}
    frozen = {"native_dataset_manifest": manifest}
    split_pin = pin(tmp_path / "split.json", json.dumps(frozen).encode())
    annotations = {"images": [{"id": 1, "file_name": "fixture.jpg", "width": 900, "height": 700},
                              {"id": 2, "file_name": "held-out.jpg", "width": 900, "height": 700}],
                   "categories": [{"id": 7, "name": "synthetic colony label"}],
                   "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [625, 300, 30, 20]},
                                   {"id": 2, "image_id": 2, "category_id": 999, "bbox": [0, 0, -1, 20], "iscrowd": 1}]}
    ann_pin = pin(tmp_path / "labels.json", json.dumps(annotations).encode())
    config = {"batches": [["fixture.jpg"]], "frozen_split": split_pin, "annotations": ann_pin,
              "frozen_dataset_id": "synthetic-dataset", "frozen_dataset_sha256": hashlib.sha256(
                  json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    inputs = {"frozen_split": str(tmp_path / "split.json"), "annotations": str(tmp_path / "labels.json"),
              "image_00": str(tmp_path / "fixture.jpg")}
    root = tmp_path / "prepared"
    root.mkdir(mode=0o700)
    return config, inputs, root


def test_native_archive_contains_verified_crops_and_parent_boxes_only(tmp_path):
    config, inputs, root = fixture(tmp_path)
    original_annotations = Path(inputs["annotations"]).read_bytes()
    archive, manifest = prepare.prepare_batch(config, 0, inputs, root)
    report = json.loads(manifest.read_text())
    assert report["consumed_original_file_ids"] == ["synthetic-train"]
    assert report["derived_image_count"] == 4
    assert report["test_targets_inspected_or_emitted"] is False
    assert report["plates"][0]["source_annotation_ids"] == [1]
    assert Path(inputs["annotations"]).read_bytes() == original_annotations
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == 5
        for tile in report["plates"][0]["tiles"]:
            raw = z.read("tiles/" + tile["file_name"])
            assert len(raw) == tile["size_bytes"] and hashlib.sha256(raw).hexdigest() == tile["sha256"]
    assert archive.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("tamper", ["split_bytes", "annotation_bytes", "image_bytes", "held_out", "unused_input", "missing_input", "manifest_identity"])
def test_changed_or_unconsumed_sources_fail_before_transforming(tmp_path, tamper):
    config, inputs, root = fixture(tmp_path)
    if tamper == "split_bytes":
        Path(inputs["frozen_split"]).write_bytes(b"changed")
    elif tamper == "annotation_bytes":
        Path(inputs["annotations"]).write_bytes(b"changed")
    elif tamper == "image_bytes":
        Path(inputs["image_00"]).write_bytes(b"changed")
    elif tamper == "held_out":
        config["batches"] = [["held-out.jpg"]]
    elif tamper == "unused_input":
        inputs["image_01"] = inputs["image_00"]
    elif tamper == "missing_input":
        inputs.pop("image_00")
    else:
        config["frozen_dataset_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        prepare.prepare_batch(config, 0, inputs, root)
    assert not list(root.iterdir())


def test_connected_typed_inputs_allow_absent_unused_slots(tmp_path, monkeypatch):
    from biosimulant import BioModule, BioWorld, ExecutionPolicy

    config, inputs, root = fixture(tmp_path)
    model = tmp_path / "model"
    (model / "src").mkdir(parents=True)
    (model / "preparation-plan.json").write_text(json.dumps(config))
    monkeypatch.setattr(prepare, "__file__", str(model / "src/prepare.py"))
    monkeypatch.chdir(tmp_path)
    component = prepare.PrepareADBCTiles()
    file_inputs = dict(inputs)

    class Source(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

        def outputs(self):
            return {k: component.inputs()[k] for k in inputs}

        def execute(self, inputs, *, context):
            return file_inputs

    world = BioWorld(communication_step=0.01)
    world.add_biomodule("source", Source())
    world.add_biomodule("prepare", component)
    for name in inputs:
        world.connect("source." + name, "prepare." + name)
    world.run(duration=0.01)
    report = json.loads(Path(world.get_outputs("prepare")["prepared_manifest_path"].value).read_text())
    assert report["derived_image_count"] == 4
    assert report["effective_environment"]["biosimulant"] == "0.0.34"


def test_fixed_output_budget_does_not_silently_exclude_a_plate(tmp_path, monkeypatch):
    config, inputs, root = fixture(tmp_path)
    monkeypatch.setattr(prepare, "MAX_DERIVED_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        prepare.prepare_batch(config, 0, inputs, root)
    assert not (root / "prepared-tiles.zip").exists()


def test_partial_parent_group_fails_before_any_transformation(tmp_path):
    config, inputs, root = fixture(tmp_path)
    frozen = json.loads(Path(inputs["frozen_split"]).read_text())
    original = frozen["native_dataset_manifest"]["members"][0]
    frozen["native_dataset_manifest"]["members"].append({**original, "sample_id": "related.jpg", "file_id": "related-original"})
    config["frozen_split"] = pin(Path(inputs["frozen_split"]), json.dumps(frozen).encode())
    config["frozen_dataset_sha256"] = hashlib.sha256(json.dumps(frozen["native_dataset_manifest"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="every original"):
        prepare.prepare_batch(config, 0, inputs, root)
    assert not list(root.iterdir())
