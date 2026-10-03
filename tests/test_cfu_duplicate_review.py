"""Algorithm/transport fixtures, not evidence about the ADBC dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "labs/preparation/models/review/src/review.py"
)
spec = importlib.util.spec_from_file_location("cfu_duplicate_review", SOURCE)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fixture_archive(tmp_path):
    image = Image.new("RGB", (96, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 10, 37, 62), fill="black")
    draw.ellipse((45, 12, 80, 47), fill="gray")
    contents = {}
    for name, quality in [("first.jpg", 95), ("recompressed.jpg", 90)]:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        contents[name] = buffer.getvalue()
    # An intentionally unparseable annotation fixture proves this stage reads identities only.
    contents["annotations.json"] = b"unopened annotation outcomes"
    members = [
        {
            "file_name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "publisher_md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "size_bytes": len(content),
            "split": "train" if i == 0 else "test",
        }
        for i, (name, content) in enumerate(contents.items())
    ]
    inventory = {"source_snapshot_sha256": "a" * 64, "batches": [members]}
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
        archive.writestr(
            "batch-manifest.json",
            json.dumps(
                {
                    "batch_index": 0,
                    "members": members,
                    "source_snapshot_sha256": inventory["source_snapshot_sha256"],
                }
            ),
        )
    return inventory, path


def test_recompressed_duplicates_crossing_splits_require_review(tmp_path):
    inventory, path = fixture_archive(tmp_path)
    result = review.review_archives(inventory, [path])
    assert len(result["images"]) == 2
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["byte_identical"] is False
    assert result["candidates"][0]["crosses_preliminary_split"] is True
    assert result["status"] == "manual_review_required"
    assert result["training_authorized"] is False
    assert len(result["annotation_identities"]) == 1
    assert "unopened annotation outcomes" not in json.dumps(result)


@pytest.mark.parametrize(
    "tamper", ["hash", "missing_batch", "extra_entry", "duplicate_entry", "manifest"]
)
def test_original_and_archive_identity_tampering_fails_closed(tmp_path, tamper):
    inventory, path = fixture_archive(tmp_path)
    if tamper == "hash":
        inventory["batches"][0][0]["sha256"] = "b" * 64
    elif tamper == "missing_batch":
        with pytest.raises(ValueError, match="Every retained"):
            review.review_archives(inventory, [])
        return
    elif tamper in {"extra_entry", "duplicate_entry"}:
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("unexpected.txt" if tamper == "extra_entry" else "first.jpg", b"bad")
    else:
        inventory["source_snapshot_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        review.review_archives(inventory, [path])


def test_oversized_decoded_original_is_rejected(monkeypatch):
    monkeypatch.setattr(review, "MAX_SOURCE_IMAGE_PIXELS", 10)
    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), "white").save(buffer, format="JPEG")
    with pytest.raises(ValueError, match="outside"):
        review.image_identity(buffer.getvalue())


def test_largest_retained_source_dimensions_can_be_reviewed():
    # Synthetic software fixture using verified source dimensions, not CFU evidence.
    buffer = io.BytesIO()
    Image.new("L", (5927, 5968), "white").save(buffer, format="JPEG")
    payload = buffer.getvalue()
    identity = review.image_identity(payload)
    assert (identity["oriented_width"], identity["oriented_height"]) == (5927, 5968)
    assert len(identity["dhash128"]) == 32 and len(identity["ahash256"]) == 64



def test_above_source_bound_is_rejected():
    buffer = io.BytesIO()
    Image.new("L", (6500, 6500), "white").save(buffer, format="JPEG")
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ValueError, match="dimensions=6500x6500"):
            review.image_identity(buffer.getvalue())


def test_rejected_original_names_the_file_and_dimensions(tmp_path, monkeypatch):
    inventory, path = fixture_archive(tmp_path)
    monkeypatch.setattr(review, "MAX_SOURCE_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="first.jpg:.*dimensions=96x80"):
        review.review_archives(inventory, [path])


def fixture_runtime_archives(tmp_path, monkeypatch):
    inventory, first = fixture_archive(tmp_path)
    paths = [first]
    for index in range(1, 16):
        inventory["batches"].append([])
        path = tmp_path / f"batch-{index}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "batch-manifest.json",
                json.dumps(
                    {
                        "batch_index": index,
                        "members": [],
                        "source_snapshot_sha256": inventory["source_snapshot_sha256"],
                    }
                ),
            )
        paths.append(path)
    model = tmp_path / "model"
    (model / "src").mkdir(parents=True)
    (model / "inventory.json").write_text(json.dumps(inventory))
    monkeypatch.setattr(review, "__file__", str(model / "src" / "review.py"))
    monkeypatch.chdir(tmp_path)
    return paths


def test_review_executes_from_connected_file_signals_and_commits_typed_output(
    tmp_path, monkeypatch
):
    from biosimulant import BioModule, BioSignal, BioWorld, ExecutionPolicy

    paths = fixture_runtime_archives(tmp_path, monkeypatch)
    component = review.ReviewADBC()

    class RetainedArchives(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

        def outputs(self):
            return component.inputs()

        def execute(self, inputs, *, context):
            return {f"archive_{index}": str(path) for index, path in enumerate(paths)}

    world = BioWorld(communication_step=0.01)
    world.add_biomodule("retained", RetainedArchives())
    world.add_biomodule("review", component)
    for index in range(16):
        world.connect(f"retained.archive_{index}", f"review.archive_{index}")
    world.run(duration=0.03)
    outputs = world.get_outputs("review")
    assert set(outputs) == {"review_path"}
    result = outputs["review_path"]
    assert isinstance(result, BioSignal) and result.source == "review"
    assert result.emitted_at == 0.0
    path = Path(result.value)
    report = json.loads(path.read_text())
    assert len(report["images"]) == 2 and report["candidate_count"] == 1
    assert len(report["annotation_identities"]) == 1
    assert report["training_authorized"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert len(list(tmp_path.glob("cfu-duplicate-review-*"))) == 1
    world.run(duration=0.01)
    replay_path = Path(world.get_outputs("review")["review_path"].value)
    assert json.loads(replay_path.read_text()) == report
    assert len(list(tmp_path.glob("cfu-duplicate-review-*"))) == 2


def test_review_accepts_resolved_initial_file_paths(tmp_path, monkeypatch):
    from biosimulant import ExecutionContext, ExecutionPolicy

    paths = fixture_runtime_archives(tmp_path, monkeypatch)
    component = review.ReviewADBC()
    outputs = component.execute(
        {f"archive_{index}": str(path) for index, path in enumerate(paths)},
        context=ExecutionContext(policy=ExecutionPolicy.ONCE_BEFORE_RUN, run_start=0, run_end=0.01),
    )
    assert json.loads(Path(outputs["review_path"]).read_text())["candidate_count"] == 1


@pytest.mark.parametrize("invalid", [None, "", {"kind": "stored_file", "file_id": "unresolved"}])
def test_review_rejects_unresolved_inputs_before_review_output(tmp_path, monkeypatch, invalid):
    paths = fixture_runtime_archives(tmp_path, monkeypatch)
    inputs = {f"archive_{index}": str(path) for index, path in enumerate(paths)}
    inputs["archive_15"] = invalid
    with pytest.raises(ValueError, match="archive_15 must resolve"):
        review.ReviewADBC().execute(inputs, context=None)
    assert not list(tmp_path.glob("cfu-duplicate-review-*"))
