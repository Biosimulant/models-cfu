"""Synthetic image/box geometry checks; no real plate accuracy evidence."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "models/tiles/src/geometry.py"
spec = importlib.util.spec_from_file_location("cfu_tile_geometry", SOURCE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def fixture():
    stream = io.BytesIO()
    Image.new("RGB", (900, 700), "white").save(stream, format="JPEG")
    raw = stream.getvalue()
    original = {"file_id": "synthetic-id", "sample_id": "fixture.jpg", "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "group_id": "synthetic-group", "split": "train"}
    image = {"id": 1, "file_name": "fixture.jpg", "width": 900, "height": 700}
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [625, 300, 30, 20]},
                   {"id": 2, "image_id": 1, "category_id": 8, "bbox": [100, 100, 20, 20]}]
    return raw, original, image, annotations


@pytest.mark.parametrize("length,tile,overlap,expected", [(500, 640, 0.2, [0]), (640, 640, 0.2, [0]),
                                                        (641, 640, 0.2, [0, 1]), (1500, 640, 0.2, [0, 512, 860])])
def test_windows_cover_edges_and_use_declared_overlap(length, tile, overlap, expected):
    assert geometry.starts(length, tile, overlap) == expected
    intervals = [(x, min(x + tile, length)) for x in expected]
    assert intervals[0][0] == 0 and intervals[-1][1] == length
    assert all(left[1] >= right[0] for left, right in zip(intervals, intervals[1:]))


def test_partial_boxes_clip_in_local_coordinates_without_changing_source():
    bbox = [625, 300, 30, 20]
    clipped, fraction = geometry.clipped_box(bbox, (0, 0, 640, 640))
    assert clipped == [625, 300, 15, 20] and fraction == 0.5
    full, fraction = geometry.clipped_box(bbox, (260, 60, 900, 700))
    assert full == [365, 240, 30, 20] and fraction == 1
    assert bbox == [625, 300, 30, 20]
    omitted, fraction = geometry.clipped_box([630, 300, 30, 20], (0, 0, 640, 640))
    assert omitted is None and fraction == pytest.approx(1/3)


def test_actual_crops_preserve_group_split_parents_and_source_ground_truth(tmp_path):
    raw, original, image, annotations = fixture()
    report = geometry.prepare_plate(raw, original, image, annotations, tmp_path)
    assert len(report["tiles"]) == 4
    assert report["source_annotation_ids"] == [1, 2]
    assert annotations[0]["bbox"] == [625, 300, 30, 20]
    for tile in report["tiles"]:
        assert (tile["group_id"], tile["split"], tile["parent_file_id"]) == ("synthetic-group", "train", "synthetic-id")
        content = (tmp_path / tile["file_name"]).read_bytes()
        assert len(content) == tile["size_bytes"] and hashlib.sha256(content).hexdigest() == tile["sha256"]
        with Image.open(io.BytesIO(content)) as crop:
            assert crop.size == (tile["width"], tile["height"])
        assert (tmp_path / tile["file_name"]).stat().st_mode & 0o777 == 0o600
    assert all(t["class_id"] == 0 for tile in report["tiles"] for t in tile["targets"])


def test_held_out_training_preparation_fails_before_writing(tmp_path):
    raw, original, image, annotations = fixture()
    original["split"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        geometry.prepare_plate(raw, original, image, annotations, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tamper", ["sha", "dimensions", "bbox", "crowd", "missing_image"])
def test_changed_bytes_or_incompatible_annotation_contracts_are_not_silently_excluded(tmp_path, tamper):
    raw, original, image, annotations = fixture()
    if tamper == "sha":
        original["sha256"] = "f" * 64
    elif tamper == "dimensions":
        image["width"] += 1
    elif tamper == "bbox":
        annotations[0]["bbox"] = [901, 0, 30, 20]
    elif tamper == "crowd":
        annotations[0]["iscrowd"] = 1
    document = {"images": [] if tamper == "missing_image" else [image], "annotations": annotations,
                "categories": [{"id": 7}, {"id": 8}]}
    with pytest.raises(ValueError):
        _, indexed = geometry.index_coco(document, {"fixture.jpg"})
        geometry.prepare_plate(raw, original, image, indexed["fixture.jpg"], tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [0, 0, 10, -1], [0, float("nan"), 10, 10], [0, 0, True, 10]])
def test_invalid_box_geometry_fails(bbox):
    with pytest.raises(ValueError):
        geometry.checked_box(bbox)


@pytest.mark.parametrize("orientation", list(range(1, 9)))
def test_explicit_exif_frame_is_used_without_rewriting_source_boxes(tmp_path, orientation):
    from PIL import ImageOps

    stream = io.BytesIO()
    image = Image.new("RGB", (90, 70), "white")
    for x in range(20):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 0))
    exif = Image.Exif()
    exif[274] = orientation
    image.save(stream, format="JPEG", exif=exif)
    raw = stream.getvalue()
    with Image.open(io.BytesIO(raw)) as encoded:
        canonical = ImageOps.exif_transpose(encoded)
        width, height = canonical.size
    original = {"file_id": "exif-fixture", "sample_id": "fixture.jpg", "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "group_id": "synthetic-group", "split": "train"}
    annotation = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [30, 30, 5, 5]}
    measured = geometry.prepare_plate(raw, original, {"width": width, "height": height}, [annotation], tmp_path)
    assert measured["normalized_image_dimensions"] == [width, height]
    assert measured["stored_image_dimensions"] == [90, 70]
    assert measured["source_annotations"][0]["bbox"] == [30, 30, 5, 5]
    with Image.open(tmp_path / measured["tiles"][0]["file_name"]) as tile:
        # The retained orientation, including reflections, determines actual crop pixels.
        for point in [(5, 5), (width - 6, 5), (5, height - 6), (width - 6, height - 6)]:
            assert abs(tile.getpixel(point)[0] - canonical.getpixel(point)[0]) < 15
    canonical.close()


def test_partial_edge_and_zero_area_references_are_retained_without_fabricated_labels(tmp_path):
    raw, original, image, annotations = fixture()
    annotations.extend([{"id": 3, "image_id": 1, "category_id": 7, "bbox": [-7, 20, 33, 41]},
                        {"id": 4, "image_id": 1, "category_id": 7, "bbox": [50, 50, 8, 0]}])
    report = geometry.prepare_plate(raw, original, image, annotations, tmp_path)
    assert report["source_annotation_ids"] == [1, 2, 3, 4]
    assert report["source_annotations"][2]["bbox"] == [-7, 20, 33, 41]
    assert report["unlocalizable_source_annotations"][0]["source_annotation_id"] == 4
    targets = [t for tile in report["tiles"] for t in tile["targets"]]
    assert any(t["source_annotation_id"] == 3 and t["bbox"] == [0, 20, 26, 41] for t in targets)
    assert all(t["source_annotation_id"] != 4 for t in targets)
