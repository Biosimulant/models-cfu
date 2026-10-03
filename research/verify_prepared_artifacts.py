"""Independent checks of MCP-owned preparation artifacts; URLs stay ephemeral.

This reads completed artifacts, never executes training or creates managed evidence.
Original target inspection is restricted to the exact train/validation batches.
"""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


def download(envelope):
    if not envelope["download"].startswith("https://") or not 0 < envelope["size_bytes"] <= 128 * 1024 * 1024:
        raise ValueError("Invalid bounded owned-artifact envelope")
    request = urllib.request.Request(envelope["download"], headers={"User-Agent": "Biosimulant-MCP-Artifact-Verification/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read(envelope["size_bytes"] + 1)
    if len(raw) != envelope["size_bytes"] or hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("Artifact byte identity changed")
    return raw


def positions(length):
    if length <= 640:
        return [0]
    result = list(range(0, length - 639, 512))
    if result[-1] != length - 640:
        result.append(length - 640)
    return result


def verify(envelopes, root, config, *, reader=download):
    frozen = json.loads((root / "verified-frozen-original-split.json").read_text())
    originals = {m["sample_id"]: m for m in frozen["native_dataset_manifest"]["members"] if m["role"] == "image"}
    annotation_bytes = (root / "verified-original-annotations.json").read_bytes()
    assert hashlib.sha256(annotation_bytes).hexdigest() == config["annotations"]["sha256"]
    labels = json.loads(annotation_bytes)
    image_metadata = {Path(i["file_name"]).name: i for i in labels["images"]}
    summary, all_tiles = [], []
    for item in envelopes:
        artifacts = {role: reader(value) for role, value in item["artifacts"].items()}
        if "results" in artifacts:
            assert len(json.loads(artifacts["results"])["outputs"]) > 0
        raw = artifacts["prepared_manifest_path"]
        report = json.loads(raw)
        batch = report["batch_index"]
        if "batch_index" in item:
            assert batch == item["batch_index"]
        planned = config["batches"][batch]
        assert report["frozen_dataset_id"] == config["frozen_dataset_id"]
        assert report["frozen_dataset_sha256"] == config["frozen_dataset_sha256"]
        assert report["test_targets_inspected_or_emitted"] is False and report["source_annotations_modified"] is False
        assert report["training_executed"] is False
        assert [p["parent"]["sample_id"] for p in report["plates"]] == planned
        assert report["consumed_original_file_ids"] == [originals[n]["file_id"] for n in planned]
        assert all(originals[n]["split"] in {"train", "validation"} for n in planned)
        selected_image_ids = {image_metadata[n]["id"] for n in planned}
        selected_annotations = {i: [] for i in selected_image_ids}
        for annotation in labels["annotations"]:
            if annotation["image_id"] in selected_annotations:
                selected_annotations[annotation["image_id"]].append(annotation)
        tiles = [t for p in report["plates"] for t in p["tiles"]]
        expected_names = {"prepared-batch.json"} | {"tiles/" + t["file_name"] for t in tiles}
        assert len(expected_names) == len(tiles) + 1 and report["derived_image_count"] == len(tiles)
        with zipfile.ZipFile(io.BytesIO(artifacts["tiles_archive_path"])) as archive:
            assert len(archive.namelist()) == len(expected_names) and set(archive.namelist()) == expected_names
            assert archive.read("prepared-batch.json") == raw
            for plate in report["plates"]:
                parent = plate["parent"]
                assert parent == originals[parent["sample_id"]]
                coco = image_metadata[parent["sample_id"]]
                width, height = coco["width"], coco["height"]
                legacy_pilot = batch == 0 and plate["coordinate_basis"] == "source stored pixel frame"
                if legacy_pilot:
                    assert plate["exif_orientation_retained_as_metadata"] == 1
                    assert parent["sample_id"] in config["batches"][0]
                else:
                    assert plate["normalized_image_dimensions"] == [width, height]
                    assert plate["coordinate_basis"] == "EXIF-normalized original annotation frame"
                assert plate["exif_orientation_retained_as_metadata"] in range(1, 9)
                expected = selected_annotations[coco["id"]]
                if not legacy_pilot:
                    assert plate["source_annotations"] == [{"id": a["id"], "bbox": a["bbox"], "category_id": a["category_id"]} for a in expected]
                assert plate["source_annotation_ids"] == sorted(a["id"] for a in expected)
                zero = {a["id"] for a in expected if 0 in a["bbox"][2:]}
                assert {a["source_annotation_id"] for a in plate.get("unlocalizable_source_annotations", [])} == zero
                boxes = {a["id"]: a["bbox"] for a in expected}
                expected_windows = [[x, y, min(x + 640, width), min(y + 640, height)] for y in positions(height) for x in positions(width)]
                assert [t["source_window_xyxy"] for t in plate["tiles"]] == expected_windows
                for tile in plate["tiles"]:
                    assert all(tile[k] == parent[k] for k in ["sample_id", "group_id", "split"])
                    assert tile["parent_file_id"] == parent["file_id"] and tile["parent_sha256"] == parent["sha256"]
                    content = archive.read("tiles/" + tile["file_name"])
                    assert len(content) == tile["size_bytes"] and hashlib.sha256(content).hexdigest() == tile["sha256"]
                    with Image.open(io.BytesIO(content)) as image:
                        assert image.format == "JPEG" and list(image.size) == [tile["width"], tile["height"]]
                        image.load()
                    left, top, right, bottom = tile["source_window_xyxy"]
                    assert [tile["width"], tile["height"]] == [right - left, bottom - top]
                    expected_targets = {}
                    for identifier, (x, y, w, h) in boxes.items():
                        if w == 0 or h == 0:
                            continue
                        x0, y0, x1, y1 = max(x, left), max(y, top), min(x + w, right), min(y + h, bottom)
                        cw, ch = max(0, x1 - x0), max(0, y1 - y0)
                        fraction = cw * ch / (w * h)
                        if fraction >= .5 and min(cw, ch) >= 1:
                            expected_targets[identifier] = ([x0 - left, y0 - top, cw, ch], fraction)
                    actual_targets = {t["source_annotation_id"]: t for t in tile["targets"]}
                    assert len(actual_targets) == len(tile["targets"]) and set(actual_targets) == set(expected_targets)
                    for identifier, (bbox, fraction) in expected_targets.items():
                        actual = actual_targets[identifier]
                        assert actual["bbox"] == bbox and actual["retained_area_fraction"] == fraction and actual["class_id"] == 0
        assert sum(t["size_bytes"] for t in tiles) == report["derived_image_bytes"]
        path = root / f"verified-prepared-batch-{batch:02d}.json"
        path.write_bytes(raw)
        path.chmod(0o600)
        tile_path = root / f"verified-prepared-tile-identities-{batch:02d}.json"
        tile_path.write_text(json.dumps(tiles, separators=(",", ":")))
        tile_path.chmod(0o600)
        summary.append({"run_id": item["run_id"], "batch_index": batch, "source_plates": len(planned), "derived_tiles": len(tiles),
                        "derived_image_bytes": report["derived_image_bytes"], "effective_environment": report["effective_environment"],
                        "zero_area_references_retained": sum(len(p.get("unlocalizable_source_annotations", [])) for p in report["plates"]),
                        "test_outcomes_inspected": False, "training_executed": False})
        all_tiles.extend(tiles)
    splits = {}
    for tile in all_tiles:
        splits.setdefault(tile["sha256"], set()).add(tile["split"])
    assert all(len(s) == 1 for s in splits.values()), "Byte-identical crops cross frozen partitions"
    return summary


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.read())
        verified = verify(request["envelopes"], Path(request["private_root"]), request["config"])
        print(json.dumps(verified))
    except Exception as error:
        # Never echo exceptions containing short-lived owner URLs.
        print(json.dumps({"verification_failed": True, "error_type": type(error).__name__, "http_code": getattr(error, "code", None)}))
        sys.exit(1)
