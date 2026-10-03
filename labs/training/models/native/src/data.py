"""Retained training crops and loss targets; no label truncation or test access."""

import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import zipfile

from PIL import Image

from .source import pinned_bytes

MAX_TARGETS = 4096


def loss_targets(tile):
    """Native YOLOX rows: class, center-x, center-y, width, height, in pixels."""
    if tile["split"] != "train":
        raise ValueError("Optimizer targets must come from the frozen training partition")
    if len(tile["targets"]) > MAX_TARGETS:
        raise ValueError("Dense crop exceeds the explicit target bound; never truncate it")
    rows = []
    seen = set()
    for target in tile["targets"]:
        annotation_id = target["source_annotation_id"]
        if annotation_id in seen or target["class_id"] != 0:
            raise ValueError("Crop repeats an original annotation or changes the colony class")
        seen.add(annotation_id)
        x, y, width, height = target["bbox"]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (x, y, width, height)) or x < 0 or y < 0 or width < 1 or height < 1
                or x + width > tile["width"] or y + height > tile["height"]):
            raise ValueError("Prepared training target is outside its verified crop contract")
        rows.append([0.0, x + width / 2, y + height / 2, width, height])
    return rows


class TrainingArchive:
    def __init__(self, path, frozen_path, plan, *, require_training=True):
        raw = pinned_bytes(path, plan["archive"], 128 * 1024 * 1024)
        frozen = json.loads(pinned_bytes(frozen_path, plan["frozen_split"], 1024 * 1024))
        self.archive = zipfile.ZipFile(io.BytesIO(raw))
        try:
            entries = self.archive.infolist()
            if len(entries) > 10000 or len({i.filename for i in entries}) != len(entries):
                raise ValueError("Prepared archive has repeated or excessive members")
            for entry in entries:
                p = PurePosixPath(entry.filename)
                if p.is_absolute() or ".." in p.parts or "\\" in entry.filename or str(p) != entry.filename:
                    raise ValueError("Prepared archive member is not a canonical relative path")
            info = self.archive.getinfo("prepared-batch.json")
            if info.file_size != plan["manifest"]["size_bytes"]:
                raise ValueError("Prepared manifest byte length differs from its pin")
            manifest_raw = self.archive.read(info)
            if hashlib.sha256(manifest_raw).hexdigest() != plan["manifest"]["sha256"]:
                raise ValueError("Prepared manifest byte digest differs from its pin")
            manifest = json.loads(manifest_raw)
            if (manifest["batch_index"] != plan["preparation_batch"]
                    or manifest["frozen_dataset_id"] != plan["frozen_dataset_id"]
                    or manifest["frozen_dataset_sha256"] != plan["frozen_dataset_sha256"]
                    or manifest["source_annotations_modified"] is not False
                    or manifest["test_targets_inspected_or_emitted"] is not False):
                raise ValueError("Preparation manifest differs from the frozen scientific inputs")
            originals = {m["sample_id"]: m for m in frozen["native_dataset_manifest"]["members"] if m["role"] == "image"}
            self.tiles, self.total_tiles, expected = [], 0, {"prepared-batch.json"}
            for plate in manifest["plates"]:
                for tile in plate["tiles"]:
                    name = "tiles/" + tile["file_name"]
                    if name in expected:
                        raise ValueError("Preparation repeats a derived crop identity")
                    expected.add(name)
                    parent = originals.get(tile["sample_id"])
                    if (not parent or tile["split"] not in {"train", "validation"}
                            or tile["parent_file_id"] != parent["file_id"]
                            or tile["parent_sha256"] != parent["sha256"]
                            or any(tile[k] != parent[k] for k in ("group_id", "split"))):
                        raise ValueError("Crop parent or partition differs from frozen original")
                    if not 0 < tile["width"] <= 640 or not 0 < tile["height"] <= 640:
                        raise ValueError("Crop dimensions exceed the fixed 640-pixel input")
                    entry = self.archive.getinfo(name)
                    if entry.file_size != tile["size_bytes"] or not 0 < entry.file_size <= 20 * 1024 * 1024:
                        raise ValueError("Prepared crop byte length differs from its retained identity")
                    if hashlib.sha256(self.archive.read(entry)).hexdigest() != tile["sha256"]:
                        raise ValueError("Prepared crop digest differs from its retained identity")
                    self.total_tiles += 1
                    if tile["split"] == "train":
                        loss_targets(tile)
                        self.tiles.append(tile)
            if expected != {i.filename for i in entries} or self.total_tiles != manifest["derived_image_count"]:
                raise ValueError("Prepared archive is incomplete or has unrecorded members")
            if require_training and not self.tiles:
                raise ValueError("Benchmark archive contains no frozen training crops")
            # Dense crops exercise assignment memory; deterministic byte ordering breaks ties.
            self.tiles.sort(key=lambda t: (-len(t["targets"]), t["sha256"], t["file_name"]))
        except Exception:
            self.archive.close()
            raise

    def image(self, tile):
        with Image.open(io.BytesIO(self.archive.read("tiles/" + tile["file_name"]))) as image:
            if image.format != "JPEG" or image.size != (tile["width"], tile["height"]):
                raise ValueError("Decoded crop differs from its prepared image contract")
            image.load()
            return image.convert("RGB")

    def close(self):
        self.archive.close()
