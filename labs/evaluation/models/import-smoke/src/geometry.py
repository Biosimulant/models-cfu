"""Source-faithful crop geometry for the forthcoming managed preparation task.

This library alone is not a runnable Lab and does not freeze datasets, train,
evaluate, or authorize release. It is developed on staging with software fixtures.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

MAX_SOURCE_PIXELS = 40_000_000
MAX_SOURCE_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


def starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Image side must be a positive integer")
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("Tile side must be a positive integer")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Overlap must be finite and in [0, 1)")
    step = max(1, int(tile_size * (1 - overlap)))
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def windows(width: int, height: int, tile_size=640, overlap=0.2) -> list[tuple[int, int, int, int]]:
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in starts(height, tile_size, overlap)
            for x in starts(width, tile_size, overlap)]


def checked_box(value, *, allow_zero=False) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("COCO bbox must contain four numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("COCO bbox must contain finite numbers")
    x, y, width, height = map(float, value)
    if width < 0 or height < 0 or (not allow_zero and (width == 0 or height == 0)):
        raise ValueError("COCO bbox sides must be positive, or explicitly retained zero-area source references")
    return x, y, width, height


def clipped_box(bbox, window, *, minimum_retained_area=0.5, minimum_side=1.0):
    x, y, width, height = checked_box(bbox)
    if not math.isfinite(minimum_retained_area) or not 0 <= minimum_retained_area <= 1:
        raise ValueError("Minimum retained fraction must be in [0, 1]")
    if not math.isfinite(minimum_side) or minimum_side <= 0:
        raise ValueError("Minimum box side must be positive")
    x0, y0, x1, y1 = window
    left, top = max(x, x0), max(y, y0)
    right, bottom = min(x + width, x1), min(y + height, y1)
    clipped_width, clipped_height = max(0.0, right - left), max(0.0, bottom - top)
    fraction = clipped_width * clipped_height / (width * height)
    if fraction < minimum_retained_area or min(clipped_width, clipped_height) < minimum_side:
        return None, fraction
    return [left - x0, top - y0, clipped_width, clipped_height], fraction


def index_coco(document: dict, original_names: set[str], *, selected_names: set[str] | None = None) -> tuple[dict, dict]:
    if not isinstance(document, dict) or not isinstance(document.get("images"), list) or not isinstance(
        document.get("annotations"), list
    ) or not isinstance(document.get("categories"), list):
        raise ValueError("Original annotations are not the declared COCO contract")
    categories = {c["id"] for c in document["categories"]}
    if len(categories) != len(document["categories"]) or not categories:
        raise ValueError("Missing or repeated source categories")
    by_name, by_id = {}, {}
    for image in document["images"]:
        filename = image.get("file_name")
        if not isinstance(filename, str) or "\\" in filename:
            raise ValueError("COCO image name is missing or ambiguous")
        name = PurePosixPath(filename).name
        if name not in original_names or name in by_name or image["id"] in by_id:
            raise ValueError("COCO image identity differs from retained originals")
        if any(isinstance(image[k], bool) or not isinstance(image[k], int) or image[k] <= 0 for k in ["width", "height"]):
            raise ValueError("COCO image dimensions must be positive integers")
        by_name[name] = image
        by_id[image["id"]] = name
    if set(by_name) != original_names:
        raise ValueError("COCO labels do not cover every retained original")
    selected = original_names if selected_names is None else selected_names
    if not selected <= original_names:
        raise ValueError("Selected preparation images differ from retained originals")
    annotations = {name: [] for name in selected}
    seen = set()
    for annotation in document["annotations"]:
        if annotation["image_id"] not in by_id:
            raise ValueError("COCO annotation image identity is invalid")
        name = by_id[annotation["image_id"]]
        if name not in selected:
            # Do not inspect or emit outcomes for plates outside this batch.
            continue
        if annotation["id"] in seen or annotation["category_id"] not in categories:
            raise ValueError("COCO annotation identity or source category is invalid")
        seen.add(annotation["id"])
        checked_box(annotation["bbox"], allow_zero=True)
        if annotation.get("iscrowd", 0) != 0:
            raise ValueError("Crowd labels require an explicit ground-truth decision; no silent exclusion")
        annotations[name].append(annotation)
    return by_name, annotations


def prepare_plate(content: bytes, original: dict, coco_image: dict, annotations: list,
                  directory: Path, *, tile_size=640, overlap=0.2,
                  minimum_retained_area=0.5, minimum_side=1.0) -> dict:
    if original["split"] not in {"train", "validation"}:
        raise ValueError("Training preparation must not crop or emit held-out test targets")
    name = original["sample_id"]
    if Path(name).name != name:
        raise ValueError("Unsafe original name")
    if len(content) != original["size_bytes"] or len(content) > MAX_SOURCE_BYTES or hashlib.sha256(content).hexdigest() != original["sha256"]:
        raise ValueError(f"{name}: original image byte identity changed")
    with Image.open(io.BytesIO(content)) as encoded:
        if encoded.format != "JPEG" or encoded.width * encoded.height > MAX_SOURCE_PIXELS:
            raise ValueError(f"{name}: image is outside the offline source contract")
        orientation = encoded.getexif().get(274, 1)
        if isinstance(orientation, bool) or orientation not in range(1, 9):
            raise ValueError(f"{name}: invalid EXIF orientation; no inferred rotation")
        stored_dimensions = [encoded.width, encoded.height]
        source = encoded if orientation == 1 else ImageOps.exif_transpose(encoded)
        if (source.width, source.height) != (coco_image["width"], coco_image["height"]):
            if source is not encoded:
                source.close()
            raise ValueError(f"{name}: EXIF-normalized dimensions disagree with source annotations")
        source.load()
        unlocalizable = []
        for annotation in annotations:
            x, y, w, h = checked_box(annotation["bbox"], allow_zero=True)
            if w == 0 or h == 0:
                unlocalizable.append({"source_annotation_id": annotation["id"], "bbox": list(annotation["bbox"]),
                                      "reason": "zero-area original reference; retained count, no fabricated training geometry"})
            elif min(x + w, source.width) <= max(x, 0) or min(y + h, source.height) <= max(y, 0):
                if source is not encoded:
                    source.close()
                raise ValueError(f"{name}: source bbox has no visible intersection with its annotation frame")
        tiles = []
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        for index, window in enumerate(windows(source.width, source.height, tile_size, overlap)):
            identifier = f"{original['sha256'][:24]}-{index:04d}"
            path = directory / (identifier + ".jpg")
            if path.exists():
                raise ValueError("A derived tile would overwrite an existing artifact")
            crop = source.crop(window).convert("RGB")
            crop.save(path, format="JPEG", quality=95, subsampling=0)
            path.chmod(0o600)
            targets, omitted_fragments = [], []
            for annotation in annotations:
                if 0 in annotation["bbox"][2:]:
                    continue
                box, fraction = clipped_box(annotation["bbox"], window,
                                            minimum_retained_area=minimum_retained_area,
                                            minimum_side=minimum_side)
                if box is not None:
                    targets.append({"source_annotation_id": annotation["id"], "class_id": 0,
                                    "bbox": box, "retained_area_fraction": fraction})
                elif fraction > 0:
                    omitted_fragments.append({"source_annotation_id": annotation["id"],
                                              "retained_area_fraction": fraction})
            raw = path.read_bytes()
            tiles.append({"file_name": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                          "size_bytes": len(raw), "width": crop.width, "height": crop.height,
                          "source_window_xyxy": list(window), "targets": targets,
                          "omitted_partial_fragments": omitted_fragments,
                          "parent_file_id": original["file_id"], "parent_sha256": original["sha256"],
                          "sample_id": original["sample_id"], "group_id": original["group_id"],
                          "split": original["split"]})
            crop.close()
        normalized_dimensions = [source.width, source.height]
        if source is not encoded:
            source.close()
    return {"parent": dict(original), "coordinate_basis": "EXIF-normalized original annotation frame",
            "stored_image_dimensions": stored_dimensions, "normalized_image_dimensions": normalized_dimensions,
            "source_annotations": [{"id": a["id"], "bbox": list(a["bbox"]), "category_id": a["category_id"]} for a in annotations],
            "unlocalizable_source_annotations": unlocalizable,
            "exif_orientation_retained_as_metadata": orientation,
            "source_annotation_ids": sorted(a["id"] for a in annotations),
            "tile_size": tile_size, "overlap": overlap,
            "minimum_retained_area": minimum_retained_area, "minimum_side_pixels": minimum_side,
            "source_category_policy": "collapse annotated colony categories to single class 0",
            "tiles": tiles,
            "limits": "Partial fragments below the declared thresholds are logged, not removed from whole-plate ground truth."}
