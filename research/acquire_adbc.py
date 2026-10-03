"""Acquire the exact ADBC v3 originals and freeze plate splits before tiling.

Only file identity and publisher metadata are read here. The annotation bytes
are preserved, but held-out labels are not used to select a model or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "https://api.figshare.com/v2/articles/22022540/versions/3"
SPLIT_SEED = 20261001


def download(item: dict, directory: Path) -> dict:
    name = item["name"]
    if Path(name).name != name:
        raise ValueError("Unsafe publisher file name")
    path = directory / name

    def identity():
        md5, sha256 = hashlib.md5(), hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                md5.update(block)
                sha256.update(block)
                size += len(block)
        if size != item["size"] or md5.hexdigest() != item["computed_md5"]:
            raise ValueError(f"Publisher identity mismatch: {name}")
        return {
            "file_name": name,
            "size_bytes": size,
            "sha256": sha256.hexdigest(),
            "publisher_md5": md5.hexdigest(),
            "publisher_file_id": item["id"],
        }

    if path.exists():
        return identity()
    for attempt in range(3):
        incoming = path.with_suffix(path.suffix + ".incoming")
        try:
            with (
                urllib.request.urlopen(item["download_url"], timeout=90) as response,
                incoming.open("wb") as handle,
            ):
                while block := response.read(1024 * 1024):
                    handle.write(block)
            incoming.replace(path)
            return identity()
        except Exception:
            incoming.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Download retry exhausted")


def freeze_split(images: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for image in images:
        groups.setdefault(image["sha256"], []).append(image)
    keys = sorted(groups)
    random.Random(SPLIT_SEED).shuffle(keys)
    train_end = int(len(keys) * 0.8)
    validation_end = int(len(keys) * 0.9)
    rows = []
    for index, key in enumerate(keys):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        for image in groups[key]:
            rows.append({**image, "sample_id": image["file_name"], "group_id": key, "split": split})
    return sorted(rows, key=lambda row: row["file_name"])


def acquire(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "source.json"
    if not snapshot.exists():
        snapshot.write_bytes(urllib.request.urlopen(SOURCE, timeout=30).read())
    source = json.loads(snapshot.read_bytes())
    if source.get("version") != 3 or source.get("license", {}).get("name") != "CC BY 4.0":
        raise ValueError("Unexpected source version/license")
    chosen = [
        item
        for item in source["files"]
        if item["name"].lower().endswith(".jpg") or item["name"] == "annot_COCO.json"
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        identities = list(pool.map(lambda item: download(item, directory), chosen))
    images = [item for item in identities if item["file_name"].lower().endswith(".jpg")]
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "source_version": 3,
        "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "license": source["license"],
        "attribution": source.get("authors", []),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": SPLIT_SEED,
        "allocation": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_unit": "whole original image; byte-identical images grouped",
        "acquisition_groups": (
            "Not supplied by publisher; filenames are not treated as "
            "independent acquisition batches."
        ),
        "near_duplicate_review": "Not yet performed; required before training.",
        "images": freeze_split(images),
        "annotations": next(item for item in identities if item["file_name"] == "annot_COCO.json"),
    }
    # A resumed acquisition may verify, but never overwrite a frozen split.
    output = directory / "dataset-manifest.json"
    if output.exists():
        existing = json.loads(output.read_bytes())
        if (
            existing["images"] != manifest["images"]
            or existing["annotations"] != manifest["annotations"]
        ):
            raise ValueError("Frozen dataset identity/split changed")
        return existing
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = acquire(args.directory)
    print(
        json.dumps(
            {
                "images": len(manifest["images"]),
                "splits": {
                    split: sum(row["split"] == split for row in manifest["images"])
                    for split in ("train", "validation", "test")
                },
                "annotations_sha256": manifest["annotations"]["sha256"],
            }
        )
    )
