"""Review retained original photographs before tiling or reading held-out labels.

Perceptual hashes identify review candidates, not proof of independent samples.
No network requests or annotation parsing occur in this component.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from itertools import combinations
from pathlib import Path

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec
from PIL import Image, ImageOps

MAX_ORIGINAL_BYTES = 20 * 1024 * 1024
# Offline source review must preserve the publisher's originals. Thirteen
# retained ADBC photographs exceed the separate 25-million-pixel upload bound;
# the largest verified original has 35,372,336 pixels. Hosting keeps its limit.
MAX_SOURCE_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_IMAGE_PIXELS


def image_identity(content: bytes) -> dict:
    with Image.open(io.BytesIO(content)) as source:
        if source.format != "JPEG" or source.width * source.height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(
                "Original image is outside the pinned source-review contract: "
                f"format={source.format}, dimensions={source.width}x{source.height}, "
                f"maximum_pixels={MAX_SOURCE_IMAGE_PIXELS}"
            )
        source.load()
        image = ImageOps.exif_transpose(source).convert("L")
        # Two directional 64-bit difference hashes and one 256-bit average hash.
        horizontal = image.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        vertical = image.resize((8, 9), Image.Resampling.LANCZOS).tobytes()
        thumbnail = image.resize((16, 16), Image.Resampling.LANCZOS).tobytes()
        differences = [
            horizontal[y * 9 + x] > horizontal[y * 9 + x + 1] for y in range(8) for x in range(8)
        ] + [vertical[y * 8 + x] > vertical[(y + 1) * 8 + x] for y in range(8) for x in range(8)]
        average = sum(thumbnail) / len(thumbnail)
        dhash = sum(int(bit) << i for i, bit in enumerate(differences))
        ahash = sum(int(value > average) << i for i, value in enumerate(thumbnail))
        return {
            "oriented_width": image.width,
            "oriented_height": image.height,
            "dhash128": f"{dhash:032x}",
            "ahash256": f"{ahash:064x}",
            "thumbnail_sha256": hashlib.sha256(thumbnail).hexdigest(),
        }


def review_archives(inventory: dict, paths: list[Path]) -> dict:
    if len(paths) != len(inventory["batches"]):
        raise ValueError("Every retained acquisition batch is required")
    images = []
    annotation_identities = []
    names = set()
    for batch_index, (members, path) in enumerate(zip(inventory["batches"], paths, strict=True)):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            expected = {member["file_name"] for member in members} | {"batch-manifest.json"}
            if len(infos) != len(expected) or {info.filename for info in infos} != expected:
                raise ValueError("Archive entries differ from the immutable inventory")
            retained_members = []
            for member in members:
                name = member["file_name"]
                if Path(name).name != name or name in names:
                    raise ValueError("Unsafe or repeated original filename")
                names.add(name)
                info = archive.getinfo(name)
                if info.file_size != member["size_bytes"] or info.file_size > MAX_ORIGINAL_BYTES:
                    raise ValueError("Original file size differs or exceeds the bounded contract")
                content = archive.read(info)
                if (
                    hashlib.sha256(content).hexdigest() != member["sha256"]
                    or hashlib.md5(content, usedforsecurity=False).hexdigest()
                    != member["publisher_md5"]
                ):
                    raise ValueError("Original file checksum changed")
                retained_members.append({k: v for k, v in member.items() if k != "download_url"})
                if name.lower().endswith(".jpg"):
                    try:
                        identity = image_identity(content)
                    except ValueError as exc:
                        raise ValueError(f"{name}: {exc}") from exc
                    images.append({**retained_members[-1], **identity})
                else:
                    # Verify annotation bytes, but do not inspect any colony outcomes.
                    annotation_identities.append(retained_members[-1])
            manifest_info = archive.getinfo("batch-manifest.json")
            if manifest_info.file_size > 512 * 1024:
                raise ValueError("Batch manifest exceeds its read bound")
            batch_manifest = json.loads(archive.read(manifest_info))
            if (
                batch_manifest.get("batch_index") != batch_index
                or batch_manifest.get("members") != retained_members
                or batch_manifest.get("source_snapshot_sha256")
                != inventory["source_snapshot_sha256"]
            ):
                raise ValueError("Retained batch manifest differs from its inventory")
    candidates = []
    for left, right in combinations(images, 2):
        difference = (int(left["dhash128"], 16) ^ int(right["dhash128"], 16)).bit_count()
        average = (int(left["ahash256"], 16) ^ int(right["ahash256"], 16)).bit_count()
        exact = left["sha256"] == right["sha256"]
        if exact or (difference <= 8 and average <= 8):
            candidates.append(
                {
                    "left": left["file_name"],
                    "right": right["file_name"],
                    "byte_identical": exact,
                    "difference_distance": difference,
                    "average_distance": average,
                    "crosses_preliminary_split": left["split"] != right["split"],
                }
            )
    return {
        "schema_version": 1,
        "stage": "duplicate_review",
        "source_snapshot_sha256": inventory["source_snapshot_sha256"],
        "images": images,
        "annotation_identities": annotation_identities,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "source_review_max_pixels": MAX_SOURCE_IMAGE_PIXELS,
        "method": (
            "Pillow EXIF-oriented grayscale; LANCZOS dHash128 and aHash256; "
            "both Hamming distances <=8, or identical SHA256"
        ),
        "status": "manual_review_required" if candidates else "no_candidates_under_declared_method",
        "training_authorized": False,
        "limits": (
            "Perceptual screening cannot prove acquisition independence. "
            "Resolve candidates and finalize group-preserving splits before training. "
            "No annotation outcomes were read."
        ),
    }


class ReviewADBC(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {
            f"archive_{i}": SignalSpec.scalar(dtype="str", value_type="file", format="zip")
            for i in range(16)
        }

    def outputs(self):
        return {"review_path": SignalSpec.scalar(dtype="str", value_type="file", format="json")}

    def execute(self, inputs, *, context):
        paths = []
        for index in range(16):
            name = f"archive_{index}"
            value = inputs.get(name)
            value = value.value if isinstance(value, BioSignal) else value
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must resolve to a retained archive file path")
            paths.append(Path(value))
        inventory = json.loads((Path(__file__).resolve().parents[1] / "inventory.json").read_text())
        report = review_archives(inventory, paths)
        root = Path(tempfile.mkdtemp(prefix="cfu-duplicate-review-", dir=Path.cwd()))
        root.chmod(0o700)
        output = root / "duplicate-review.json"
        output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        output.chmod(0o600)
        return {"review_path": str(output)}
