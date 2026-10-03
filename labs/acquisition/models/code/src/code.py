"""Retain exact Apache-licensed YOLOX source for native recorded training.

This task never executes fetched code or downloads third-party checkpoints.
The source commit and primary license/README byte identities are pinned.
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from biosimulant import BioModule, ExecutionPolicy, SignalSpec

COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
SOURCE_URL = f"https://codeload.github.com/Megvii-BaseDetection/YOLOX/zip/{COMMIT}"
PRIMARY_DIGESTS = {
    "LICENSE": "0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088",
    "README.md": "023dbf0a5f2c48c944cc6220f9c4750febe2ceafc535445278455186bec7ec6b",
}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 5000


def inspect_source_archive(raw: bytes) -> dict:
    if not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Source archive is empty or exceeds its byte bound")
    prefix = f"YOLOX-{COMMIT}"
    members = []
    primary = {}
    total = 0
    seen = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError("Source archive exceeds its member bound")
        for info in infos:
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute() or ".." in path.parts or "\\" in info.filename
                or not path.parts or path.parts[0] != prefix
                or info.filename in seen
            ):
                raise ValueError(f"Source archive contains an unsafe or repeated path: {info.filename!r}")
            seen.add(info.filename)
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Source archive exceeds its decoded byte bound")
            if info.is_dir():
                continue
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative == ".":
                raise ValueError("Source archive member has no relative filename")
            content = archive.read(info)
            digest = hashlib.sha256(content).hexdigest()
            member = {"path": relative, "size_bytes": len(content), "sha256": digest,
                      "kind": "regular_file"}
            if stat.S_ISLNK(info.external_attr >> 16):
                # Retain the literal link bytes as evidence, never follow or extract it.
                target = content.decode("utf-8")
                target_path = PurePosixPath(target)
                if not target or target_path.is_absolute() or "\\" in target:
                    raise ValueError(f"Source symlink has an unsafe target: {relative!r}")
                resolved = list(path.parts[:-1])
                for part in target_path.parts:
                    if part == "..":
                        if len(resolved) <= 1:
                            raise ValueError(f"Source symlink escapes the archive root: {relative!r}")
                        resolved.pop()
                    elif part != ".":
                        resolved.append(part)
                member.update(kind="symlink_bytes_only", target=target,
                              resolved_archive_path="/".join(resolved))
            members.append(member)
            if relative in PRIMARY_DIGESTS:
                if member["kind"] != "regular_file":
                    raise ValueError(f"Primary source record must be a regular file: {relative!r}")
                primary[relative] = digest
    if primary != PRIMARY_DIGESTS:
        raise ValueError("Source archive does not match retained primary license/README records")
    required = {"yolox/models/yolox.py", "yolox/models/yolo_pafpn.py", "exps/default/yolox_tiny.py"}
    if not required <= {m["path"] for m in members if m["kind"] == "regular_file"}:
        raise ValueError("Pinned native detector source is incomplete")
    return {
        "schema_version": 1,
        "stage": "native_training_source_acquisition",
        "repository": "Megvii-BaseDetection/YOLOX",
        "commit": COMMIT,
        "source_url": SOURCE_URL,
        "source_archive_sha256": hashlib.sha256(raw).hexdigest(),
        "source_archive_size_bytes": len(raw),
        "members": sorted(members, key=lambda m: m["path"]),
        "decoded_size_bytes": total,
        "primary_records_verified": primary,
        "code_license": "Apache-2.0; retain upstream license and notices",
        "code_executed": False,
        "archive_extracted": False,
        "symlinks_followed": False,
        "downstream_extraction_policy": "Only verified regular files may be extracted; never create or follow captured symlinks.",
        "third_party_checkpoint_downloaded": False,
        "starting_weight_decision": (
            "Random initialization for the initial bounded benchmark: third-party weight-specific "
            "permission was not established by the retained primary records. No enterprise purchase. "
            "The benchmark must determine affordable training and recreation budgets."
        ),
        "scientific_acceptance_established": False,
        "source_inspection_passed": True,
    }


def collect_source(root: Path, *, opener=urllib.request.urlopen) -> tuple[Path, Path]:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Biosimulant-CFU-source-audit/1.0"})
    with opener(request, timeout=90) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != "codeload.github.com":
            raise ValueError("Native source redirected outside its declared HTTPS host")
        raw = response.read(MAX_ARCHIVE_BYTES + 1)
    if not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Source archive is empty or exceeds its byte bound")
    try:
        report = inspect_source_archive(raw)
    except (ValueError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        # An acquisition receipt is diagnostic evidence, not permission to use
        # rejected code. Retaining exact bounded bytes makes the rejection auditable.
        report = {
            "schema_version": 1, "stage": "native_training_source_acquisition",
            "repository": "Megvii-BaseDetection/YOLOX", "commit": COMMIT,
            "source_url": SOURCE_URL, "source_archive_sha256": hashlib.sha256(raw).hexdigest(),
            "source_archive_size_bytes": len(raw), "source_inspection_passed": False,
            "inspection_failure": str(error), "downstream_use_permitted": False,
            "code_executed": False, "archive_extracted": False, "symlinks_followed": False,
            "third_party_checkpoint_downloaded": False, "scientific_acceptance_established": False,
        }
    archive_path = root / "yolox-source.zip"
    archive_path.write_bytes(raw)
    archive_path.chmod(0o600)
    manifest_path = root / "native-source-manifest.json"
    manifest_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    manifest_path.chmod(0o600)
    return archive_path, manifest_path


class AcquireNativeSource(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {}

    def outputs(self):
        return {name: SignalSpec.scalar(dtype="str", value_type="file", format=fmt)
                for name, fmt in [("source_archive_path", "zip"), ("source_manifest_path", "json")]}

    def execute(self, inputs, *, context):
        root = Path(tempfile.mkdtemp(prefix="cfu-native-source-", dir=Path.cwd()))
        root.chmod(0o700)
        archive, manifest = collect_source(root)
        return {"source_archive_path": str(archive), "source_manifest_path": str(manifest)}
