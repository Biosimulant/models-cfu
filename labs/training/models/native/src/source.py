"""Verify retained native source before extracting regular files for a managed task.

This library does not import extracted code, download weights, train or run a Lab.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
PRIMARY_DIGESTS = {
    "LICENSE": "0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088",
    "README.md": "023dbf0a5f2c48c944cc6220f9c4750febe2ceafc535445278455186bec7ec6b",
}
REQUIRED = {"yolox/models/yolox.py", "yolox/models/yolo_pafpn.py", "exps/default/yolox_tiny.py"}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 5000


def pinned_bytes(path, pin, maximum):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != pin["size_bytes"] or pin["size_bytes"] > maximum:
        raise ValueError("Retained source input length differs from its pinned identity")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Retained source input digest differs from its pinned identity")
    return raw


def extract_native_source(archive_path, manifest_path, destination, *, archive_pin, manifest_pin):
    raw = pinned_bytes(archive_path, archive_pin, MAX_ARCHIVE_BYTES)
    manifest = json.loads(pinned_bytes(manifest_path, manifest_pin, 1024 * 1024))
    if manifest.get("source_inspection_passed") is not True or manifest.get("commit") != COMMIT:
        raise ValueError("Native source has not passed the pinned source inspection")
    if (manifest.get("source_archive_sha256") != archive_pin["sha256"]
            or manifest.get("source_archive_size_bytes") != archive_pin["size_bytes"]
            or manifest.get("primary_records_verified") != PRIMARY_DIGESTS):
        raise ValueError("Native source receipt differs from the retained archive or primary records")
    records = manifest["members"]
    if len(records) > MAX_MEMBERS or len({m["path"] for m in records}) != len(records):
        raise ValueError("Native source receipt contains repeated or excessive members")
    members = {m["path"]: m for m in records}
    regular, links, seen, decoded = {}, [], set(), 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist()) > MAX_MEMBERS:
            raise ValueError("Native source archive exceeds its member bound")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or not path.parts or path.parts[0] != f"YOLOX-{COMMIT}"
                    or str(path) != info.filename.rstrip("/")):
                raise ValueError("Native source contains a noncanonical or unsafe archive path")
            if info.is_dir():
                continue
            name = str(PurePosixPath(*path.parts[1:]))
            if name == "." or name in seen or name not in members:
                raise ValueError("Native source archive differs from its unique member receipt")
            seen.add(name)
            member = members[name]
            decoded += info.file_size
            if decoded > MAX_DECODED_BYTES or info.file_size != member["size_bytes"]:
                raise ValueError("Native source decoded size differs or exceeds its bound")
            content = archive.read(info)
            if hashlib.sha256(content).hexdigest() != member["sha256"]:
                raise ValueError("Native source member digest changed")
            is_link = stat.S_ISLNK(info.external_attr >> 16)
            if is_link != (member["kind"] == "symlink_bytes_only"):
                raise ValueError("Native source member kind differs from its receipt")
            if is_link:
                if content.decode("utf-8") != member["target"]:
                    raise ValueError("Native source link bytes differ from its receipt")
                links.append(name)
            elif member["kind"] == "regular_file":
                regular[name] = content
            else:
                raise ValueError("Native source receipt declares an unsupported member kind")
    if seen != set(members) or not (REQUIRED | set(PRIMARY_DIGESTS)) <= set(regular):
        raise ValueError("Native source is incomplete or requires a link instead of a regular file")
    if any(hashlib.sha256(regular[n]).hexdigest() != digest for n, digest in PRIMARY_DIGESTS.items()):
        raise ValueError("Native source primary record bytes changed")
    destination = Path(destination)
    # Require a fresh owned tree; never write through pre-existing links/files.
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    selected = {n: content for n, content in regular.items()
                if n in PRIMARY_DIGESTS or n.startswith(("yolox/", "exps/"))}
    try:
        for name, content in sorted(selected.items()):
            target = destination / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            target.chmod(0o600)
    except Exception:
        shutil.rmtree(destination)
        raise
    return {"source_commit": COMMIT, "source_archive_sha256": archive_pin["sha256"],
            "verified_archive_members": len(seen), "extracted_regular_files": len(selected),
            "unextracted_regular_files": sorted(set(regular) - set(selected)),
            "extraction_scope": "regular yolox/ and exps/ packages plus primary license/readme; all archive members verified",
            "unextracted_symlink_entries": sorted(links),
            "code_executed": False}
