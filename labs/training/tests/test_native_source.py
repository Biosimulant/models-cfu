"""Source extraction integrity/security fixtures, never native training evidence."""

import hashlib
import importlib.util
import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/native/src/source.py"
spec = importlib.util.spec_from_file_location("cfu_training_source_fixture", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def pin(path, raw):
    path.write_bytes(raw)
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(tmp_path, monkeypatch, *, unsafe=False):
    files = {"LICENSE": b"synthetic license", "README.md": b"synthetic source description",
             **{n: b"synthetic source, never imported" for n in source.REQUIRED}}
    monkeypatch.setattr(source, "PRIMARY_DIGESTS", {n: hashlib.sha256(files[n]).hexdigest() for n in ["LICENSE", "README.md"]})
    stream, members = io.BytesIO(), []
    with zipfile.ZipFile(stream, "w") as archive:
        for name, raw in files.items():
            archive.writestr(f"YOLOX-{source.COMMIT}/{name}", raw)
            members.append({"path": name, "kind": "regular_file", "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        name = "docs/../escape" if unsafe else "docs/reference.md"
        info = zipfile.ZipInfo(f"YOLOX-{source.COMMIT}/{name}")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        raw = b"../README.md"
        archive.writestr(info, raw)
        members.append({"path": name, "kind": "symlink_bytes_only", "target": raw.decode(),
                        "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    archive_path = tmp_path / "source.zip"
    archive_pin = pin(archive_path, stream.getvalue())
    manifest = {"source_inspection_passed": True, "commit": source.COMMIT,
                "source_archive_sha256": archive_pin["sha256"], "source_archive_size_bytes": archive_pin["size_bytes"],
                "primary_records_verified": source.PRIMARY_DIGESTS, "members": members}
    manifest_path = tmp_path / "receipt.json"
    manifest_pin = pin(manifest_path, json.dumps(manifest).encode())
    return archive_path, manifest_path, archive_pin, manifest_pin, manifest


def test_only_verified_regular_files_are_created_without_following_links(tmp_path, monkeypatch):
    archive, manifest, archive_pin, manifest_pin, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "native"
    report = source.extract_native_source(archive, manifest, output, archive_pin=archive_pin, manifest_pin=manifest_pin)
    assert report["extracted_regular_files"] == 5
    assert report["unextracted_symlink_entries"] == ["docs/reference.md"]
    assert report["code_executed"] is False
    assert not (output / "docs/reference.md").exists()
    assert (output / "LICENSE").read_bytes() == b"synthetic license"
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.rglob("*") if p.is_file())


@pytest.mark.parametrize("tamper", ["archive", "receipt", "inspection", "commit", "member_digest", "member_kind", "missing_member"])
def test_unverified_or_changed_source_fails_before_creating_any_tree(tmp_path, monkeypatch, tamper):
    archive, manifest_path, archive_pin, manifest_pin, manifest = fixture(tmp_path, monkeypatch)
    if tamper == "archive":
        archive.write_bytes(b"changed")
    elif tamper == "receipt":
        manifest_path.write_bytes(b"changed")
    else:
        if tamper == "inspection":
            manifest["source_inspection_passed"] = False
        elif tamper == "commit":
            manifest["commit"] = "unapproved"
        elif tamper == "member_digest":
            manifest["members"][0]["sha256"] = "f" * 64
        elif tamper == "member_kind":
            manifest["members"][-1]["kind"] = "regular_file"
        else:
            manifest["members"].pop()
        manifest_pin = pin(manifest_path, json.dumps(manifest).encode())
    output = tmp_path / "native"
    with pytest.raises(ValueError):
        source.extract_native_source(archive, manifest_path, output, archive_pin=archive_pin, manifest_pin=manifest_pin)
    assert not output.exists()


def test_noncanonical_archive_path_is_rejected_even_with_matching_receipt(tmp_path, monkeypatch):
    archive, manifest, archive_pin, manifest_pin, _ = fixture(tmp_path, monkeypatch, unsafe=True)
    with pytest.raises(ValueError, match="unsafe archive path"):
        source.extract_native_source(archive, manifest, tmp_path / "native", archive_pin=archive_pin, manifest_pin=manifest_pin)
    assert not (tmp_path / "native").exists()


def test_existing_destination_is_never_overwritten(tmp_path, monkeypatch):
    archive, manifest, archive_pin, manifest_pin, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "native"
    output.mkdir()
    (output / "owned.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        source.extract_native_source(archive, manifest, output, archive_pin=archive_pin, manifest_pin=manifest_pin)
    assert [p.name for p in output.iterdir()] == ["owned.txt"]
