"""Synthetic archive and network-response fixtures; no scientific execution."""

import hashlib
import importlib.util
import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/code/src/code.py"
spec = importlib.util.spec_from_file_location("cfu_native_code", SOURCE)
code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code)


def fixture_archive(monkeypatch, *, extra=None, missing=None, altered=False):
    files = {"LICENSE": b"synthetic license", "README.md": b"synthetic readme",
             "yolox/models/yolox.py": b"fixture code, never executed",
             "yolox/models/yolo_pafpn.py": b"fixture code, never executed",
             "exps/default/yolox_tiny.py": b"fixture code, never executed"}
    monkeypatch.setattr(code, "PRIMARY_DIGESTS", {n: hashlib.sha256(files[n]).hexdigest() for n in ["LICENSE", "README.md"]})
    if altered:
        files["LICENSE"] = b"changed license"
    if missing:
        files.pop(missing)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, body in files.items():
            archive.writestr(f"YOLOX-{code.COMMIT}/{name}", body)
        if extra is not None:
            archive.writestr(extra, b"unsafe entry")
    return stream.getvalue()


def test_only_verified_source_is_retained_and_no_code_is_executed(tmp_path, monkeypatch):
    raw = fixture_archive(monkeypatch)

    class Response(io.BytesIO):
        def geturl(self):
            return code.SOURCE_URL

    def opener(request, *, timeout):
        assert request.full_url == code.SOURCE_URL and timeout == 90
        return Response(raw)

    archive, manifest = code.collect_source(tmp_path, opener=opener)
    report = json.loads(manifest.read_text())
    assert archive.read_bytes() == raw
    assert report["source_archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["source_archive_size_bytes"] == len(raw)
    assert len(report["members"]) == 5
    assert report["code_executed"] is False
    assert report["third_party_checkpoint_downloaded"] is False
    assert archive.stat().st_mode & 0o777 == 0o600
    assert manifest.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("extra", ["/absolute.py", "../escape.py", f"YOLOX-{code.COMMIT}/../escape.py",
                                  "wrong-prefix/file.py", f"YOLOX-{code.COMMIT}/LICENSE", f"YOLOX-{code.COMMIT}/bad\\path.py"])
def test_unsafe_or_repeated_source_paths_fail(extra, monkeypatch):
    raw = fixture_archive(monkeypatch, extra=extra)
    with pytest.raises(ValueError, match="unsafe or repeated"):
        code.inspect_source_archive(raw)


@pytest.mark.parametrize("missing", ["LICENSE", "README.md", "yolox/models/yolox.py"])
def test_missing_source_and_primary_record_changes_fail(missing, monkeypatch):
    with pytest.raises(ValueError):
        code.inspect_source_archive(fixture_archive(monkeypatch, missing=missing))


def test_a_changed_license_is_not_silently_accepted(monkeypatch):
    with pytest.raises(ValueError, match="primary"):
        code.inspect_source_archive(fixture_archive(monkeypatch, altered=True))


def test_external_redirect_is_rejected_before_writing_outputs(tmp_path):
    class Response(io.BytesIO):
        def geturl(self):
            return "https://unapproved.invalid/source.zip"

    with pytest.raises(ValueError, match="HTTPS host"):
        code.collect_source(tmp_path, opener=lambda *a, **k: Response(b"bytes"))
    assert not list(tmp_path.iterdir())


def test_source_size_bounds_are_enforced(monkeypatch):
    raw = fixture_archive(monkeypatch)
    monkeypatch.setattr(code, "MAX_ARCHIVE_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="byte bound"):
        code.inspect_source_archive(raw)


def test_decoded_archive_bound_is_enforced(monkeypatch):
    raw = fixture_archive(monkeypatch)
    monkeypatch.setattr(code, "MAX_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="decoded"):
        code.inspect_source_archive(raw)


def with_symlink(raw, name, target):
    stream = io.BytesIO(raw)
    with zipfile.ZipFile(stream, "a") as archive:
        info = zipfile.ZipInfo(f"YOLOX-{code.COMMIT}/{name}")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)
    return stream.getvalue()


def test_confined_source_link_is_retained_only_as_literal_bytes(monkeypatch):
    raw = with_symlink(fixture_archive(monkeypatch), "demo/linked.py", "../yolox/models/yolox.py")
    report = code.inspect_source_archive(raw)
    link = next(m for m in report["members"] if m["path"] == "demo/linked.py")
    assert link["kind"] == "symlink_bytes_only"
    assert link["sha256"] == hashlib.sha256(b"../yolox/models/yolox.py").hexdigest()
    assert link["resolved_archive_path"] == f"YOLOX-{code.COMMIT}/yolox/models/yolox.py"
    assert report["archive_extracted"] is False
    assert report["symlinks_followed"] is False


@pytest.mark.parametrize("target", ["/outside", "../../escape", "bad\\path", ""])
def test_source_link_cannot_resolve_outside_archive(monkeypatch, target):
    raw = with_symlink(fixture_archive(monkeypatch), "demo/linked.py", target)
    with pytest.raises(ValueError, match="symlink"):
        code.inspect_source_archive(raw)


def test_rejected_archive_is_a_retained_diagnostic_not_usable_code(tmp_path, monkeypatch):
    raw = fixture_archive(monkeypatch, extra="wrong-prefix/file.py")

    class Response(io.BytesIO):
        def geturl(self):
            return code.SOURCE_URL

    archive, manifest = code.collect_source(tmp_path, opener=lambda *a, **k: Response(raw))
    report = json.loads(manifest.read_text())
    assert archive.read_bytes() == raw
    assert report["source_inspection_passed"] is False
    assert report["downstream_use_permitted"] is False
    assert "wrong-prefix/file.py" in report["inspection_failure"]
    assert report["source_archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["code_executed"] is False and report["archive_extracted"] is False
