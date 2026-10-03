"""Software source-capture fixtures, never proof of upstream weight permissions."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "labs/acquisition/models/upstream/src/upstream.py"
)
spec = importlib.util.spec_from_file_location("cfu_upstream_sources", SOURCE)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class Response(io.BytesIO):
    def __init__(self, content, url):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


def opener(*, content=None, redirect=None):
    def open(request, timeout):
        assert timeout == 30
        source = (
            content
            if content is not None
            else (
                b'[{"tag_name":"fixture-tag","assets":[{"name":"fixture.pth","size":12}]}]'
                if "releases?" in request.full_url
                else b"software source fixture; no weight terms"
            )
        )
        return Response(source, redirect or request.full_url)

    return open


def test_retains_actual_bytes_and_does_not_grant_weight_permission(tmp_path):
    report = upstream.collect_sources(tmp_path, opener=opener())
    assert len(report["members"]) == 3
    assert report["weight_assets"][0]["name"] == "fixture.pth"
    assert report["weight_permission_assessed"] is False
    assert report["weight_bytes_downloaded"] is False
    for member in report["members"]:
        content = (tmp_path / member["file_name"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == member["sha256"]
        assert len(content) == member["size_bytes"]


@pytest.mark.parametrize("bad", ["redirect", "oversize", "empty", "malformed", "index_size"])
def test_rejects_invalid_source_evidence(tmp_path, bad):
    options = {}
    if bad == "redirect":
        options["redirect"] = "https://unrelated.example/claim"
    elif bad == "oversize":
        options["content"] = b"x" * (upstream.MAX_SOURCE_BYTES + 1)
    elif bad == "empty":
        options["content"] = b""
    elif bad == "malformed":
        options["content"] = b"{}"
    else:
        options["content"] = json.dumps([{"assets": []}] * 11).encode()
    with pytest.raises(ValueError):
        upstream.collect_sources(tmp_path, opener=opener(**options))
