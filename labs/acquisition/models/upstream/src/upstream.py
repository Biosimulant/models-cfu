"""Retain official software/weight-source records; do not infer weight permission."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from biosimulant import BioModule, ExecutionPolicy, SignalSpec

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
SOURCES = {
    "LICENSE.txt": f"https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{YOLOX_COMMIT}/LICENSE",
    "README.md": f"https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{YOLOX_COMMIT}/README.md",
    "releases.json": "https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases?per_page=10",
}
MAX_SOURCE_BYTES = 512 * 1024


def collect_sources(root: Path, *, opener=urllib.request.urlopen) -> dict:
    members = []
    for name, url in SOURCES.items():
        request = urllib.request.Request(
            url, headers={"User-Agent": "Biosimulant-CFU-source-audit/1.0"}
        )
        with opener(request, timeout=30) as response:
            final_url = response.geturl()
            parsed = urlsplit(final_url)
            if parsed.scheme != "https" or parsed.hostname not in {
                "raw.githubusercontent.com",
                "api.github.com",
            }:
                raise ValueError("Official source redirected outside its declared HTTPS hosts")
            content = response.read(MAX_SOURCE_BYTES + 1)
        if not content or len(content) > MAX_SOURCE_BYTES:
            raise ValueError("Official source is empty or exceeds its byte bound")
        content.decode("utf-8")
        path = root / name
        path.write_bytes(content)
        path.chmod(0o600)
        members.append(
            {
                "file_name": name,
                "url": url,
                "retrieved_url": final_url,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    releases = json.loads((root / "releases.json").read_bytes())
    if not isinstance(releases, list) or len(releases) > 10:
        raise ValueError("Official release index differs from the bounded list contract")
    assets = []
    for release in releases:
        if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
            raise ValueError("Official release metadata is malformed")
        for asset in release["assets"]:
            if (
                isinstance(asset, dict)
                and isinstance(asset.get("name"), str)
                and asset["name"].endswith(".pth")
            ):
                assets.append(
                    {
                        key: asset.get(key)
                        for key in ["id", "name", "size", "digest", "browser_download_url"]
                    }
                    | {"tag": release.get("tag_name"), "release_url": release.get("html_url")}
                )
    return {
        "schema_version": 1,
        "stage": "upstream_source_acquisition",
        "repository": "Megvii-BaseDetection/YOLOX",
        "code_commit": YOLOX_COMMIT,
        "members": members,
        "weight_assets": assets,
        "weight_bytes_downloaded": False,
        "weight_permission_assessed": False,
        "limitations": "Repository license and release links are retained primary records. Availability alone does not establish starting-weight permission. Inspect exact source text before selecting weights. Release-index metadata is a retrieval snapshot, not an immutable release tag.",
    }


class CollectYOLOXSources(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {}

    def outputs(self):
        return {
            name: SignalSpec.scalar(dtype="str", value_type="file", format=format)
            for name, format in [("archive_path", "zip"), ("manifest_path", "json")]
        }

    def execute(self, inputs, *, context):
        root = Path(tempfile.mkdtemp(prefix="cfu-upstream-source-", dir=Path.cwd()))
        root.chmod(0o700)
        report = collect_sources(root)
        manifest = root / "source-manifest.json"
        manifest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        manifest.chmod(0o600)
        archive = root / "upstream-sources.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for name in [*SOURCES, manifest.name]:
                output.write(root / name, name)
        archive.chmod(0o600)
        for name in SOURCES:
            (root / name).unlink()
        return {"archive_path": str(archive), "manifest_path": str(manifest)}
