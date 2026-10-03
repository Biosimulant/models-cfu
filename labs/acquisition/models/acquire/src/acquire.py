"""Finite, checksum-pinned ADBC acquisition; no annotation outcomes are read."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from biosimulant import BioModule, ExecutionPolicy, SignalSpec


class AcquireADBC(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, batch_index: int = 0):
        self.batch_index = int(batch_index)

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "archive_path": SignalSpec.scalar(dtype="str", value_type="file", format="zip"),
            "manifest_path": SignalSpec.scalar(dtype="str", value_type="file", format="json"),
        }

    def execute(self, inputs, *, context):
        inventory = json.loads((Path(__file__).resolve().parents[1] / "inventory.json").read_text())
        if not 0 <= self.batch_index < len(inventory["batches"]):
            raise ValueError("Unknown immutable acquisition batch")
        batch = inventory["batches"][self.batch_index]
        root = Path(tempfile.mkdtemp(prefix=f"cfu-acquisition-{self.batch_index}-", dir=Path.cwd()))
        retained = []
        for member in batch:
            name = member["file_name"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Unsafe publisher file name")
            url = member["download_url"]
            if not url.startswith("https://ndownloader.figshare.com/files/"):
                raise ValueError("Source is not the pinned publisher download endpoint")
            path = root / name
            for attempt in range(3):
                try:
                    sha = hashlib.sha256()
                    md5 = hashlib.md5(usedforsecurity=False)
                    size = 0
                    with (
                        urllib.request.urlopen(url, timeout=90) as response,
                        path.open("wb") as handle,
                    ):
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > member["size_bytes"]:
                                raise ValueError("Publisher bytes exceed pinned size")
                            sha.update(chunk)
                            md5.update(chunk)
                            handle.write(chunk)
                    if (size, sha.hexdigest(), md5.hexdigest()) != (
                        member["size_bytes"],
                        member["sha256"],
                        member["publisher_md5"],
                    ):
                        raise ValueError("Publisher size or checksum changed")
                    retained.append({k: v for k, v in member.items() if k != "download_url"})
                    break
                except Exception:
                    path.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        manifest = root / "batch-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "acquisition",
                    "batch_index": self.batch_index,
                    "source": inventory["source"],
                    "source_version": 3,
                    "source_snapshot_sha256": inventory["source_snapshot_sha256"],
                    "license": inventory["license"],
                    "attribution": inventory["attribution"],
                    "split_seed": inventory["split_seed"],
                    "split_unit": inventory["split_unit"],
                    "near_duplicate_review": "Pending; training must not start before review.",
                    "members": retained,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        archive = root / "dataset.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
            for member in retained:
                handle.write(root / member["file_name"], member["file_name"])
            handle.write(manifest, "batch-manifest.json")
        if archive.stat().st_size > 48 * 1024 * 1024:
            raise ValueError("Acquisition archive exceeds the retained artifact budget")
        for member in retained:
            (root / member["file_name"]).unlink()
        return {"archive_path": str(archive), "manifest_path": str(manifest)}
