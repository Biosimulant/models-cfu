"""Verify original bytes through short-lived owner-bound links returned by MCP.

The input envelope is temporary and contains signed URLs; never commit it.
This is independent local verification of managed outputs, not a managed Run.
"""

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def verify(envelopes, inventory, evidence_dir):
    verified = []
    for item in envelopes:
        batch = item["batch"]
        metadata = item["artifact"]
        if (
            not item["download"].startswith("https://")
            or not 0 < metadata["size_bytes"] <= 48 * 1024 * 1024
        ):
            raise ValueError("Invalid bounded MCP download envelope")
        with tempfile.TemporaryDirectory(prefix="cfu-archive-verification-") as scratch:
            archive = Path(scratch) / "originals.zip"
            sha = hashlib.sha256()
            size = 0
            request = urllib.request.Request(
                item["download"],
                headers={"User-Agent": "Biosimulant-MCP-Artifact-Verification/1.0"},
            )
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                archive.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > metadata["size_bytes"]:
                        raise ValueError("Archive exceeds the pinned size")
                    sha.update(chunk)
                    output.write(chunk)
            if (size, sha.hexdigest()) != (metadata["size_bytes"], metadata["sha256"]):
                raise ValueError("Archive size or SHA-256 differs")
            members = inventory["batches"][batch]
            manifest = evidence_dir / f"acquisition-batch{batch}-manifest.json"
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
                if len(names) != len(set(names)) or set(names) != {
                    m["file_name"] for m in members
                } | {"batch-manifest.json"}:
                    raise ValueError("Archive has missing, extra or duplicate members")
                if handle.read("batch-manifest.json") != manifest.read_bytes():
                    raise ValueError(
                        "Archive manifest differs from the independently verified artifact"
                    )
                for member in members:
                    info = handle.getinfo(member["file_name"])
                    if (
                        info.file_size != member["size_bytes"]
                        or info.compress_type != zipfile.ZIP_STORED
                    ):
                        raise ValueError("Original archive member bounds differ")
                    digest, md5, count = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
                    with handle.open(info) as original:
                        while chunk := original.read(1024 * 1024):
                            count += len(chunk)
                            digest.update(chunk)
                            md5.update(chunk)
                    if (count, digest.hexdigest(), md5.hexdigest()) != (
                        member["size_bytes"],
                        member["sha256"],
                        member["publisher_md5"],
                    ):
                        raise ValueError("Original archive member identity differs")
            verified.append(
                {
                    "batch": batch,
                    "run_id": item["run_id"],
                    "artifact_id": metadata["artifact_id"],
                    "sha256": sha.hexdigest(),
                    "size_bytes": size,
                    "members_verified": len(members),
                    "verification_origin": "independent_local_verification_of_MCP_retrieved_bytes",
                }
            )
    return verified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("envelopes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        records = verify(
            json.loads(args.envelopes.read_text()),
            json.loads((root.parent / "labs/acquisition/models/acquire/inventory.json").read_text()),
            root / "evidence",
        )
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "archives_verified": len(records),
                    "members_verified": sum(r["members_verified"] for r in records),
                }
            )
        )
    finally:
        args.envelopes.unlink(missing_ok=True)
