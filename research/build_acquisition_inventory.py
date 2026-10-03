"""Package a publisher-verified local preflight into a pinned managed acquisition inventory."""

import json
from pathlib import Path


def build(source_dir: Path, destination: Path):
    source = json.loads((source_dir / "source.json").read_text())
    frozen = json.loads((source_dir / "dataset-manifest.json").read_text())
    by_id = {file["id"]: file for file in source["files"]}
    selected = [frozen["annotations"], *frozen["images"]]
    batches = [[]]
    size = 0
    for member in selected:
        if size + member["size_bytes"] > 40 * 1024 * 1024:
            batches.append([])
            size = 0
        publisher = by_id[member["publisher_file_id"]]
        if (
            publisher["size"] != member["size_bytes"]
            or publisher["computed_md5"] != member["publisher_md5"]
        ):
            raise ValueError("Publisher identity differs from the frozen preflight")
        batches[-1].append({**member, "download_url": publisher["download_url"]})
        size += member["size_bytes"]
    inventory = {
        key: frozen[key]
        for key in (
            "source",
            "source_snapshot_sha256",
            "license",
            "attribution",
            "split_seed",
            "split_unit",
        )
    }
    inventory["batches"] = batches
    destination.write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n")
    return len(batches)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build(args.source_dir, args.destination))
