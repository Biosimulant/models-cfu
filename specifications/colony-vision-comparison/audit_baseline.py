"""Read-only recovery of retained evidence; never runs a model or an API call.

Run from staging. Outputs are private and must stay outside Git.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

IDENTITIES = {
    "split": ("verified-frozen-original-split.json", 237988, "81156e7833415b05b39659828d3fc4b8b006b6f2b24e1d4665de654efae6ae3f"),
    "annotations": ("verified-original-annotations.json", 6386192, "062f08c159004e89b96f93bea05d95403b01b1fcfb73384f6eda21ba9842e12c"),
    "predictions": ("native-primary-locked-test-verified-predictions_path-20261003.json", 1545733, "77776df3cf307a9a53bcb89509fa1d1124b42dae47fdc893ee7e719857fe54c9"),
    "summary": ("native-primary-locked-test-verified-comparison_path-20261003.json", 831437, "f145b91baac46c0387911d109b672e04dc8d88e0b3d7276efb3739d405588a44"),
    "results": ("native-primary-locked-test-verified-results-20261003.json", 481864, "05842beecb2326870b1f03180a5b1981dd65d0ace27f084aee4fc285879da602"),
}


def verified_json(path, size, digest):
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Evidence identity mismatch: {path.name}")
    return json.loads(raw)


def stratum(count):
    return "sparse" if count <= 30 else "moderate" if count <= 300 else "dense"


def selection(items, per_stratum, key):
    selected = []
    for density in ("sparse", "moderate", "dense"):
        pool = sorted((x for x in items if x["stratum"] == density),
                      key=lambda x: hashlib.sha256((key + ":" + x["sample_id"]).encode()).hexdigest())
        seen = set()
        for item in pool:
            if item["group_id"] not in seen:
                selected.append(item["sample_id"])
                seen.add(item["group_id"])
                if len(seen) == per_stratum:
                    break
        if len(seen) < per_stratum:
            raise ValueError(f"Insufficient distinct {density} groups")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output_dir.resolve().is_relative_to(repo):
        raise ValueError("Private evidence outputs must be outside the repository")
    objects = {key: verified_json(args.evidence_dir / name, size, sha)
               for key, (name, size, sha) in IDENTITIES.items()}
    split, coco, predictions = (objects[k] for k in ("split", "annotations", "predictions"))
    members = {m["sample_id"]: m for m in split["native_dataset_manifest"]["members"]}
    expected = sorted(m["sample_id"] for m in members.values() if m.get("split") == "test" and m["role"] == "image")
    groups = {name: g["group_id"] for g in split["groups"] for name in g["samples"]}
    images = {i["file_name"]: i for i in coco["images"]}
    references = defaultdict(list)
    for annotation in coco["annotations"]:
        x, y, w, h = annotation["bbox"]
        references[annotation["image_id"]].append([x, y, x + w, y + h])
    plates = predictions["plates"]
    assert len(expected) == 37 and sorted(p["sample_id"] for p in plates) == expected
    for plate in plates:
        image = images[plate["sample_id"]]
        assert plate["group_id"] == groups[plate["sample_id"]] == members[plate["sample_id"]]["group_id"]
        assert (plate["image_width"], plate["image_height"]) == (image["width"], image["height"])
        assert plate["reference_boxes_xyxy"] == references[image["id"]]
    metrics_path = repo / "labs/evaluation/models/native-test/src/metrics.py"
    metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert metrics_sha == predictions["evaluation_bundle"]["authored_code"]["src/metrics.py"]["sha256"]
    module_spec = importlib.util.spec_from_file_location("frozen_metrics", metrics_path)
    metrics = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metrics)
    report = metrics.summarize(plates, expected_sample_ids=expected)
    assert report == predictions["metrics"] == objects["summary"]["metrics"]
    assert report["reference_colonies"] == 6409
    inventory = []
    for plate in sorted(plates, key=lambda x: x["sample_id"]):
        member = members[plate["sample_id"]]
        inventory.append({**member, "width": plate["image_width"], "height": plate["image_height"],
                          "reference_count": len(plate["reference_boxes_xyxy"]),
                          "stratum": stratum(len(plate["reference_boxes_xyxy"])),
                          "deployed_input_eligible": plate["image_width"] * plate["image_height"] <= 25_000_000,
                          "original_image_bytes_verified_in_this_audit": False})
    development = []
    for member in members.values():
        if member.get("split") == "validation" and member["role"] == "image":
            image = images[member["sample_id"]]
            development.append({**member, "width": image["width"], "height": image["height"],
                                "stratum": stratum(len(references[image["id"]]))})
    subsets = {"development": selection(development, 2, "cfu-gpt-dev-v1"),
               "repeatability": selection(inventory, 3, "cfu-gpt-repeat-v1"),
               "localization": selection(inventory, 3, "cfu-gpt-localize-v1")}
    rows = []
    for plate in report["plates"]:
        density = stratum(plate["reference_count"])
        rows.append({**{k: plate[k] for k in ["sample_id", "group_id", "status", "reference_count", "predicted_count", "absolute_error", "signed_error", "symmetric_error", "tp", "fp", "fn"]},
                     "stratum": density, "gpt_count_status": "not_run", "gpt_count": None,
                     "gpt_localization_status": "not_run", "gpt_cost_usd": None, "gpt_elapsed_seconds": None})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    write("test-inventory.json", {"schema_version": 1, "provenance": IDENTITIES, "plates": inventory, "subsets": subsets,
                                  "development_inputs": [x for x in development if x["sample_id"] in subsets["development"]]})
    write("baseline-audit.json", {"schema_version": 1, "evidence_identity_checks": "passed", "source_reference_equality": "passed",
                                  "metric_recomputation": "exactly_equal", "metrics_sha256": metrics_sha,
                                  "metrics": {k: v for k, v in report.items() if k != "plates"},
                                  "limits": "No image-byte audit, inference, API call, managed execution or external scientific review."})
    with (args.output_dir / "per-plate-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"plates": len(rows), "reference_colonies": report["reference_colonies"],
                      "metrics_reproduced": True, "metrics_sha256": metrics_sha, "subsets": subsets,
                      "oversized": [x["sample_id"] for x in inventory if not x["deployed_input_eligible"]]}))


if __name__ == "__main__":
    main()
