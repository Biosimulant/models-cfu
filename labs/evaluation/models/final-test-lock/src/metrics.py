"""Prespecified whole-plate count/detection metrics, independent of detector code.

Inputs must be retained whole-plate predictions, including explicit failures.
Software fixtures exercising this module do not establish CFU performance.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from fractions import Fraction

METHOD = "cfu-whole-plate-metrics-v2"
IOU_THRESHOLD = 0.5
GATES = {"wape": 0.10, "p95_symmetric_error": 0.20, "precision": 0.90, "recall": 0.90}


def box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
    ):
        raise ValueError("Bounding box must contain four finite original-frame xyxy coordinates")
    x0, y0, x1, y1 = map(float, value)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Bounding box must have positive area within the original frame")
    return x0, y0, x1, y1



def reference_box(value):
    """Unmodified finite source xyxy; zero-area labels remain count/FN references."""
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
    ):
        raise ValueError("Source reference must contain four finite coordinates")
    x0, y0, x1, y1 = map(float, value)
    if x1 < x0 or y1 < y0:
        raise ValueError("Source reference cannot have negative side lengths")
    return x0, y0, x1, y1

def iou(left, right):
    a, b = reference_box(left), reference_box(right)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def cells(value):
    x0, y0, x1, y1 = value
    for y in range(math.floor(y0 / 64), math.floor(y1 / 64) + 1):
        for x in range(math.floor(x0 / 64), math.floor(x1 / 64) + 1):
            yield x, y


def match_detections(detections, references):
    """Deterministic maximum-cardinality one-to-one matches at the fixed IoU gate."""
    truth = [reference_box(b) for b in references]
    predicted = [box(d["bbox_xyxy"]) for d in detections]
    grid = defaultdict(set)
    for index, value in enumerate(truth):
        if value[0] == value[2] or value[1] == value[3]:
            continue
        for cell in cells(value):
            grid[cell].add(index)
    adjacency = []
    for value in predicted:
        candidates = set()
        for cell in cells(value):
            candidates.update(grid[cell])
        edges = [(iou(value, truth[j]), j) for j in candidates]
        adjacency.append([j for score, j in sorted(edges, key=lambda p: (-p[0], truth[p[1]], p[1])) if score >= IOU_THRESHOLD])
    to_truth, to_prediction = {}, {}
    order = sorted(range(len(detections)), key=lambda i: (-detections[i]["score"], predicted[i], i))
    for start in order:
        queue, seen, previous = deque([start]), {start}, {}
        endpoint = None
        while queue and endpoint is None:
            prediction = queue.popleft()
            for reference in adjacency[prediction]:
                if reference in previous:
                    continue
                previous[reference] = prediction
                matched = to_prediction.get(reference)
                if matched is None:
                    endpoint = reference
                    break
                if matched not in seen:
                    seen.add(matched)
                    queue.append(matched)
        while endpoint is not None:
            prediction = previous[endpoint]
            prior_reference = to_truth.get(prediction)
            to_truth[prediction] = endpoint
            to_prediction[endpoint] = prediction
            endpoint = prior_reference
    return [{"prediction_index": i, "reference_index": j, "iou": iou(predicted[i], truth[j])}
            for i, j in sorted(to_truth.items())]


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Percentile requires observations and a quantile in [0,1]")
    values = sorted(values)
    position = (len(values) - 1) * Fraction(str(quantile))
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (position - low) * (values[high] - values[low])


def plate_metrics(record):
    references = [reference_box(b) for b in record["reference_boxes_xyxy"]]
    detections = record["detections"]
    width, height = record["image_width"], record["image_height"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in [width, height]) or width * height > 40_000_000:
        raise ValueError("Evaluation image dimensions are outside the retained offline source contract")
    source_geometry = {
        "zero_area_reference_indices": [i for i, b in enumerate(references) if b[0] == b[2] or b[1] == b[3]],
        "edge_crossing_reference_indices": [i for i, b in enumerate(references)
                                            if b[0] < 0 or b[1] < 0 or b[2] > width or b[3] > height],
    }
    for value in [box(d["bbox_xyxy"]) for d in detections]:
        if value[2] > width or value[3] > height:
            raise ValueError("Detection coordinates exceed the original image frame")
    failed = record["status"] == "failed"
    if record["status"] not in {"completed", "failed"}:
        raise ValueError("Every attempted plate requires a terminal prediction outcome")
    if failed:
        if record.get("predicted_count") is not None or detections or not record.get("failure"):
            raise ValueError("Failed inference must retain failure evidence and no plausible default count")
        # This is an explicit analytical penalty, never an emitted prediction.
        return {"sample_id": record["sample_id"], "group_id": record["group_id"], "status": "failed",
                "predicted_count": None, "reference_count": len(references), "absolute_error": len(references),
                "signed_error": None, "symmetric_error": 2.0, "tp": 0, "fp": 0, "fn": len(references),
                "source_geometry": source_geometry,
                "failure": record["failure"], "failure_penalty": "Reference-count absolute penalty; symmetric penalty2; acceptance blocked."}
    count = record.get("predicted_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(detections):
        raise ValueError("Displayed count must equal the retained detection count")
    for detection in detections:
        box(detection["bbox_xyxy"])
        score = detection["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detection scores must be finite in [0,1]")
    matches = match_detections(detections, references)
    difference = count - len(references)
    denominator = count + len(references)
    return {"sample_id": record["sample_id"], "group_id": record["group_id"], "status": "completed",
            "predicted_count": count, "reference_count": len(references), "absolute_error": abs(difference),
            "signed_error": difference, "symmetric_error": 2 * abs(difference) / denominator if denominator else 0.0,
            "tp": len(matches), "fp": count - len(matches), "fn": len(references) - len(matches), "matches": matches,
            "source_geometry": source_geometry}


def summarize(records, *, expected_sample_ids, bootstrap_replicates=2000, seed=20261001):
    expected = list(expected_sample_ids)
    names = [r["sample_id"] for r in records]
    if not expected or len(expected) != len(set(expected)) or len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("Evaluation must retain every frozen plate exactly once, including failures")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicate count must be a positive integer")
    plates = [plate_metrics(r) for r in sorted(records, key=lambda r: r["sample_id"])]
    failures = sum(p["status"] == "failed" for p in plates)
    reference_total = sum(p["reference_count"] for p in plates)
    absolute_total = sum(p["absolute_error"] for p in plates)
    tp, fp, fn = (sum(p[k] for p in plates) for k in ["tp", "fp", "fn"])
    wape = absolute_total / reference_total if reference_total else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    symmetric_exact = [Fraction(2) if p["status"] == "failed" else
                       Fraction(2 * p["absolute_error"], p["predicted_count"] + p["reference_count"])
                       if p["predicted_count"] + p["reference_count"] else Fraction(0) for p in plates]
    p95_exact = percentile(symmetric_exact, .95)
    p95 = float(p95_exact)
    groups = defaultdict(list)
    for p in plates:
        groups[p["group_id"]].append(p)
    clusters = [groups[g] for g in sorted(groups)]
    rng, bootstrap = random.Random(seed), []
    for _ in range(bootstrap_replicates):
        sampled = [p for _ in clusters for p in clusters[rng.randrange(len(clusters))]]
        denominator = sum(p["reference_count"] for p in sampled)
        if denominator:
            bootstrap.append(sum(p["absolute_error"] for p in sampled) / denominator)
    gates = {"wape": bool(reference_total and absolute_total * 10 <= reference_total),
             "p95_symmetric_error": p95_exact <= Fraction(1, 5),
             "precision": bool(tp + fp and tp * 10 >= 9 * (tp + fp)),
             "recall": bool(tp + fn and tp * 10 >= 9 * (tp + fn))}
    return {"schema_version": 1, "method": METHOD, "iou_threshold": IOU_THRESHOLD, "source_reference_policy": "Unmodified original boxes; out-of-frame portions remain in IoU; zero-area references retain counts and are always misses.", "approved_thresholds": dict(GATES),
            "plate_count": len(plates), "reference_colonies": reference_total, "absolute_count_error": absolute_total, "failures": failures,
            "wape": wape, "mae": absolute_total / len(plates),
            "signed_bias": sum(p["signed_error"] for p in plates) / len(plates) if not failures else None,
            "p95_symmetric_error": p95, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "bootstrap_wape_95pct": [percentile(bootstrap, .025), percentile(bootstrap, .975)] if bootstrap else None,
            "bootstrap": {"unit": "frozen related-image group", "seed": seed, "replicates": bootstrap_replicates,
                          "defined_replicates": len(bootstrap)},
            "percentile_convention": "Linear interpolation at(n-1)*q", "gates": gates,
            "metric_gates_passed": all(gates.values()), "execution_evidence_complete": failures == 0,
            "wape_uses_explicit_failure_penalties": failures > 0,
            "acceptance_passed": all(gates.values()) and failures == 0,
            "plates": plates, "limits": "Annotated source-domain counts only; group bootstrap does not establish acquisition or external-lab independence."}


def recreation_check(selected, recreated):
    """Compare recorded metrics; execution/lineage must be proved by managed Runs."""
    if selected["method"] != METHOD or recreated["method"] != METHOD or selected["approved_thresholds"] != GATES or recreated["approved_thresholds"] != GATES:
        raise ValueError("Recreation comparison requires the same prespecified metrics and approved gates")
    left = [(p["sample_id"], p["group_id"], p["reference_count"]) for p in selected["plates"]]
    right = [(p["sample_id"], p["group_id"], p["reference_count"]) for p in recreated["plates"]]
    if left != right:
        raise ValueError("Recreation comparison changed the frozen evaluation plates")
    difference = abs(Fraction(selected["absolute_count_error"], selected["reference_colonies"])
                     - Fraction(recreated["absolute_count_error"], recreated["reference_colonies"])) if selected["reference_colonies"] and recreated["reference_colonies"] else None
    return {"both_models_passed": selected["acceptance_passed"] and recreated["acceptance_passed"],
            "wape_difference_percentage_points": float(difference * 100) if difference is not None else None,
            "tolerance_percentage_points": 2.0,
            "recreation_passed": bool(selected["acceptance_passed"] and recreated["acceptance_passed"]
                                      and difference is not None and difference <= Fraction(1, 50))}


def replay_check(selected, replayed):
    """Check exact saved-model count/aggregate replay, not a second training claim."""
    aggregate_keys = ["method", "iou_threshold", "approved_thresholds", "plate_count", "reference_colonies",
                      "absolute_count_error", "failures", "wape", "mae", "signed_bias", "p95_symmetric_error",
                      "tp", "fp", "fn", "precision", "recall", "bootstrap_wape_95pct", "bootstrap", "gates",
                      "metric_gates_passed", "execution_evidence_complete", "wape_uses_explicit_failure_penalties", "acceptance_passed"]
    if selected["method"] != METHOD or replayed["method"] != METHOD:
        raise ValueError("Replay comparison requires the prespecified evaluation method")
    def counts(report):
        return [(p["sample_id"], p["group_id"], p["reference_count"], p["predicted_count"], p["status"])
                for p in report["plates"]]
    differing = [key for key in aggregate_keys if selected[key] != replayed[key]]
    return {"identical_plate_counts": counts(selected) == counts(replayed), "differing_aggregate_fields": differing,
            "replay_comparison_passed": counts(selected) == counts(replayed) and not differing
                                        and selected["failures"] == 0 and replayed["failures"] == 0}
