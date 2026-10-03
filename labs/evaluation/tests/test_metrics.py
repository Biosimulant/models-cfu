"""Synthetic mathematical checks, never measured source-domain colony accuracy."""

import copy
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/metrics/src/metrics.py"
spec = importlib.util.spec_from_file_location("cfu_metrics_fixture", SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def record(name="synthetic", references=100, predicted=100, group="synthetic-group"):
    boxes = [[(i % 50) * 10, (i // 50) * 10, (i % 50) * 10 + 5, (i // 50) * 10 + 5]
             for i in range(max(references, predicted))]
    return {"sample_id": name, "group_id": group, "status": "completed", "image_width": 1000, "image_height": 1000,
            "reference_boxes_xyxy": boxes[:references], "predicted_count": predicted,
            "detections": [{"bbox_xyxy": b, "score": .9} for b in boxes[:predicted]]}


def report(records):
    return metrics.summarize(records, expected_sample_ids=[r["sample_id"] for r in records], bootstrap_replicates=100)


def test_one_to_one_matching_finds_an_augmenting_assignment():
    references = [[0, 0, 10, 10], [4, 0, 14, 10]]
    predictions = [{"bbox_xyxy": [1, 0, 11, 10], "score": .9}, {"bbox_xyxy": [0, 0, 10, 10], "score": .8}]
    matched = metrics.match_detections(predictions, references)
    assert {(m["prediction_index"], m["reference_index"]) for m in matched} == {(0, 1), (1, 0)}
    assert all(m["iou"] >= .5 for m in matched)


def test_unmodified_edge_reference_and_zero_area_labels_keep_count_and_miss_denominators():
    value = record(references=0, predicted=0)
    value["reference_boxes_xyxy"] = [[-7, 20, 26, 61], [50, 50, 58, 50]]
    value["detections"] = [{"bbox_xyxy": [0, 20, 26, 61], "score": .9}]
    value["predicted_count"] = 1
    measured = metrics.plate_metrics(value)
    assert measured["reference_count"] == 2
    assert (measured["tp"], measured["fp"], measured["fn"]) == (1, 0, 1)
    assert measured["matches"][0]["iou"] == pytest.approx(26 / 33)
    assert value["reference_boxes_xyxy"] == [[-7, 20, 26, 61], [50, 50, 58, 50]]
    assert metrics.iou([50, 50, 58, 50], [50, 50, 58, 50]) == 0


def test_negative_or_zero_area_model_detections_are_still_rejected():
    for bbox in [[-7, 20, 26, 61], [50, 50, 58, 50]]:
        value = record(references=1, predicted=1)
        value["detections"][0]["bbox_xyxy"] = bbox
        with pytest.raises(ValueError):
            metrics.plate_metrics(value)


def test_repeated_detection_is_a_false_positive_and_changes_the_count():
    value = record(references=1, predicted=1)
    value["detections"].append(copy.deepcopy(value["detections"][0]))
    value["predicted_count"] = 2
    measured = metrics.plate_metrics(value)
    assert (measured["tp"], measured["fp"], measured["fn"]) == (1, 1, 0)
    assert measured["absolute_error"] == 1
    assert measured["symmetric_error"] == pytest.approx(2/3)


def test_weighted_count_error_and_matching_use_whole_plate_denominators():
    measured = report([record("a", references=100, predicted=90), record("b", references=10, predicted=9)])
    assert measured["reference_colonies"] == 110 and measured["absolute_count_error"] == 11
    assert measured["wape"] == .1 and measured["mae"] == 5.5 and measured["signed_bias"] == -5.5
    assert measured["recall"] == .9 and measured["precision"] == 1
    assert measured["acceptance_passed"] is True


def test_bootstrap_replays_exactly_and_resamples_related_groups_together():
    records = [record("a", 10, 9, "related"), record("b", 100, 90, "related"), record("c", 50, 50, "independent")]
    first = report(records)
    assert first == report(list(reversed(records)))
    assert first["bootstrap"] == {"unit": "frozen related-image group", "seed": 20261001, "replicates": 100, "defined_replicates": 100}
    assert first["bootstrap_wape_95pct"] == pytest.approx([0, .1])


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "nonterminal", "count_mismatch", "score", "coordinates"])
def test_incomplete_or_inconsistent_evaluation_cannot_qualify(change):
    records = [record()]
    if change == "missing":
        records = []
    elif change == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif change == "extra":
        records.append(record("unplanned"))
    elif change == "nonterminal":
        records[0]["status"] = "running"
    elif change == "count_mismatch":
        records[0]["predicted_count"] += 1
    elif change == "score":
        records[0]["detections"][0]["score"] = float("nan")
    else:
        records[0]["detections"][0]["bbox_xyxy"] = [0, 0, 2000, 5]
    with pytest.raises(ValueError):
        metrics.summarize(records, expected_sample_ids=["synthetic"], bootstrap_replicates=10)


def test_failures_remain_in_denominators_without_a_fabricated_zero_prediction():
    failed = record("failed", references=10, predicted=0)
    failed.update(status="failed", predicted_count=None, failure={"code": "synthetic-inference-failure"})
    measured = report([record("completed", references=100, predicted=100), failed])
    assert measured["reference_colonies"] == 110 and measured["absolute_count_error"] == 10
    assert measured["failures"] == 1 and measured["fn"] == 10
    assert measured["wape"] == pytest.approx(10/110)
    assert measured["signed_bias"] is None and measured["acceptance_passed"] is False
    assert measured["plates"][1]["predicted_count"] is None


def test_empty_reference_data_cannot_prove_the_detection_or_wape_gates():
    measured = report([record(references=0, predicted=0)])
    assert measured["wape"] is None and measured["precision"] is None and measured["recall"] is None
    assert measured["p95_symmetric_error"] == 0
    assert measured["acceptance_passed"] is False


def test_percentile_uses_the_frozen_linear_convention():
    assert metrics.percentile([0, 1, 2, 3, 4], .95) == pytest.approx(3.8)


def test_recreation_at_two_percentage_points_passes_without_binary_rounding_error():
    selected, recreated = report([record(predicted=90)]), report([record(predicted=92)])
    compared = metrics.recreation_check(selected, recreated)
    assert compared["wape_difference_percentage_points"] == 2
    assert compared["both_models_passed"] is True and compared["recreation_passed"] is True
    assert metrics.recreation_check(selected, report([record(predicted=93)]))["recreation_passed"] is False


def test_recreation_requires_same_plates_and_both_models_to_pass():
    selected = report([record(predicted=90)])
    assert metrics.recreation_check(selected, report([record(predicted=80)]))["recreation_passed"] is False
    changed = report([record("other", predicted=90)])
    with pytest.raises(ValueError, match="frozen evaluation plates"):
        metrics.recreation_check(selected, changed)


def test_saved_model_replay_requires_identical_counts_and_all_aggregate_metrics():
    selected = report([record(predicted=90)])
    assert metrics.replay_check(selected, copy.deepcopy(selected))["replay_comparison_passed"] is True
    altered = copy.deepcopy(selected)
    altered["recall"] = .899
    assert metrics.replay_check(selected, altered)["differing_aggregate_fields"] == ["recall"]
    assert metrics.replay_check(selected, report([record(predicted=91)]))["replay_comparison_passed"] is False
