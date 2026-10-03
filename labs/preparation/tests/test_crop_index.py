"""Synthetic identity/coverage fixtures; never CFU performance evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("index_fixture", Path(__file__).resolve().parents[1] / "models/verification/src/index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


def fixture():
    originals = [{"sample_id": "a", "file_id": "original-a", "group_id": "group-a", "split": "train", "role": "image"},
                 {"sample_id": "b", "file_id": "original-b", "group_id": "group-b", "split": "validation", "role": "image"},
                 {"sample_id": "held", "file_id": "original-held", "group_id": "group-held", "split": "test", "role": "image"}]
    rows = [[{"sample_id": "a", "parent_file_id": "original-a", "group_id": "group-a", "split": "train",
              "file_name": "a-0.jpg", "sha256": "a" * 64, "size_bytes": 100, "preparation_batch": 0}],
            [{"sample_id": "b", "parent_file_id": "original-b", "group_id": "group-b", "split": "validation",
              "file_name": "b-0.jpg", "sha256": "b" * 64, "size_bytes": 200, "preparation_batch": 1}]]
    return rows, originals


def test_reordered_indices_keep_exact_original_coverage_and_whole_set_hash_checks():
    rows, originals = fixture()
    result = index.check_indices(rows, originals, {0: 1, 1: 1})
    assert result == index.check_indices(list(reversed(rows)), originals, {0: 1, 1: 1})
    assert result["source_plates"] == 2 and result["derived_image_bytes"] == 300
    assert result["test_outcomes_inspected"] is False


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "hash_leak", "parent", "group", "split", "batch", "count"])
def test_leakage_changed_lineage_or_incomplete_original_coverage_cannot_pass(tamper):
    rows, originals = fixture()
    expected = {0: 1, 1: 1}
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif tamper == "hash_leak":
        rows[1][0]["sha256"] = rows[0][0]["sha256"]
    elif tamper == "parent":
        rows[0][0]["parent_file_id"] = "other"
    elif tamper in {"group", "split"}:
        rows[0][0]["group_id" if tamper == "group" else "split"] = "other"
    elif tamper == "batch":
        rows[0][0]["preparation_batch"] = 999
    else:
        expected[0] = 2
    with pytest.raises(ValueError):
        index.check_indices(rows, originals, expected)
