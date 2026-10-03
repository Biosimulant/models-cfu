"""Native parent lineage needs every original of each consumed group."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("batch_fixture", Path(__file__).resolve().parents[1] / "models/tiles/src/batches.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def member(name, group, split="train"):
    return {"sample_id": name, "group_id": group, "split": split, "role": "image"}


def test_connected_groups_remain_whole_once_and_planner_is_order_independent():
    members = [member("pilot", "p"), member("a", "g"), member("b", "g"),
               member("c", "h", "validation"), member("d", "i"), member("held", "t", "test")]
    batches = module.grouped_batches(members, ["pilot"], maximum=3)
    assert batches == module.grouped_batches(list(reversed(members)), ["pilot"], maximum=3)
    flat = [name for batch in batches for name in batch]
    assert sorted(flat) == ["a", "b", "c", "d", "pilot"]
    assert next(batch for batch in batches if "a" in batch) == next(batch for batch in batches if "b" in batch)
    assert all(len(batch) <= 3 for batch in batches)


@pytest.mark.parametrize("pilot", [["a"], ["held"], ["p", "p"], []])
def test_partial_group_or_invalid_pilot_rejected(pilot):
    members = [member("p", "p"), member("a", "g"), member("b", "g"), member("held", "t", "test")]
    with pytest.raises(ValueError):
        module.grouped_batches(members, pilot)


def test_oversized_parent_group_cannot_be_silently_split():
    members = [member("p", "p")] + [member(str(i), "large") for i in range(4)]
    with pytest.raises(ValueError, match="exceeds"):
        module.grouped_batches(members, ["p"], maximum=3)
