"""Combine checksummed crop indexes from actual completed verification Runs."""

import hashlib
import json
from pathlib import Path
import platform

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .core import check_indices

SCHEMA = {"stage": "str", "summary": "json", "verified_inputs": "json",
          "effective_environment": "json", "scientific_acceptance_established": "bool"}


def checked_json(value, pin):
    value = value.value if isinstance(value, BioSignal) else value
    if not isinstance(value, str) or not value:
        raise ValueError("Whole-set verification requires a retained File input")
    path = Path(value)
    if path.stat().st_size != pin["size_bytes"] or not 0 < pin["size_bytes"] <= 8 * 1024 * 1024:
        raise ValueError("Whole-set verification File length changed or exceeds its bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Whole-set verification File digest changed")
    return json.loads(raw)


def combine(inputs, plan):
    if len(plan["indices"]) != 6:
        raise ValueError("Whole-set check requires exactly six actual verification indexes")
    frozen = checked_json(inputs.get("frozen_split"), plan["frozen_split"])
    indices = [checked_json(inputs.get(f"index_{i:02d}"), pin) for i, pin in enumerate(plan["indices"])]
    expected = {int(k): v for k, v in plan["expected_tiles_by_batch"].items()}
    summary = check_indices(indices, frozen["native_dataset_manifest"]["members"], expected)
    return {"stage": "whole_set_prepared_crop_identity_verification", "summary": summary,
            "verified_inputs": {"frozen_split": plan["frozen_split"], "indices": plan["indices"]},
            "effective_environment": {"python": platform.python_version()}, "scientific_acceptance_established": False}


class VerifyWholeSet(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {n: SignalSpec.scalar(dtype="str", value_type="file", format="json")
                for n in ["frozen_split"] + [f"index_{i:02d}" for i in range(6)]}

    def outputs(self):
        return {"receipt": SignalSpec.record(schema=SCHEMA, emitted_unit="1")}

    def execute(self, inputs, *, context):
        plan = json.loads((Path(__file__).resolve().parent.parent / "aggregate-plan.json").read_text())
        return {"receipt": combine(inputs, plan)}
