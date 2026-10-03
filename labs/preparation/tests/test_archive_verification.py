"""Software transport/integrity fixtures, never biological qualification."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_prepare_tiles import fixture, prepare

SRC = Path(__file__).resolve().parents[1] / "models/verification/src"
spec = importlib.util.spec_from_file_location("cfu_archive_fixture", SRC / "__init__.py", submodule_search_locations=[str(SRC)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
verification = importlib.import_module("cfu_archive_fixture.verify")


def inputs_for(tmp_path):
    config, source, output = fixture(tmp_path)
    archive, manifest = prepare.prepare_batch(config, 0, source, output)
    import hashlib
    config["verification_batches"] = [[0]]
    config["archives"] = {"0": {"run_id": "synthetic-run", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                               "size_bytes": archive.stat().st_size,
                               "manifest": {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                            "size_bytes": manifest.stat().st_size}}}
    inputs = {"frozen_split": source["frozen_split"], "annotations": source["annotations"], "archive_00": str(archive)}
    root = tmp_path / "verification"
    root.mkdir(mode=0o700)
    return config, inputs, root


def test_finite_typed_transport_emits_computed_receipt_and_actual_hash_index(tmp_path, monkeypatch):
    from biosimulant import BioModule, BioWorld, ExecutionPolicy

    config, inputs, root = inputs_for(tmp_path)
    model = tmp_path / "model"
    (model / "src").mkdir(parents=True)
    (model / "verification-plan.json").write_text(json.dumps(config))
    monkeypatch.setattr(verification, "__file__", str(model / "src/verify.py"))
    monkeypatch.chdir(tmp_path)
    component = verification.VerifyPreparedArchives()

    class Source(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

        def inputs(self):
            return {}

        def outputs(self):
            return {k: component.inputs()[k] for k in inputs}

        def execute(self, inputs, *, context):
            return dict(values)

    values = inputs
    world = BioWorld(communication_step=.01)
    world.add_biomodule("source", Source())
    world.add_biomodule("verify", component)
    for name in inputs:
        world.connect("source." + name, "verify." + name)
    world.run(duration=.01)
    outputs = world.get_outputs("verify")
    receipt = outputs["receipt"].value
    assert receipt["batches"][0]["source_plates"] == 1
    assert receipt["batches"][0]["derived_tiles"] == 4
    assert receipt["test_outcomes_inspected"] is False and receipt["training_executed"] is False
    assert receipt["scientific_acceptance_established"] is False
    index = json.loads(Path(outputs["tile_hash_index_path"].value).read_text())
    assert len(index) == 4 and {t["parent_file_id"] for t in index} == {"synthetic-train"}


@pytest.mark.parametrize("tamper", ["archive", "split", "manifest", "batch", "unused"])
def test_changed_unconsumed_or_unplanned_inputs_cannot_emit_a_verification_receipt(tmp_path, tamper):
    config, inputs, root = inputs_for(tmp_path)
    batch = 0
    if tamper == "archive":
        Path(inputs["archive_00"]).write_bytes(b"changed")
    elif tamper == "split":
        Path(inputs["frozen_split"]).write_bytes(b"changed")
    elif tamper == "manifest":
        config["archives"]["0"]["manifest"]["sha256"] = "f" * 64
    elif tamper == "batch":
        batch = 1
    else:
        inputs["archive_01"] = inputs["archive_00"]
    with pytest.raises((ValueError, AssertionError, KeyError)):
        verification.check(inputs, config, batch, root)
    assert not (root / "verified-tile-hash-index.json").exists()
