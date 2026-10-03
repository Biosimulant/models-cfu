"""Target/identity fixtures, not observed native training or GPU performance."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import zipfile

from PIL import Image
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "models/native/src"
PACKAGE = "cfu_native_training_data_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data, benchmark = load("data"), load("benchmark")


def pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def tile(count=128):
    return {"split": "train", "width": 640, "height": 640,
            "targets": [{"source_annotation_id": i, "class_id": 0, "bbox": [20, 30, 10, 8]} for i in range(count)]}


def archive_fixture(tmp_path, *, mutation=None):
    image = Image.new("RGB", (640, 640), (210, 90, 40))
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=95, subsampling=0)
    raw = stream.getvalue()
    t = {**tile(), "file_name": "crops/example.jpg", **pin(raw), "sample_id": "original.jpg",
         "parent_file_id": "original-file", "parent_sha256": "a" * 64, "group_id": "original-group"}
    parent = {"file_id": "original-file", "sample_id": "original.jpg", "sha256": "a" * 64,
              "group_id": "original-group", "split": "train", "role": "image"}
    manifest = {"batch_index": 12, "frozen_dataset_id": "frozen", "frozen_dataset_sha256": "b" * 64,
                "source_annotations_modified": False, "test_targets_inspected_or_emitted": False,
                "derived_image_count": 1, "plates": [{"tiles": [t]}]}
    if mutation:
        mutation(manifest, t)
    manifest_raw = json.dumps(manifest).encode()
    archive_path = tmp_path / "prepared.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("prepared-batch.json", manifest_raw)
        z.writestr("tiles/" + t["file_name"], raw)
    frozen_raw = json.dumps({"native_dataset_manifest": {"members": [parent]}}).encode()
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_bytes(frozen_raw)
    plan = {"archive": pin(archive_path.read_bytes()), "manifest": pin(manifest_raw), "frozen_split": pin(frozen_raw),
            "preparation_batch": 12, "frozen_dataset_id": "frozen", "frozen_dataset_sha256": "b" * 64}
    return archive_path, frozen_path, plan


def test_dense_original_annotations_all_enter_native_loss_without_default50_cap():
    rows = data.loss_targets(tile(128))
    assert len(rows) == 128
    assert rows[-1] == [0, 25, 34, 10, 8]


def test_empty_training_crop_has_no_fabricated_detection_target():
    assert data.loss_targets(tile(0)) == []


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_optimizer_cannot_receive_nontraining_targets(partition):
    t = tile()
    t["split"] = partition
    with pytest.raises(ValueError, match="training partition"):
        data.loss_targets(t)


@pytest.mark.parametrize("bbox", [[0, 0, 0, 5], [0, 0, -1, 5], [-1, 0, 5, 5], [639, 0, 2, 5], [0, float("nan"), 2, 5]])
def test_invalid_geometry_cannot_be_silently_removed(bbox):
    t = tile(1)
    t["targets"][0]["bbox"] = bbox
    with pytest.raises(ValueError):
        data.loss_targets(t)


def test_one_pixel_prepared_target_is_preserved():
    t = tile(1)
    t["targets"][0]["bbox"] = [639, 639, 1, 1]
    assert data.loss_targets(t) == [[0, 639.5, 639.5, 1, 1]]


def test_repeated_annotation_and_target_overflow_fail_instead_of_truncation():
    t = tile(2)
    t["targets"][1] = copy.deepcopy(t["targets"][0])
    with pytest.raises(ValueError):
        data.loss_targets(t)
    with pytest.raises(ValueError, match="never truncate"):
        data.loss_targets(tile(data.MAX_TARGETS + 1))


def test_retained_archive_crop_and_frozen_parent_are_verified(tmp_path):
    archive, frozen, plan = archive_fixture(tmp_path)
    value = data.TrainingArchive(archive, frozen, plan)
    try:
        assert len(value.tiles) == 1
        assert len(data.loss_targets(value.tiles[0])) == 128
        with value.image(value.tiles[0]) as image:
            assert image.size == (640, 640)
            assert image.mode == "RGB"
    finally:
        value.close()


@pytest.mark.parametrize("mutation", [
    lambda m, t: t.update(split="validation"),
    lambda m, t: t.update(split="test"),
    lambda m, t: t.update(parent_file_id="unrelated-file"),
    lambda m, t: t.update(sha256="c" * 64),
    lambda m, t: m.update(batch_index=13),
    lambda m, t: m.update(source_annotations_modified=True),
    lambda m, t: t.update(file_name="../escape.jpg"),
])
def test_changed_geometry_lineage_partition_or_bytes_cannot_enter_training(tmp_path, mutation):
    archive, frozen, plan = archive_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError):
        data.TrainingArchive(archive, frozen, plan)


def test_managed_contract_is_finite_typed_and_has_no_test_or_checkpoint_input():
    module = benchmark.BenchmarkNativeDetector(timed_steps=4)
    assert set(module.inputs()) == {"source_archive", "source_receipt", "prepared_archive", "frozen_split"}
    assert set(module.outputs()) == {"receipt"}
    assert module.execution_policy.value == "once_before_run"


def test_native_tensor_packing_preserves_bgr_pixels_padding_dense_and_empty_labels():
    dense, empty = tile(128), tile(0)
    for t in (dense, empty):
        t.update(width=3, height=2)
        for target in t["targets"]:
            target["bbox"] = [0, 0, 1, 1]

    class Crops:
        tiles = [dense, empty]

        def image(self, _):
            return Image.new("RGB", (3, 2), (210, 90, 40))

    class ArrayTransfer:
        # This substitutes only the CUDA transfer, never a scientific prediction.
        def __init__(self, value):
            self.value = value

        def cuda(self):
            return self.value

    fake_torch = types.SimpleNamespace(from_numpy=ArrayTransfer)
    images, labels, counts = benchmark.tensors(Crops(), 2, fake_torch, np)
    assert images.shape == (2, 3, 640, 640) and images.dtype == np.float32
    assert images[0, :, 0, 0].tolist() == [40, 90, 210]
    assert images[0, :, 2, 3].tolist() == [114, 114, 114]
    assert labels.shape == (2, 128, 5) and labels.dtype == np.float32
    assert labels[0, -1].tolist() == [0, 0.5, 0.5, 1, 1]
    assert np.count_nonzero(labels[1]) == 0 and counts == [128, 0]


@pytest.mark.parametrize("steps", [True, 3, 31, 4.0])
def test_benchmark_cannot_expand_its_fixed_step_bound(steps):
    with pytest.raises(ValueError):
        benchmark.BenchmarkNativeDetector(timed_steps=steps)


def test_validation_only_archive_is_verified_without_entering_the_training_pool(tmp_path):
    archive,frozen,plan=archive_fixture(tmp_path,mutation=lambda manifest,t:t.update(split='validation'))
    members=json.loads(frozen.read_bytes())
    members['native_dataset_manifest']['members'][0]['split']='validation'
    raw=json.dumps(members).encode();frozen.write_bytes(raw);plan['frozen_split']=pin(raw)
    with pytest.raises(ValueError,match='no frozen training crops'):
        data.TrainingArchive(archive,frozen,plan)
    checked=data.TrainingArchive(archive,frozen,plan,require_training=False)
    assert checked.tiles==[] and checked.total_tiles==1
    checked.close()
