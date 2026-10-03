"""Uploaded-image adapter software fixtures; no learned detector or compute claim."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'models/native'
spec = importlib.util.spec_from_file_location('cfu_hosted_inference_fixture', ROOT / 'src/__init__.py',
                                            submodule_search_locations=[str(ROOT / 'src')])
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
image = __import__(spec.name + '.image', fromlist=['image'])
contract = __import__(spec.name + '.contract', fromlist=['contract'])
adapter = __import__(spec.name + '.serving', fromlist=['serving'])
inference = __import__(spec.name + '.inference', fromlist=['inference'])
core = __import__(spec.name + '.inference_core', fromlist=['inference_core'])
relative = __import__(spec.name + '.relative_area_contract', fromlist=['relative_area_contract'])


def test_filtered_release_preserves_selected_config_and_rejects_changed_ratio(bound_release):
    root, value, _ = bound_release
    value['format'] = 'cfu-locked-inference-release-v2-relative-area'
    value['configuration'] = relative.filtered_grid()[3]
    raw = json.dumps(value).encode()
    (root / 'artifacts/inference-contract.json').write_bytes(raw)
    assert contract.verify(root, image.identity(raw))[0]['configuration']['filter']['minimum_relative_area'] == .25
    value['configuration']['filter']['minimum_relative_area'] = .3
    raw = json.dumps(value).encode()
    (root / 'artifacts/inference-contract.json').write_bytes(raw)
    with pytest.raises(ValueError):
        contract.verify(root, image.identity(raw))


def test_visuals_use_verified_worker_outputs_without_second_execution(tmp_path, monkeypatch):
    from biosim.visuals import validate_visual_spec
    monkeypatch.chdir(tmp_path)
    pin = image.identity(b'synthetic release')
    release = {'weights': image.identity(b'synthetic weights'), 'configuration': relative.filtered_grid()[3]}
    monkeypatch.setattr(adapter, 'verify', lambda *_: (release, {}))
    upload = tmp_path / 'upload.png'
    Image.new('RGB', (10, 10)).save(upload)
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        request = json.loads(Path(command[-2]).read_text())
        root = Path(request['root'])
        assert root.is_relative_to(tmp_path / 'outputs')
        annotated = root / 'annotated-colonies.png'
        Image.new('RGB', (10, 74)).save(annotated, format='PNG')
        receipt = {'stage': 'actual_locked_uploaded_image_inference', 'count_unit': 'colonies',
                   'inference_contract': pin, **release, 'input': {'normalized_dimensions': [10, 10]},
                   'detections': [{'bbox_xyxy': [1., 2., 4., 5.], 'score': .8, 'class_id': 0}],
                   'warnings': ['Experimental: held-out p95 error 22.22% exceeds 20%.'],
                   'annotated_image': {**image.identity(annotated.read_bytes()), 'display_dimensions': [10, 10]}}
        Path(command[-1]).write_text(json.dumps({'count': 1, 'receipt': receipt, 'annotated_path': str(annotated)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, 'run', worker)
    obj = adapter.CountVisibleColonies(contract_sha256=pin['sha256'], contract_size_bytes=pin['size_bytes'])
    assert obj.visualize() == []
    outputs = obj.execute({'image': str(upload)}, context=None)
    visuals = obj.visualize()
    assert all(validate_visual_spec(v)[0] for v in visuals)
    assert visuals[0]['data']['source']['path'] == outputs['annotated_path']
    assert 'colonies: 1' in visuals[1]['data']['text']
    assert '22.22%' in visuals[1]['data']['text']
    assert obj.visualize() == visuals and len(calls) == 1
    with pytest.raises(ValueError):
        obj.execute({}, context=None)
    assert obj.visualize() == []






@pytest.mark.parametrize('fmt', ['JPEG', 'PNG'])
def test_decode_preserves_actual_byte_identity_and_normalized_frame(tmp_path, fmt):
    path = tmp_path / 'image'; original = Image.new('RGB', (13, 9), '#f06030')
    exif = Image.Exif(); exif[274] = 6
    original.save(path, format=fmt, exif=exif)
    decoded, metadata = image.decode(path)
    try:
        assert decoded.size == (9, 13)
        assert metadata['stored_dimensions'] == [13, 9]
        assert metadata['normalized_dimensions'] == [9, 13]
        assert metadata['exif_orientation'] == 6
        assert {k: metadata[k] for k in ['sha256', 'size_bytes']} == image.identity(path.read_bytes())
    finally: decoded.close(); original.close()


@pytest.mark.parametrize('variant', ['corrupt', 'unsupported', 'symlink', 'pixel_limit', 'animated'])
def test_unsafe_or_unsupported_image_fails_explicitly(tmp_path, monkeypatch, variant):
    path = tmp_path / 'image'
    if variant == 'corrupt': path.write_bytes(b'not an image')
    elif variant == 'unsupported': Image.new('RGB', (4, 5)).save(path, format='GIF')
    elif variant == 'animated':
        Image.new('RGB', (4, 5)).save(path, format='PNG', save_all=True,
                                    append_images=[Image.new('RGB', (4, 5), 'red')])
    else:
        Image.new('RGB', (4, 5)).save(path, format='PNG')
        if variant == 'pixel_limit': monkeypatch.setattr(image, 'MAX_PIXELS', 19)
        else: target = tmp_path / 'target'; path.rename(target); path.symlink_to(target)
    with pytest.raises((ValueError, OSError)): image.decode(path)


def test_annotated_derivative_keeps_detections_and_source_unchanged(tmp_path):
    source = Image.new('RGB', (2800, 2000), '#702018')
    detections = [{'bbox_xyxy': [4., 5., 90., 100.], 'score': .9, 'class_id': 0}]
    prior = copy.deepcopy(detections); pixels = source.tobytes(); path = tmp_path / 'annotated.png'
    receipt = image.annotate(source, detections, path)
    with Image.open(path) as saved:
        assert saved.size == (1400, 1064)
        assert saved.format == 'PNG'
    assert receipt['scale'] == .5 and receipt['size_bytes'] <= image.MAX_ANNOTATED_BYTES
    assert image.identity(path.read_bytes()) == {k: receipt[k] for k in ['sha256', 'size_bytes']}
    assert source.tobytes() == pixels and detections == prior
    source.close()


@pytest.fixture
def bound_release(tmp_path):
    root = tmp_path / 'release'; (root / 'src').mkdir(parents=True); (root / 'artifacts').mkdir()
    (root / 'src/fixture.py').write_bytes(b'# synthetic source\n')
    value = {'format': 'cfu-locked-inference-release-v1', 'configuration': core.candidate_grid()[2],
             'coordinate_basis': 'EXIF-normalized original image pixels', 'input_size': 640,
             'overlap': .2, 'batch_size': 8, 'minimum_confidence': .05, 'maximum_candidates': 20_000,
             'authored_code': {'src/fixture.py': image.identity((root / 'src/fixture.py').read_bytes())}}
    for name, (filename, _) in contract.ASSETS.items():
        payload = ('synthetic:' + name).encode(); (root / 'artifacts' / filename).write_bytes(payload)
        value[name] = image.identity(payload)
    raw = json.dumps(value).encode(); (root / 'artifacts/inference-contract.json').write_bytes(raw)
    return root, value, image.identity(raw)


@pytest.mark.parametrize('mutation', ['none', 'source', 'weights', 'configuration', 'extra_code', 'symlink'])
def test_bound_release_rejects_changed_assets_and_inference_semantics(bound_release, mutation):
    root, value, pin = bound_release
    if mutation == 'source': (root / 'src/fixture.py').write_bytes(b'changed')
    elif mutation == 'weights': (root / 'artifacts/ema-weights.pth').write_bytes(b'changed')
    elif mutation == 'configuration':
        value['configuration']['confidence'] = .333
        raw = json.dumps(value).encode(); (root / 'artifacts/inference-contract.json').write_bytes(raw)
        pin = image.identity(raw)
    elif mutation == 'extra_code': (root / 'src/extra.py').write_bytes(b'# changed bundle')
    elif mutation == 'symlink':
        path = root / 'artifacts/ema-weights.pth'; path.rename(root / 'weights'); path.symlink_to(root / 'weights')
    if mutation == 'none': assert contract.verify(root, pin)[0] == value
    else:
        with pytest.raises(ValueError): contract.verify(root, pin)


class Tensor:
    def __init__(self, array): self.array = np.asarray(array)
    @property
    def ndim(self): return self.array.ndim
    @property
    def shape(self): return self.array.shape
    def cuda(self): return self
    def cpu(self): return self
    def tolist(self): return self.array.tolist()


class FakeTorch:
    from_numpy = staticmethod(Tensor)
    inference_mode = staticmethod(contextlib.nullcontext)
    isfinite = staticmethod(lambda x: np.isfinite(x.array))


def test_synthetic_tiled_transport_preserves_bgr_batch_geometry_and_scores():
    pixels = np.zeros((80, 1000, 3), dtype=np.uint8); pixels[:, :, 0] = 17
    def preproc(crop, size):
        assert size == (640, 640) and crop[0, 0, 0] == 17
        return np.zeros((3, 640, 640), dtype=np.float32), 1.
    def slices(**kwargs):
        assert kwargs['overlap_width_ratio'] == .2 and kwargs['auto_slice_resolution'] is False
        return [[0, 0, 640, 80], [360, 0, 1000, 80]]
    def model(batch):
        assert batch.shape == (2, 3, 640, 640)
        return Tensor([[[20, 30, 10, 20, .8, .5]], [[20, 30, 10, 20, .8, .5]]])
    detections = inference.decoded_predictions(pixels, model, {'mode': 'tiled'}, preproc,
                                               slices, np, FakeTorch, time.monotonic() + 60)
    assert [d['bbox_xyxy'] for d in detections] == [[15., 20., 25., 40.], [375., 20., 385., 40.]]
    assert [d['score'] for d in detections] == [.4, .4]


@pytest.mark.parametrize('variant', ['timeout', 'nonfinite', 'batch_mismatch', 'candidate_limit'])
def test_invalid_native_predictions_never_produce_a_default_count(variant):
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    outputs = [[[5., 5., 2., 2., .8, .8]]]
    if variant == 'nonfinite': outputs[0][0][0] = float('nan')
    elif variant == 'batch_mismatch': outputs = []
    elif variant == 'candidate_limit': outputs = [outputs[0] * 20_001]
    def model(batch): return Tensor(outputs if variant != 'batch_mismatch' else np.zeros((0, 1, 6)))
    deadline = time.monotonic() - 1 if variant == 'timeout' else time.monotonic() + 60
    with pytest.raises((TimeoutError, RuntimeError)):
        inference.decoded_predictions(pixels, model, {'mode': 'raw'}, lambda crop, size: (crop, 1.),
                                      None, np, FakeTorch, deadline)




@pytest.mark.parametrize('mutation', ['none', 'count', 'detections', 'image_pin', 'image_escape', 'weights'])
def test_terminal_count_detections_and_annotated_artifact_are_bound(tmp_path, mutation):
    root = tmp_path / 'task'; root.mkdir(); path = root / 'annotated.png'
    Image.new('RGB', (10, 74)).save(path, format='PNG')
    pin = image.identity(b'synthetic contract'); weights = image.identity(b'synthetic weights')
    config = core.candidate_grid()[2]; detections = [{'bbox_xyxy': [1., 2., 4., 5.], 'score': .8, 'class_id': 0}]
    release = {'weights': weights, 'configuration': config}
    receipt = {'stage': 'actual_locked_uploaded_image_inference', 'count_unit': 'colonies',
               'inference_contract': pin, 'weights': weights, 'configuration': config,
               'input': {'normalized_dimensions': [10, 10]}, 'detections': detections,
               'annotated_image': {**image.identity(path.read_bytes()), 'display_dimensions': [10, 10]}}
    outputs = {'count': 1, 'receipt': receipt, 'annotated_path': str(path)}
    if mutation == 'count': outputs['count'] = 0
    elif mutation == 'detections': receipt['detections'][0]['bbox_xyxy'][2] = 11.
    elif mutation == 'image_pin': receipt['annotated_image']['sha256'] = 'f' * 64
    elif mutation == 'image_escape': outputs['annotated_path'] = str(tmp_path / 'outside'); Path(outputs['annotated_path']).write_bytes(path.read_bytes())
    elif mutation == 'weights': receipt['weights'] = image.identity(b'changed weights')
    if mutation == 'none': assert adapter.outputs_checked(outputs, root, pin, release) == outputs
    else:
        with pytest.raises((ValueError, RuntimeError)):
            adapter.outputs_checked(outputs, root, pin, release)
