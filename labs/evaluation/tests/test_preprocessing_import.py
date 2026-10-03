"""Import/identity software evidence, not detector-performance evidence."""
import importlib.util
import inspect
import os
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]/'models/native-validation'
spec=importlib.util.spec_from_file_location('cfu_preprocessing_import',ROOT/'src/preprocessing.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)


def test_direct_import_never_executes_native_dataset_package_initializer(tmp_path,monkeypatch):
    root=tmp_path/'verified';data=root/'yolox/data';data.mkdir(parents=True)
    (root/'yolox/__init__.py').write_text('')
    (data/'__init__.py').write_text("raise RuntimeError('Unused COCO reader must not initialize')")
    (data/'data_augment.py').write_text('def preproc(value, size):\n    return value, size\n')
    monkeypatch.syspath_prepend(str(root))
    function=loader.load_preproc(root)
    assert function('fixture',(640,640))==('fixture',(640,640))
    assert function.__code__.co_filename==str(data/'data_augment.py')


@pytest.mark.parametrize('change',['missing','link','outside_parent','oversized','no_function'])
def test_loader_only_accepts_a_bounded_owned_regular_native_file(tmp_path,change):
    root=tmp_path/'verified';data=root/'yolox/data';data.mkdir(parents=True)
    path=data/'data_augment.py'
    if change=='link':
        other=tmp_path/'other.py';other.write_text('def preproc(x):return x');path.symlink_to(other)
    elif change=='outside_parent':
        data.rmdir();other=tmp_path/'other';other.mkdir();(other/'data_augment.py').write_text('def preproc(x):return x');data.symlink_to(other,target_is_directory=True)
    elif change=='oversized':path.write_bytes(b'#'*(128*1024+1))
    elif change=='no_function':path.write_text('preproc = 1')
    with pytest.raises(ValueError):loader.load_preproc(root)


def test_actual_pinned_native_file_import_preserves_preproc_source_without_coco(tmp_path,monkeypatch):
    # Offline software source fixture: dependency stubs are never inference evidence.
    archive=Path(os.environ.get('CFU_VERIFIED_YOLOX_SOURCE', '.artifacts/verified-yolox-source.zip'))
    if not archive.is_file():pytest.skip('Retained pinned source fixture is unavailable')
    pin=json.loads((ROOT/'validation-plan.json').read_text())['source_archive']
    assert archive.stat().st_size==pin['size_bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==pin['sha256']
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('/yolox/data/data_augment.py'))
        raw=z.read(name)
    data=tmp_path/'yolox/data';data.mkdir(parents=True);path=data/'data_augment.py';path.write_bytes(raw)
    (data/'__init__.py').write_text("raise RuntimeError('Unused dataset import')")
    cv=types.ModuleType('cv2');utils=types.ModuleType('yolox.utils');utils.xyxy2cxcywh=lambda x:x
    monkeypatch.setitem(sys.modules,'cv2',cv);monkeypatch.setitem(sys.modules,'yolox.utils',utils)
    function=loader.load_preproc(tmp_path)
    assert inspect.getsource(function).encode() in raw
    assert function.__name__=='preproc' and function.__module__=='_cfu_verified_native_preprocessing'
    assert function.__code__.co_filename==str(path)


def test_validation_and_test_use_identical_loader_with_locked_identity():
    test=ROOT.parent/'native-test'
    assert (ROOT/'src/preprocessing.py').read_bytes()==(test/'src/preprocessing.py').read_bytes()
    for path in [ROOT/'src/validation_lib.py',test/'src/test_lib.py']:
        body=path.read_text()
        assert 'from yolox.data.data_augment import preproc' not in body
        assert "preproc = load_preproc(root/'native-source')" in body
