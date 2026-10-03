"""Derive an immutable lock from exact actual validation bytes, before test access."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .freeze import checked_origin
from .lock import digest, freeze_from_artifacts
from .native_source import pinned_bytes


def run(request,contract,module_root):
    module_root=Path(module_root);root=Path(request['root'])
    for name,pin in contract['authored_code'].items():
        pinned_bytes(module_root/'src'/name,pin,128*1024)
    if set(request['inputs'])!={'frozen_split','comparison','predictions','weights'}:
        raise ValueError('Managed lock cannot consume test images or annotations')
    checked_origin(request['validation_origin'],request['weights_file'])
    pinned_bytes(request['inputs']['weights'],request['weights_file'],64*1024*1024)
    packages={n:importlib.metadata.version(n) for n in contract['required_package_versions']}
    if packages!=contract['required_package_versions']:
        raise RuntimeError('Lock software runtime differs from its exact declared dependencies')
    lock=freeze_from_artifacts(contract['validation_plan'],
                              frozen_path=request['inputs']['frozen_split'],
                              comparison_path=request['inputs']['comparison'],
                              predictions_path=request['inputs']['predictions'],
                              validation_origin=request['validation_origin'],
                              weights_file=request['weights_file'],
                              evaluation_bundle=contract['evaluation_bundle'])
    path=root/'final-test-lock.json'
    raw=(json.dumps(lock,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if not 0<len(raw)<=8*1024*1024:raise ValueError('Pre-test lock exceeds its bound')
    path.write_bytes(raw);path.chmod(0o600)
    receipt={'stage':'managed_validation_selected_pre_test_lock','lock_digest':digest(lock),
             'configuration':lock['configuration'],'weights':lock['weights'],
             'validation_origin':lock['validation_origin'],'validation_metrics':lock['validation_metrics'],
             'test_plate_count':len(lock['test_images']),'test_outcomes_inspected':False,
             'scientific_acceptance_established':False,
             'environment':{'python':platform.python_version(),'package_versions':packages},
             'artifact':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}}
    return {'receipt':receipt,'lock_path':str(path.resolve())}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_bytes())
    module_root=Path(__file__).resolve().parent.parent
    contract=json.loads((module_root/'freeze-plan.json').read_bytes())
    outputs=run(request,contract,module_root)
    Path(sys.argv[2]).write_text(json.dumps(outputs,allow_nan=False));Path(sys.argv[2]).chmod(0o600)
