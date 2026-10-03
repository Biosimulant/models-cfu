"""Managed lock adapter evidence using explicit software fixtures, never CFU outcomes."""
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]/'models/final-test-lock'
spec=importlib.util.spec_from_file_location('cfu_managed_lock_fixture',ROOT/'src/__init__.py',
                                          submodule_search_locations=[str(ROOT/'src')])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
adapter=__import__(spec.name+'.freeze',fromlist=['freeze'])
worker=__import__(spec.name+'.worker',fromlist=['worker'])
fixture_spec=importlib.util.spec_from_file_location('cfu_existing_lock_fixture',Path(__file__).with_name('test_final_test_lock.py'))
fixture_module=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture_module)
evidence=fixture_module.evidence


def pin(path):
    raw=path.read_bytes()
    return {'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def request_contract(tmp_path,evidence):
    inputs=copy.deepcopy(evidence[0]);plan=inputs['plan']
    paths={n:tmp_path/(n+'.'+ext) for n,ext in adapter.FORMATS.items()}
    paths['weights'].write_bytes(b'Explicit synthetic weight bytes; this fixture never imports Torch or performs inference.')
    inputs['weights_file'].update(pin(paths['weights']))
    inputs['comparison']['weights']={k:inputs['weights_file'][k] for k in ('sha256','size_bytes')}
    paths['frozen_split'].write_text(json.dumps(inputs['frozen']));plan['frozen_split']=pin(paths['frozen_split'])
    paths['comparison'].write_text(json.dumps(inputs['comparison']))
    with zipfile.ZipFile(paths['predictions'],'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in inputs['predictions'].items():archive.writestr(name+'.json',json.dumps(payload))
    for name in ('comparison','predictions'):inputs['validation_origin'][name].update(pin(paths[name]))
    contract=json.loads((ROOT/'freeze-plan.json').read_bytes())
    contract.update(validation_plan=plan,evaluation_bundle=inputs['evaluation_bundle'])
    # Software fixtures use the observed local test runtime; the managed contract stays pinned.
    contract['required_package_versions']={n:importlib.metadata.version(n) for n in contract['required_package_versions']}
    request={'inputs':{n:str(p) for n,p in paths.items()},'root':str(tmp_path),
             'weights_file':inputs['weights_file'],'validation_origin':inputs['validation_origin']}
    return request,contract


def test_managed_worker_derives_lock_from_retained_fixture_bytes(request_contract):
    request,contract=request_contract
    result=worker.run(request,contract,ROOT);receipt=result['receipt']
    path=Path(result['lock_path']);lock=json.loads(path.read_bytes())
    assert receipt['artifact']==pin(path)
    assert receipt['lock_digest']==worker.digest(lock)
    assert receipt['test_plate_count']==37
    assert receipt['scientific_acceptance_established'] is False
    assert receipt['test_outcomes_inspected'] is False
    assert all(r['split']=='test' and 'reference_boxes_xyxy' not in r for r in lock['test_images'])


@pytest.mark.parametrize('mutation',['test_annotations','changed_weight_bytes','changed_origin_digest','missing_code_pin','wrong_runtime'])
def test_managed_lock_rejects_changed_inputs_before_qualification(request_contract,mutation):
    request,contract=request_contract
    if mutation=='test_annotations':request['inputs']['annotations']='not-accepted.json'
    elif mutation=='changed_weight_bytes':Path(request['inputs']['weights']).write_bytes(b'changed')
    elif mutation=='changed_origin_digest':request['validation_origin']['predictions']['sha256']='f'*64
    elif mutation=='missing_code_pin':contract['authored_code']['worker.py']['sha256']='f'*64
    else:contract['required_package_versions']['Pillow']='0.0.0'
    with pytest.raises((ValueError,RuntimeError)):worker.run(request,contract,ROOT)
    assert not (Path(request['root'])/'final-test-lock.json').exists()


def test_finite_adapter_manifest_agrees_and_has_no_test_outcome_port():
    manifest=yaml.safe_load((ROOT/'model.yaml').read_text());obj=adapter.FreezeFinalTest()
    assert set(obj.inputs())==set(p['name'] for p in manifest['io']['inputs'])==set(adapter.FORMATS)
    assert set(obj.outputs())==set(p['name'] for p in manifest['io']['outputs'])
    assert obj.execution_policy.value=='once_before_run'
    with pytest.raises(ValueError):obj.execute({'annotations':None},context=None)


def test_future_test_bundle_matches_current_staged_bytes():
    test=ROOT.parent/'native-test';contract=json.loads((ROOT/'freeze-plan.json').read_bytes())
    bundle=contract['evaluation_bundle']
    assert bundle['model_manifest']==pin(test/'model.yaml')
    assert bundle['authored_code']=={str(p.relative_to(test)):pin(p) for p in sorted((test/'src').glob('*.py'))}
    assert contract['validation_plan']==json.loads((ROOT.parent/'native-validation'/'validation-plan.json').read_bytes())
