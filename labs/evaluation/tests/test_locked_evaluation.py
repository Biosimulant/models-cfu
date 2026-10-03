"""Locked evaluation software fixtures, without native detector execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

import test_final_test_lock as lock_fixture

evidence = lock_fixture.evidence  # Reuse synthetic validation-only fixture.

ROOT=Path(__file__).resolve().parents[1]/'models/native-test'
VALIDATION=ROOT.parent/'native-validation'
spec=importlib.util.spec_from_file_location('cfu_locked_evaluation_fixture',ROOT/'src/__init__.py',
                                          submodule_search_locations=[str(ROOT/'src')])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
core=__import__(spec.name+'.core',fromlist=['core'])
adapter=__import__(spec.name+'.qualification',fromlist=['qualification'])
lib=__import__(spec.name+'.test_lib',fromlist=['test_lib'])
policy=__import__(spec.name+'.lock',fromlist=['lock'])


@pytest.fixture
def bound_evidence(evidence):
    inputs=copy.deepcopy(evidence[0]);inputs['evaluation_bundle']=adapter.bundle_identity(ROOT)
    return inputs,policy.freeze(**inputs)


def test_preserved_metric_source_geometry_and_inference_algorithms_are_identical():
    for name in ('metrics.py','native_source.py','sahi_source.py','process.py','geometry.py','lock.py','preprocessing.py'):
        assert (ROOT/'src'/name).read_bytes()==(VALIDATION/'src'/name).read_bytes()
    def function(path,name):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    assert function(ROOT/'src/test_lib.py','merge')==function(VALIDATION/'src/validation_lib.py','merge')
    for name in ('original_image','original_detection','canonical_detections'):
        assert function(ROOT/'src/core.py',name)==function(VALIDATION/'src/core.py',name)


def test_manifest_and_runtime_ports_match_with_explicit_optional_baseline():
    model=yaml.safe_load((ROOT/'model.yaml').read_text());instance=adapter.QualifyNativeDetector()
    assert {p['name'] for p in model['io']['inputs']}==set(instance.inputs())
    assert {p['name'] for p in model['io']['outputs']}==set(instance.outputs())
    assert model['io']['outputs'][0]['schema']==adapter.SCHEMA
    assert next(p for p in model['io']['inputs'] if p['name']=='baseline')['required'] is False
    assert all(p['format']=='json' for p in model['io']['outputs'][1:])
    assert model['biosim']['execution_policy']=='once_before_run'
    assert model['runtime']['dependencies']==yaml.safe_load((VALIDATION/'model.yaml').read_text())['runtime']['dependencies']


@pytest.mark.parametrize('phase',['primary','replay','recreation'])
def test_exact_input_set_enforced_for_each_phase(tmp_path,phase):
    values={name:str(tmp_path/'input') for name in adapter.QualifyNativeDetector().inputs() if name!='baseline'}
    (tmp_path/'input').write_bytes(b'fixture')
    if phase!='primary':values['baseline']=str(tmp_path/'input')
    assert set(adapter.checked_request_inputs(values,phase))==set(values)
    bad=dict(values);bad['extra_test_variant']=str(tmp_path/'input')
    with pytest.raises(ValueError):adapter.checked_request_inputs(bad,phase)
    if phase=='primary':bad=dict(values,baseline=str(tmp_path/'input'))
    else:bad={k:v for k,v in values.items() if k!='baseline'}
    with pytest.raises(ValueError):adapter.checked_request_inputs(bad,phase)


@pytest.mark.parametrize('phase',['training','',None])
def test_unknown_phase_rejected(phase):
    with pytest.raises(ValueError):adapter.QualifyNativeDetector(phase=phase)


@pytest.mark.parametrize('change',['lock_digest','weights','code','primary_baseline','missing_replay_baseline'])
def test_lock_and_phase_fail_before_any_test_annotation_read(bound_evidence,tmp_path,monkeypatch,change):
    inputs,lock=bound_evidence;root=tmp_path/'task';root.mkdir()
    frozen_raw=json.dumps(inputs['frozen']).encode();lock_raw=json.dumps(lock).encode()
    values={'frozen_split':'fixture:frozen','test_lock':'fixture:lock','annotations':'must-not-read'}
    weights={k:inputs['weights_file'][k] for k in ('sha256','size_bytes')}
    sha=policy.digest(lock);bundle=copy.deepcopy(inputs['evaluation_bundle']);phase='primary'
    if change=='lock_digest':sha='f'*64
    elif change=='weights':weights['sha256']='f'*64
    elif change=='code':bundle['authored_code']['src/test_lib.py']['sha256']='f'*64
    elif change=='primary_baseline':values['baseline']='must-not-read'
    else:phase='replay'
    reads=[]
    original=lib.pinned_bytes
    def checked(path,pin,maximum):
        reads.append(str(path))
        if str(path)=='fixture:frozen':return frozen_raw
        if str(path)=='fixture:lock':return lock_raw
        if str(path)=='must-not-read':pytest.fail('Test annotations/baseline read before lock and phase guards')
        return original(path,pin,maximum)
    monkeypatch.setattr(lib,'pinned_bytes',checked)
    with pytest.raises(ValueError):
        lib.run(values,inputs['plan'],root,weights_pin=weights,work_seconds=60,
                lock_pin={'sha256':'a'*64,'size_bytes':len(lock_raw)},lock_digest=sha,
                evaluation_bundle=bundle,phase=phase)
    assert 'must-not-read' not in reads


def test_test_reference_selection_preserves_source_anomalies_and_ignores_other_fold_outcomes(bound_evidence):
    inputs,lock=bound_evidence;members=inputs['frozen']['native_dataset_manifest']['members']
    images=[{'id':i,'file_name':r['sample_id'],'width':20,'height':20} for i,r in enumerate(members)]
    annotations=[{'id':i,'image_id':i,'category_id':0,
                  'bbox':[-1,2,4,0] if r['split']=='test' else object()} for i,r in enumerate(members)]
    refs=core.locked_test_references({'images':images,'annotations':annotations,'categories':[{'id':0}]},
                                    inputs['frozen'],lock['test_images'])
    assert len(refs)==37
    assert all(r['reference_boxes_xyxy']==[[-1,2,3,2]] for r in refs.values())
    with pytest.raises(ValueError):
        core.locked_test_references({'images':images,'annotations':annotations,'categories':[{'id':0}]},
                                    inputs['frozen'],inputs['plan']['validation_images'])


@pytest.mark.parametrize('change',['actual_weights','actual_source'])
def test_valid_declared_lock_still_checks_actual_artifacts_before_test_outcomes(bound_evidence,tmp_path,monkeypatch,change):
    inputs,lock=bound_evidence
    rows=policy.test_members(inputs['frozen'],inputs['plan'])
    values={name:'fixture:'+name for name in adapter.QualifyNativeDetector().inputs() if name!='baseline'}
    def checked(path,pin,maximum):
        if str(path)=='fixture:annotations':pytest.fail('Actual test outcome values read before artifact guards')
        if str(path)=='fixture:frozen_split':return json.dumps(inputs['frozen']).encode()
        if str(path)=='fixture:test_lock':return json.dumps(lock).encode()
        if str(path)=='fixture:weights' and change=='actual_weights':raise ValueError('Wrong actual weight bytes')
        return b'fixture:already-checked-by-independent-test'
    def invalid_source(*args,**kwargs):raise ValueError('Wrong actual source bytes')
    monkeypatch.setattr(lib,'pinned_bytes',checked)
    monkeypatch.setattr(lib,'extract_native_source',invalid_source)
    assert len(rows)==37
    with pytest.raises(ValueError,match='Wrong actual'):
        lib.run(values,inputs['plan'],tmp_path,weights_pin={k:inputs['weights_file'][k] for k in ('sha256','size_bytes')},
                work_seconds=60,lock_pin={'sha256':'a'*64,'size_bytes':1},lock_digest=policy.digest(lock),
                evaluation_bundle=inputs['evaluation_bundle'],phase='primary')


def baseline_fixture(lock):
    plates=[{'sample_id':r['sample_id'],'group_id':r['group_id'],'image_width':10,'image_height':10,
             'reference_boxes_xyxy':[[0,0,5,5]],'detections':[{'bbox_xyxy':[0,0,5,5],'score':.9}],
             'predicted_count':1,'status':'completed'} for r in lock['test_images']]
    report=lib.summarize(plates,expected_sample_ids=[r['sample_id'] for r in plates])
    return {'stage':'actual_locked_test_evaluation','phase':'primary','lock_digest':policy.digest(lock),
            'configuration':lock['configuration'],'weights':{k:lock['weights'][k] for k in ('sha256','size_bytes')},
            'evaluation_bundle':lock['evaluation_bundle'],'plates':plates,'metrics':report}


@pytest.mark.parametrize('change',['ready','metrics','references','group','lock','weights','configuration','phase'])
def test_baseline_must_match_locked_primary_and_recompute_actual_metrics(bound_evidence,change):
    _,lock=bound_evidence;baseline=baseline_fixture(lock);records=copy.deepcopy(baseline['plates'])
    if change=='metrics':baseline['metrics']['absolute_count_error']=999
    elif change=='references':baseline['plates'][0]['reference_boxes_xyxy']=[]
    elif change=='group':baseline['plates'][0]['group_id']='other'
    elif change=='lock':baseline['lock_digest']='f'*64
    elif change=='weights':baseline['weights']['sha256']='f'*64
    elif change=='configuration':baseline['configuration']={'id':'different'}
    elif change=='phase':baseline['phase']='recreation'
    if change=='ready':assert lib.checked_baseline(baseline,lock,records,'replay')==baseline['metrics']
    else:
        with pytest.raises(ValueError):lib.checked_baseline(baseline,lock,records,'replay')


def test_package_source_bundle_and_validation_plan_have_pinned_identities():
    bundle=adapter.bundle_identity(ROOT)
    assert policy.REQUIRED_TEST_CODE<=set(bundle['authored_code'])
    for name,pin in bundle['authored_code'].items():
        raw=(ROOT/name).read_bytes();assert len(raw)==pin['size_bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
    assert (ROOT/'validation-plan.json').read_bytes()==(VALIDATION/'validation-plan.json').read_bytes()
