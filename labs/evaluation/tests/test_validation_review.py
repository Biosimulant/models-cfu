"""Visual-review contracts exercised with synthetic software evidence only."""
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]/'models/validation-review'
spec=importlib.util.spec_from_file_location('cfu_validation_review_fixture',ROOT/'src/__init__.py',
                                          submodule_search_locations=[str(ROOT/'src')])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
worker=__import__(spec.name+'.worker',fromlist=['worker'])
adapter=__import__(spec.name+'.review',fromlist=['review'])
s=importlib.util.spec_from_file_location('cfu_review_existing_fixture',Path(__file__).with_name('test_final_test_lock.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);evidence=m.evidence


def pin(raw):return {'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def request_contract(tmp_path,evidence):
    old=copy.deepcopy(evidence[0]);plan=old['plan'];comparison=old['comparison']
    contract=json.loads((ROOT/'review-plan.json').read_bytes())
    photo=Image.new('RGB',(10,10),'#202020');stream=io.BytesIO();photo.save(stream,format='JPEG');photo.close();raw=stream.getvalue()
    inputs={};rows=old['frozen']['native_dataset_manifest']['members'][:37]
    for i,row in enumerate(rows):
        row.update(pin(raw));path=tmp_path/row['sample_id'];path.write_bytes(raw);inputs[f'image_{i:02d}']=str(path)
    plan['validation_images']=copy.deepcopy(rows)
    plan['frozen_dataset_sha256']=hashlib.sha256(json.dumps(old['frozen']['native_dataset_manifest'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    comparison['frozen_dataset_sha256']=plan['frozen_dataset_sha256'];comparison['weights_progress']={'epoch':0,'cursor':1,'global_step':1}
    comparison['source']['untrusted_note']='</script><script>alert(1)</script>'
    frozen_raw=json.dumps(old['frozen']).encode();plan['frozen_split']=pin(frozen_raw)
    p=tmp_path/'frozen.json';p.write_bytes(frozen_raw);inputs['frozen_split']=str(p)
    p=tmp_path/'comparison.json';raw=json.dumps(comparison).encode();p.write_bytes(raw);inputs['comparison']=str(p)
    origin=old['validation_origin'];origin['comparison'].update(pin(raw))
    p=tmp_path/'predictions.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in old['predictions'].items():archive.writestr(name+'.json',json.dumps(payload))
    inputs['predictions']=str(p);origin['predictions'].update(pin(p.read_bytes()))
    contract['validation_plan']=plan
    # Local fixture environment is explicitly observed; the managed runtime pins stay unchanged.
    contract['required_package_versions']={n:importlib.metadata.version(n) for n in contract['required_package_versions']}
    return {'inputs':inputs,'root':str(tmp_path),'validation_origin':origin},contract,old


def test_review_retains_actual_original_coordinates_and_complete_visuals(request_contract):
    request,contract,old=request_contract;outputs=worker.run(request,contract,ROOT)
    path=Path(outputs['review_path']);receipt=outputs['receipt'];assert receipt['artifact']==pin(path.read_bytes())
    assert receipt['plate_count']==37 and receipt['test_outcomes_inspected'] is False and receipt['scientific_acceptance_established'] is False
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();assert len(names)==76 and len(set(names))==76
        report=json.loads(archive.read('review.json'));html=archive.read('index.html').decode()
        assert html.count('<script>')==1 and '<script>alert(1)' not in html
        chosen=old['predictions'][old['comparison']['selected_configuration']['id']]
        assert report['metrics']==chosen['metrics']
        for actual,expected in zip(report['plates'],chosen['plates']):
            assert actual['detections']==expected['detections'] and actual['reference_boxes_xyxy']==expected['reference_boxes_xyxy']
            with Image.open(io.BytesIO(archive.read(actual['annotated_image']))) as image:assert image.size==(10,100)


@pytest.mark.parametrize('mutation',['annotations','origin_digest','photo_bytes','wrong_runtime','comparison_metrics'])
def test_visual_review_rejects_changed_scientific_evidence(request_contract,mutation):
    request,contract,old=request_contract
    if mutation=='annotations':request['inputs']['annotations']='forbidden.json'
    elif mutation=='origin_digest':request['validation_origin']['predictions']['sha256']='f'*64
    elif mutation=='photo_bytes':Path(request['inputs']['image_00']).write_bytes(b'changed')
    elif mutation=='wrong_runtime':contract['required_package_versions']['Pillow']='0.0.0'
    else:
        p=Path(request['inputs']['comparison']);d=json.loads(p.read_bytes());d['selected_validation_metrics']['wape']=1.5
        raw=json.dumps(d).encode();p.write_bytes(raw);request['validation_origin']['comparison'].update(pin(raw))
    with pytest.raises((ValueError,RuntimeError)):worker.run(request,contract,ROOT)


def test_review_manifest_matches_finite_port_contract():
    manifest=yaml.safe_load((ROOT/'model.yaml').read_text());obj=adapter.RenderValidationReview()
    assert obj.execution_policy.value=='once_before_run'
    assert set(obj.inputs())==set(p['name'] for p in manifest['io']['inputs'])==set(adapter.FORMATS)
    assert set(obj.outputs())==set(p['name'] for p in manifest['io']['outputs'])
    with pytest.raises(ValueError):obj.execute({'annotations':None},context=None)
