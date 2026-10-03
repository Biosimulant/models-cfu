"""Scientific software contracts only; no real detector-performance evidence."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]/'models/native-validation'
spec=importlib.util.spec_from_file_location('cfu_validation_fixture',ROOT/'src/__init__.py',submodule_search_locations=[str(ROOT/'src')])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
core=__import__(spec.name+'.core',fromlist=['core'])
metrics=__import__(spec.name+'.metrics',fromlist=['metrics'])
adapter=__import__(spec.name+'.validation',fromlist=['validation'])


def frozen_fixture():
    rows=[{'role':'image','sample_id':f'val{i:02d}.jpg','split':'validation','group_id':f'group{i:02d}',
           'file_id':str(i),'size_bytes':1,'sha256':'a'*64} for i in range(37)]
    manifest={'members':rows+[{'role':'image','sample_id':'test.jpg','split':'test','group_id':'test-only'}]}
    frozen={'native_dataset_manifest':manifest}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return frozen,{'frozen_dataset_sha256':digest,'validation_images':copy.deepcopy(rows)}


def test_frozen_validation_coverage_requires_all_images_and_whole_group_split():
    frozen,plan=frozen_fixture()
    assert len(core.validation_members(frozen,plan))==37
    changed=copy.deepcopy(frozen)
    changed['native_dataset_manifest']['members'][-1]['group_id']='group00'
    plan['frozen_dataset_sha256']=hashlib.sha256(json.dumps(changed['native_dataset_manifest'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError,match='Related'):
        core.validation_members(changed,plan)


@pytest.mark.parametrize('change',['missing','extra','changed_pin','wrong_fold'])
def test_validation_member_mutations_cannot_change_frozen_evaluation(change):
    frozen,plan=frozen_fixture()
    if change=='missing':plan['validation_images'].pop()
    elif change=='extra':plan['validation_images'].append(copy.deepcopy(plan['validation_images'][0]))
    elif change=='changed_pin':plan['validation_images'][0]['sha256']='b'*64
    else:plan['validation_images'][0]['split']='test'
    with pytest.raises(ValueError):core.validation_members(frozen,plan)


def test_reference_selection_never_examines_other_fold_bbox_and_preserves_zero_area_edges():
    frozen={'native_dataset_manifest':{'members':[{'role':'image','sample_id':'val.jpg'},
                                                  {'role':'image','sample_id':'test.jpg'}]}}
    rows=[{'sample_id':'val.jpg','split':'validation'}]
    document={'images':[{'id':1,'file_name':'val.jpg','width':100,'height':80},
                        {'id':2,'file_name':'test.jpg','width':100,'height':80}],
              'categories':[{'id':0}],
              'annotations':[{'id':1,'image_id':1,'category_id':0,'bbox':[-2,3,5,4]},
                             {'id':2,'image_id':1,'category_id':0,'bbox':[8,9,5,0]},
                             {'id':3,'image_id':2,'bbox':object()}]}
    result=core.validation_references(document,frozen,rows)
    assert set(result)=={'val.jpg'}
    assert result['val.jpg']['reference_boxes_xyxy']==[[-2,3,3,7],[8,9,13,9]]
    assert result['val.jpg']['source_annotation_ids']==[1,2]
    with pytest.raises(ValueError,match='cannot inspect'):
        core.validation_references(document,frozen,[{'sample_id':'test.jpg','split':'test'}])


def test_tile_coordinate_offset_and_letterbox_scale_map_to_original_frame():
    d=core.original_detection([20,30,10,12,.8,.5],ratio=.5,window=[100,200,200,300],
                              width=500,height=500,confidence=.3)
    assert d=={'bbox_xyxy':[130,248,150,272],'score':.4,'class_id':0}
    assert core.original_detection([20,30,10,12,.8,.5],ratio=.5,window=[100,200,200,300],
                                   width=500,height=500,confidence=.5) is None


def test_padded_region_predictions_are_removed_and_edge_boxes_clip_to_visible_window():
    assert core.original_detection([500,500,10,10,.9,.9],ratio=1,window=[0,0,50,40],
                                   width=50,height=40,confidence=.05) is None
    d=core.original_detection([1,1,8,10,.9,.9],ratio=1,window=[0,0,50,40],
                              width=50,height=40,confidence=.05)
    assert d['bbox_xyxy']==[0,0,5,6]


@pytest.mark.parametrize('row',[[1,1,2,2,float('nan'),.5],[1,1,2,2,1.2,.5],
                              [1,1,-2,2,.5,.5],[1,1,2,2,.5]])
def test_invalid_native_tensors_are_explicit_errors(row):
    with pytest.raises(ValueError):
        core.original_detection(row,ratio=1,window=[0,0,20,20],width=20,height=20,confidence=.05)


def test_original_exif_coordinates_and_byte_pins_are_checked(tmp_path):
    image=Image.new('RGB',(40,20));exif=Image.Exif();exif[274]=6
    stream=io.BytesIO();image.save(stream,format='JPEG',exif=exif);image.close();raw=stream.getvalue()
    path=tmp_path/'source.jpg';path.write_bytes(raw)
    pin={'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    with core.original_image(path,pin,{'width':20,'height':40}) as normalized:assert normalized.size==(20,40)
    with pytest.raises(ValueError,match='annotation frame'):core.original_image(path,pin,{'width':40,'height':20})
    path.write_bytes(b'x'*len(raw))
    with pytest.raises(ValueError,match='digest'):core.original_image(path,pin,{'width':20,'height':40})


def test_empty_predictions_do_not_crash_candidate_selection_or_establish_performance():
    record={'sample_id':'synthetic','group_id':'synthetic','image_width':10,'image_height':10,
            'status':'completed','reference_boxes_xyxy':[[0,0,3,3]],'detections':[],'predicted_count':0}
    report=metrics.summarize([record],expected_sample_ids=['synthetic'],bootstrap_replicates=10)
    selected,measured=core.choose_validation_candidate([(core.candidate_grid()[0],report)])
    assert selected['mode']=='raw' and measured['precision'] is None and measured['acceptance_passed'] is False


def test_gate_passing_candidate_precedes_lower_count_error_with_bad_localization():
    a={'id':'passing'};b={'id':'low-count-error-bad-boxes'}
    passing={'failures':0,'acceptance_passed':True,'wape':.05,'p95_symmetric_error':.1,'recall':.95,'precision':.95}
    bad={**passing,'acceptance_passed':False,'wape':0,'recall':.1,'precision':.1}
    assert core.choose_validation_candidate([(b,bad),(a,passing)])[0]==a


@pytest.mark.parametrize('sha,size',[('',0),('a'*64,True),('A'*64,1),('a'*64,64*1024*1024+1)])
def test_unpinned_or_oversized_weights_cannot_start_validation(sha,size):
    with pytest.raises(ValueError):adapter.checked_weight_pin(sha,size)


def test_python_and_manifest_interfaces_match_and_candidate_grid_is_fixed():
    model=yaml.safe_load((ROOT/'model.yaml').read_text());instance=adapter.ValidateNativeDetector()
    assert {s['name'] for s in model['io']['inputs']}==set(instance.inputs())
    assert {s['name'] for s in model['io']['outputs']}==set(instance.outputs())
    assert len(core.candidate_grid())==32 and len({c['id'] for c in core.candidate_grid()})==32
    assert all(c['confidence']>=.05 for c in core.candidate_grid())
    assert model['io']['outputs'][0]['schema']==adapter.SCHEMA


def test_vendored_reference_geometry_and_metric_algorithms_are_identical_to_reviewed_modules():
    research=ROOT.parents[3]
    assert (ROOT/'src/metrics.py').read_bytes()==(ROOT.parents[0]/'metrics/src/metrics.py').read_bytes()
    assert (ROOT/'src/geometry.py').read_bytes()==(research/'labs/preparation/models/tiles/src/geometry.py').read_bytes()



@pytest.mark.parametrize('change',['ready','unqualified','traversal','symlink','changed_member'])
def test_sahi_source_guards_verify_before_extraction_and_retain_primary_notice(tmp_path,monkeypatch,change):
    import zipfile,stat
    source=__import__(spec.name+'.sahi_source',fromlist=['sahi_source'])
    stream=io.BytesIO()
    bodies={'sahi/slicing.py':b'# synthetic; never executed','sahi-0.11.21.dist-info/licenses/LICENSE':b'synthetic primary notice'}
    if change=='traversal':bodies['../outside.py']=b'unsafe'
    with zipfile.ZipFile(stream,'w') as z:
        for name,body in bodies.items():z.writestr(name,body)
        if change=='symlink':
            info=zipfile.ZipInfo('sahi/link.py');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,b'outside')
            bodies['sahi/link.py']=b'outside'
    raw=stream.getvalue();archive=tmp_path/'wheel.zip';archive.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest();license_sha=hashlib.sha256(b'synthetic primary notice').hexdigest()
    monkeypatch.setattr(source,'WHEEL_SHA',sha);monkeypatch.setattr(source,'LICENSE_SHA',license_sha)
    records=[{'path':n,'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in bodies.items()]
    if change=='changed_member':records[0]['sha256']='0'*64
    m={'source_inspection_passed':change!='unqualified','package':'sahi','version':'0.11.21',
       'provenance':{'wheel_sha256':sha,'wheel_size_bytes':len(raw)},'members':records,
       'license_observed':{'license':'MIT','records':[{'path':'sahi-0.11.21.dist-info/licenses/LICENSE','sha256':license_sha}]}}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    def pin(p):return {'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    destination=tmp_path/'extracted'
    if change=='ready':
        r=source.extract_sahi(archive,manifest,destination,archive_pin=pin(archive),manifest_pin=pin(manifest))
        assert r['third_party_code_executed_by_extraction'] is False
        assert (destination/'sahi-0.11.21.dist-info/licenses/LICENSE').read_bytes()==b'synthetic primary notice'
    else:
        with pytest.raises(ValueError):
            source.extract_sahi(archive,manifest,destination,archive_pin=pin(archive),manifest_pin=pin(manifest))
        assert not destination.exists()
    assert not (tmp_path/'outside.py').exists()
