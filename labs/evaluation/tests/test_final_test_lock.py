"""Software-only pre-outcome guards; fixtures do not measure CFU accuracy."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import stat
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]/'models/native-validation'
spec = importlib.util.spec_from_file_location('cfu_lock_fixture', ROOT/'src/__init__.py',
                                             submodule_search_locations=[str(ROOT/'src')])
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
policy = __import__(spec.name+'.lock', fromlist=['lock'])


@pytest.fixture(scope='module')
def evidence():
    plan = json.loads((ROOT/'validation-plan.json').read_bytes())
    rows = []
    for split in ('validation', 'test'):
        for i in range(37):
            rows.append({'role':'image', 'sample_id':f'{split}{i:02d}.jpg', 'split':split,
                         'group_id':f'{split}-{i}', 'sha256':'a'*64, 'size_bytes':10,
                         'file_id':f'00000000-0000-4000-8000-{len(rows)+1:012d}'})
    frozen = {'native_dataset_manifest':{'members':rows}}
    plan['frozen_dataset_sha256'] = policy.digest(frozen['native_dataset_manifest'])
    plan['validation_images'] = copy.deepcopy(rows[:37])
    plates = [{'sample_id':r['sample_id'], 'group_id':r['group_id'], 'status':'completed',
               'image_width':10, 'image_height':10, 'reference_boxes_xyxy':[[0,0,5,5]],
               'detections':[{'bbox_xyxy':[0,0,5,5], 'score':.9}], 'predicted_count':1} for r in rows[:37]]
    report = policy.summarize(plates, expected_sample_ids=[r['sample_id'] for r in rows[:37]])
    grid = policy.candidate_grid()
    selected, _ = policy.choose_validation_candidate([(c,report) for c in grid])
    pin = {'sha256':'b'*64, 'size_bytes':100}
    weights = {**pin, 'file_id':'00000000-0000-4000-8000-000000000100'}
    origin = {'run_id':'00000000-0000-4000-8000-000000000101',
              'revision_id':'00000000-0000-4000-8000-000000000102',
              'comparison':{**pin,'file_id':'00000000-0000-4000-8000-000000000103'},
              'predictions':{**pin,'file_id':'00000000-0000-4000-8000-000000000104'}}
    bundle = {'entrypoint':'src.qualification:QualifyNativeDetector', 'model_manifest':pin,
              'authored_code':{n:copy.deepcopy(pin) for n in policy.REQUIRED_TEST_CODE}}
    for name in policy.UNCHANGED_ALGORITHMS:
        bundle['authored_code']['src/'+name] = copy.deepcopy(plan['authored_code'][name])
    comparison = {'stage':'actual_validation_candidate_comparison', 'weights':pin,
                  'frozen_dataset_id':plan['frozen_dataset_id'],
                  'frozen_dataset_sha256':plan['frozen_dataset_sha256'],
                  'environment':policy.expected_environment(plan), 'candidate_grid':grid,
                  'candidates':[{'configuration':c,'metrics':report} for c in grid],
                  'source':{'source_commit':policy.COMMIT,'source_archive_sha256':plan['source_archive']['sha256']},
                  'sahi':{'version':policy.VERSION, 'source_sha256':plan['sahi_archive']['sha256'],
                          'license_sha256':policy.LICENSE_SHA},
                  'selected_configuration':selected, 'selected_validation_metrics':report,
                  'validation_selection_complete':True, 'test_outcomes_inspected':False,
                  'final_test_completed':False, 'scientific_acceptance_established':False}
    predictions = {c['id']:{'configuration':c,'plates':plates,'metrics':report} for c in grid}
    result = {'plan':plan,'frozen':frozen,'comparison':comparison,'predictions':predictions,
              'validation_origin':origin,'weights_file':weights,'evaluation_bundle':bundle}
    lock = policy.freeze(**result)
    return result, lock


def test_lock_binds_all_test_originals_and_remains_unqualified(evidence):
    inputs, lock = evidence
    assert len(lock['test_images']) == 37
    assert {r['split'] for r in lock['test_images']} == {'test'}
    assert lock['excluded_test_images'] == []
    assert lock['scientific_acceptance_established'] is False
    assert lock['test_outcomes_inspected_at_lock'] is False
    assert 'reference_boxes_xyxy' not in json.dumps(lock['test_images'])
    assert policy.verify(lock, expected_sha256=policy.digest(lock), plan=inputs['plan'],
                         frozen=inputs['frozen'], evaluation_bundle=inputs['evaluation_bundle'],
                         weights_file=inputs['weights_file']) == lock['configuration']


@pytest.mark.parametrize('field,value', [
    ('validation_selection_complete',False), ('test_outcomes_inspected',True),
    ('final_test_completed',True), ('scientific_acceptance_established',True),
    ('stage','synthetic_validation'), ('weights',{'sha256':'c'*64,'size_bytes':100}),
    ('selected_configuration',{'id':'not-in-grid'}),
    ('environment',{'python':'wrong'}), ('frozen_dataset_sha256','d'*64),
    ('source',{'source_commit':'wrong','source_archive_sha256':'a'*64}),
    ('sahi',{'version':'wrong','source_sha256':'a'*64,'license_sha256':'a'*64})])
def test_changed_validation_scope_or_identity_cannot_lock(evidence, field, value):
    inputs = copy.deepcopy(evidence[0]); inputs['comparison'][field] = value
    with pytest.raises(ValueError): policy.freeze(**inputs)


@pytest.mark.parametrize('change', ['missing_config','duplicate_config','altered_metric','missing_plate',
                                  'changed_group','changed_box','failed_plate','not_passing'])
def test_lock_recomputes_all_candidate_metrics_and_requires_complete_predictions(evidence, change):
    inputs = copy.deepcopy(evidence[0]); config = inputs['comparison']['candidate_grid'][0]
    payload = inputs['predictions'][config['id']]
    if change == 'missing_config': inputs['predictions'].pop(config['id'])
    elif change == 'duplicate_config': inputs['comparison']['candidates'][1]['configuration'] = config
    elif change == 'altered_metric': payload['metrics']['absolute_count_error'] = 900
    elif change == 'missing_plate': payload['plates'].pop()
    elif change == 'changed_group': payload['plates'][0]['group_id'] = 'test-group'
    elif change == 'changed_box': payload['plates'][0]['detections'][0]['bbox_xyxy'] = [5,5,9,9]
    elif change == 'failed_plate':
        payload['plates'][0].update(status='failed',detections=[],predicted_count=None,failure={'message':'offline'})
    else:
        for plate in payload['plates']: plate.update(detections=[],predicted_count=0)
        report = policy.summarize(payload['plates'],expected_sample_ids=[r['sample_id'] for r in inputs['plan']['validation_images']])
        # deepcopy preserves shared fixture aliases: all32 recorded reports change.
        for document in inputs['predictions'].values(): document['metrics'] = report
        for candidate in inputs['comparison']['candidates']: candidate['metrics'] = report
        inputs['comparison']['selected_validation_metrics'] = report
    with pytest.raises(ValueError): policy.freeze(**inputs)


@pytest.mark.parametrize('change',['missing_test','duplicate_test','cross_fold_group','changed_algorithm',
                                  'missing_code','traversal_code','unbounded_pin'])
def test_test_partition_and_execution_bundle_are_locked_before_outcomes(evidence, change):
    inputs = copy.deepcopy(evidence[0]); members = inputs['frozen']['native_dataset_manifest']['members']
    bundle = inputs['evaluation_bundle']
    if change == 'missing_test': members.pop()
    elif change == 'duplicate_test': members[-1]['sample_id'] = members[-2]['sample_id']
    elif change == 'cross_fold_group': members[-1]['group_id'] = members[0]['group_id']
    elif change == 'changed_algorithm': bundle['authored_code']['src/metrics.py']['sha256'] = 'f'*64
    elif change == 'missing_code': bundle['authored_code'].pop('src/worker.py')
    elif change == 'traversal_code': bundle['authored_code']['../outside.py'] = {'sha256':'a'*64,'size_bytes':1}
    else: inputs['weights_file']['size_bytes'] = 64*1024*1024+1
    with pytest.raises(ValueError): policy.freeze(**inputs)


@pytest.mark.parametrize('binding',['weights','configuration','metric_plan','test_images','excluded_test_images',
                                   'evaluation_bundle','evaluation_plan','test_outcomes_inspected_at_lock'])
def test_lock_digest_detects_any_later_binding_mutation(evidence,binding):
    inputs, lock = evidence; changed = copy.deepcopy(lock); changed[binding] = None
    with pytest.raises(ValueError):
        policy.verify(changed, expected_sha256=policy.digest(lock), plan=inputs['plan'],
                      frozen=inputs['frozen'],evaluation_bundle=inputs['evaluation_bundle'],weights_file=inputs['weights_file'])


@pytest.mark.parametrize('changed',['weights','code','plan'])
def test_unchanged_lock_rejects_changed_execution_inputs(evidence,changed):
    inputs, lock = copy.deepcopy(evidence)
    if changed == 'weights': inputs['weights_file']['sha256'] = 'f'*64
    elif changed == 'code': inputs['evaluation_bundle']['authored_code']['src/worker.py']['sha256'] = 'f'*64
    else: inputs['plan']['preprocessing'] = 'rescaled coordinates'
    with pytest.raises(ValueError):
        policy.verify(lock,expected_sha256=policy.digest(lock),plan=inputs['plan'],frozen=inputs['frozen'],
                      evaluation_bundle=inputs['evaluation_bundle'],weights_file=inputs['weights_file'])


@pytest.mark.parametrize('change',['ready','digest','missing','extra','duplicate','link','not_json','oversized'])
def test_artifact_reader_checks_bytes_and_safe_complete_zip_before_lock(evidence,tmp_path,change):
    inputs = copy.deepcopy(evidence[0])
    frozen_path = tmp_path/'frozen.json'; frozen_path.write_text(json.dumps(inputs['frozen']))
    comparison_path = tmp_path/'comparison.json'; comparison_path.write_text(json.dumps(inputs['comparison']))
    predictions_path = tmp_path/'predictions.zip'
    documents = inputs['predictions']; first = next(iter(documents))
    with zipfile.ZipFile(predictions_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for key, document in documents.items():
            if key == first and change == 'missing': continue
            name = key+'.json'; body = json.dumps(document)
            if key == first:
                if change == 'link':
                    name = zipfile.ZipInfo(name); name.external_attr = (stat.S_IFLNK|0o777)<<16
                elif change == 'not_json': body = 'bad JSON'
                elif change == 'oversized': body = b'0'*(32*1024*1024+1)
            archive.writestr(name, body)
        if change == 'extra': archive.writestr('../outside.json','{}')
        if change == 'duplicate':
            with pytest.warns(UserWarning,match='Duplicate name'): archive.writestr(first+'.json','{}')
    def pin(path):
        raw = path.read_bytes()
        return {'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    inputs['plan']['frozen_split'].update(pin(frozen_path))
    inputs['validation_origin']['comparison'].update(pin(comparison_path))
    inputs['validation_origin']['predictions'].update(pin(predictions_path))
    if change == 'digest': inputs['validation_origin']['predictions']['sha256'] = 'f'*64
    kwargs = {k:inputs[k] for k in ('validation_origin','weights_file','evaluation_bundle')}
    kwargs.update(frozen_path=frozen_path,comparison_path=comparison_path,predictions_path=predictions_path)
    if change == 'ready':
        lock = policy.freeze_from_artifacts(inputs['plan'],**kwargs)
        assert lock['scientific_acceptance_established'] is False
        assert len(lock['test_images']) == 37
        assert not (tmp_path.parent/'outside.json').exists()
    else:
        with pytest.raises((ValueError,json.JSONDecodeError)): policy.freeze_from_artifacts(inputs['plan'],**kwargs)
