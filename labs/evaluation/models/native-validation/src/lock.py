"""Pre-test identity lock derived from retained validation predictions.

This is a software guard, not proof of managed execution or publication approval.
The caller must independently verify artifact bytes and owned Run/File origins.
No test annotation values are needed or accepted by this module.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import PurePosixPath
import stat
from uuid import UUID
import zipfile

from .core import candidate_grid, choose_validation_candidate, validation_members
from .metrics import GATES, METHOD, summarize
from .native_source import COMMIT, pinned_bytes
from .sahi_source import LICENSE_SHA, VERSION

FORMAT = 'cfu-final-test-lock-v1'
REQUIRED_TEST_CODE = {'src/test_lib.py', 'src/worker.py', 'src/qualification.py',
                      'src/core.py', 'src/metrics.py', 'src/native_source.py',
                      'src/sahi_source.py', 'src/process.py', 'src/lock.py', 'src/preprocessing.py'}
UNCHANGED_ALGORITHMS = ('metrics.py', 'native_source.py', 'sahi_source.py', 'process.py', 'preprocessing.py')
PLAN_KEYS = ('frozen_split', 'annotations', 'frozen_dataset_id', 'frozen_dataset_sha256',
             'source_archive', 'source_receipt', 'sahi_archive', 'sahi_receipt',
             'training_contract_sha256', 'training_pool_sha256', 'training_recipe',
             'required_environment', 'required_package_versions', 'preprocessing', 'prediction_policy')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked_pin(value, *, maximum=64*1024*1024, file_required=False):
    sha, size = value.get('sha256'), value.get('size_bytes')
    if (not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
            or isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= maximum):
        raise ValueError('Lock needs exact bounded artifact byte identities')
    if file_required:
        UUID(value['file_id'])
    return copy.deepcopy(value)


def test_members(frozen, plan):
    # Reuse the validation guard for frozen dataset identity and group isolation.
    validation_members(frozen, plan)
    rows = sorted((r for r in frozen['native_dataset_manifest']['members']
                   if r['role'] == 'image' and r['split'] == 'test'), key=lambda r:r['sample_id'])
    if len(rows) != 37 or len({r['sample_id'] for r in rows}) != 37:
        raise ValueError('Final test requires every one of the37 frozen test originals')
    for row in rows:
        checked_pin(row, maximum=20*1024*1024, file_required=True)
    return rows


def checked_bundle(bundle):
    if bundle['entrypoint'] != 'src.qualification:QualifyNativeDetector':
        raise ValueError('Final test must use its separately locked finite adapter')
    code = bundle['authored_code']
    if not REQUIRED_TEST_CODE <= set(code):
        raise ValueError('Final test code bundle is incomplete')
    for name, pin in code.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or not name.endswith('.py'):
            raise ValueError('Locked code path must be a canonical relative Python source')
        checked_pin(pin, maximum=128*1024)
    checked_pin(bundle['model_manifest'], maximum=128*1024)
    return copy.deepcopy(bundle)


def expected_environment(plan):
    return {**plan['required_environment'], 'package_versions':{
        k: plan['required_environment'].get(k, v) for k, v in plan['required_package_versions'].items()},
        'opencv_api':'4.10.0'}


def freeze(plan, frozen, comparison, predictions, *, validation_origin, weights_file,
           evaluation_bundle):
    """Freeze one passing validation-selected configuration, before test access.

    predictions maps every prespecified configuration ID to the decoded retained
    ZIP JSON document. Recompute metrics rather than accepting recorded pass flags.
    validation_origin binds independently verified native File copies of both
    output artifacts to the same actual completed validation Run and revision.
    """
    rows = validation_members(frozen, plan)
    held_out = test_members(frozen, plan)
    UUID(validation_origin['run_id']); UUID(validation_origin['revision_id'])
    for name in ('comparison', 'predictions'):
        checked_pin(validation_origin[name], file_required=True)
    weights = checked_pin(weights_file, file_required=True)
    weight_identity = {k:weights[k] for k in ('sha256', 'size_bytes')}
    if (comparison['stage'] != 'actual_validation_candidate_comparison'
            or comparison['weights'] != weight_identity
            or comparison['frozen_dataset_id'] != plan['frozen_dataset_id']
            or comparison['frozen_dataset_sha256'] != plan['frozen_dataset_sha256']
            or comparison['environment'] != expected_environment(plan)
            or comparison['source']['source_commit'] != COMMIT
            or comparison['source']['source_archive_sha256'] != plan['source_archive']['sha256']
            or comparison['sahi']['version'] != VERSION
            or comparison['sahi']['source_sha256'] != plan['sahi_archive']['sha256']
            or comparison['sahi']['license_sha256'] != LICENSE_SHA
            or comparison['validation_selection_complete'] is not True
            or comparison['test_outcomes_inspected'] is not False
            or comparison['final_test_completed'] is not False
            or comparison['scientific_acceptance_established'] is not False):
        raise ValueError('Lock requires complete validation-only evidence from the exact measured contract')
    grid = candidate_grid()
    if comparison['candidate_grid'] != grid or set(predictions) != {c['id'] for c in grid}:
        raise ValueError('Lock requires all32 prespecified validation configurations')
    recorded = comparison['candidates']
    if len(recorded) != len(grid) or [r['configuration'] for r in recorded] != grid:
        raise ValueError('Validation candidate report changed the prespecified grid')
    expected = {r['sample_id']:r for r in rows}
    reports = []
    for config, candidate in zip(grid, recorded):
        payload = predictions[config['id']]
        if payload['configuration'] != config:
            raise ValueError('Retained validation predictions changed their configuration')
        records = payload['plates']
        for plate in records:
            if (plate['sample_id'] not in expected
                    or plate['group_id'] != expected[plate['sample_id']]['group_id']):
                raise ValueError('Validation predictions changed frozen source identity')
        measured = summarize(records, expected_sample_ids=expected)
        if measured != payload['metrics'] or measured != candidate['metrics']:
            raise ValueError('Recorded validation metrics differ from retained predictions')
        if measured['failures']:
            raise ValueError('Every validation configuration must complete every planned plate')
        reports.append((config, measured))
    config, report = choose_validation_candidate(reports)
    if (config != comparison['selected_configuration']
            or report != comparison['selected_validation_metrics'] or not report['acceptance_passed']):
        raise ValueError('Final-test lock needs a passing candidate selected exclusively on validation')
    bundle = checked_bundle(evaluation_bundle)
    if any(bundle['authored_code']['src/'+name] != plan['authored_code'][name]
           for name in UNCHANGED_ALGORITHMS):
        raise ValueError('Final test changed the reviewed metric, source, or input-path algorithms')
    locked_plan = {k:copy.deepcopy(plan[k]) for k in PLAN_KEYS}
    lock = {'format':FORMAT, 'weights':weights, 'configuration':copy.deepcopy(config),
            'validation_origin':copy.deepcopy(validation_origin), 'validation_metrics':report,
            'validation_plan_sha256':digest(plan), 'evaluation_plan':locked_plan,
            'evaluation_bundle':bundle, 'test_images':copy.deepcopy(held_out),
            'metric_plan':{'method':METHOD, 'approved_thresholds':dict(GATES), 'iou_threshold':.5,
                           'bootstrap_replicates':2000, 'bootstrap_seed':20261001,
                           'recreation_wape_tolerance_percentage_points':2.0},
            'excluded_test_images':[], 'test_outcomes_inspected_at_lock':False,
            'scientific_acceptance_established':False,
            'limits':'Source-domain qualification pending actual locked test, replay, and independent training recreation; manual source review required.'}
    return lock


def verify(lock, *, expected_sha256, plan, frozen, evaluation_bundle, weights_file):
    """Fail before test annotations are opened if any execution binding changed."""
    if digest(lock) != expected_sha256 or lock['format'] != FORMAT:
        raise ValueError('Final-test lock byte-independent canonical digest changed')
    if (lock['validation_plan_sha256'] != digest(plan)
            or lock['evaluation_plan'] != {k:plan[k] for k in PLAN_KEYS}
            or lock['evaluation_bundle'] != checked_bundle(evaluation_bundle)
            or lock['weights'] != checked_pin(weights_file, file_required=True)
            or lock['test_images'] != test_members(frozen, plan)
            or lock['configuration'] not in candidate_grid()
            or lock['excluded_test_images'] != []
            or lock['test_outcomes_inspected_at_lock'] is not False
            or lock['scientific_acceptance_established'] is not False
            or lock['metric_plan'] != {'method':METHOD, 'approved_thresholds':GATES, 'iou_threshold':.5,
                                      'bootstrap_replicates':2000, 'bootstrap_seed':20261001,
                                      'recreation_wape_tolerance_percentage_points':2.0}):
        raise ValueError('Final-test execution differs from its pre-outcome lock')
    return copy.deepcopy(lock['configuration'])


def freeze_from_artifacts(plan, *, frozen_path, comparison_path, predictions_path,
                          validation_origin, weights_file, evaluation_bundle):
    """Read bounded, checksum-pinned originals of the actual validation artifacts.

    Native File origin and terminal Run identity still need account-backed readback.
    Archive members are decoded as JSON only; no artifact content is executed.
    """
    frozen = json.loads(pinned_bytes(frozen_path, plan['frozen_split'], 1024*1024))
    comparison = json.loads(pinned_bytes(comparison_path, validation_origin['comparison'], 64*1024*1024))
    raw = pinned_bytes(predictions_path, validation_origin['predictions'], 64*1024*1024)
    expected = {c['id']+'.json' for c in candidate_grid()}
    predictions = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != len(expected) or {m.filename for m in members} != expected
                or sum(m.file_size for m in members) > 128*1024*1024):
            raise ValueError('Validation ZIP must contain exactly32 bounded configuration documents')
        for member in members:
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (member.is_dir() or member.flag_bits & 1 or kind not in (0, stat.S_IFREG)
                    or not 0 < member.file_size <= 32*1024*1024):
                raise ValueError('Validation ZIP member is not bounded regular unencrypted JSON')
            body = archive.read(member)
            if len(body) != member.file_size:
                raise ValueError('Validation ZIP member length changed')
            predictions[member.filename[:-5]] = json.loads(body)
    return freeze(plan, frozen, comparison, predictions, validation_origin=validation_origin,
                  weights_file=weights_file, evaluation_bundle=evaluation_bundle)
