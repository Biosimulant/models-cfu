"""Independent arithmetic/lineage inspection of retained managed training artifacts.

Does not deserialize checkpoints, run a detector, optimize, or inspect test outcomes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID


def verified_json(path, size, sha):
    raw = path.read_bytes()
    assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha
    return json.loads(raw)


def check(receipt, history, baseline, split):
    prior = baseline['outputs']['train']['receipt']['value']
    parents = {row['file_id'] for row in split['native_dataset_manifest']['members'] if row['split'] == 'train'}
    assert len(parents) == 295
    assert receipt['stage'] == 'native_frozen_training_chunk'
    assert history['stage'] == 'actual_native_training_steps'
    assert receipt['start_progress'] == prior['end_progress'] == history['start']
    for key in ['source', 'contract_sha256', 'recipe', 'environment', 'training_pool']:
        assert receipt[key] == prior[key], key
    assert receipt['targets_truncated'] == history['targets_truncated'] == 0
    assert receipt['test_outcomes_inspected'] is False
    assert receipt['scientific_acceptance_established'] is False
    assert receipt['weights_changed'] is True
    assert receipt['training_pool']['optimizer_validation_or_test_crops'] == 0
    assert receipt['training_pool']['all_frozen_training_parents_covered'] is True
    pool = receipt['training_pool']['crops']
    batch = receipt['recipe']['batch_size']
    assert pool == 9862 and batch == 16
    cursor = dict(history['start'])
    crops = 0
    duration = 0
    losses = []
    for step in history['steps']:
        assert step['before'] == cursor
        assert 0 <= cursor['epoch'] < receipt['recipe']['max_epochs']
        assert 0 <= cursor['cursor'] < pool and cursor['cursor'] % batch == 0
        assert cursor['global_step'] == cursor['epoch'] * math.ceil(pool / batch) + cursor['cursor'] // batch
        count = min(batch, pool - cursor['cursor'])
        for key in ['target_counts', 'crop_sha256', 'parent_file_ids']:
            assert len(step[key]) == count
        assert all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 4096 for v in step['target_counts'])
        assert all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in step['crop_sha256'])
        assert all(v in parents for v in step['parent_file_ids'])
        assert all(math.isfinite(step[key]) and step[key] > 0 for key in ['loss', 'lr', 'seconds'])
        assert step['l1_enabled'] == (cursor['epoch'] >= receipt['recipe']['max_epochs'] - 5)
        nxt = {'epoch': cursor['epoch'], 'cursor': cursor['cursor'] + count, 'global_step': cursor['global_step'] + 1}
        if nxt['cursor'] == pool:
            nxt.update(epoch=nxt['epoch'] + 1, cursor=0)
        assert step['after'] == nxt
        cursor = nxt
        crops += count
        duration += step['seconds']
        losses.append(step['loss'])
    assert cursor == history['end'] == receipt['end_progress']
    assert len(history['steps']) == receipt['steps_executed'] == cursor['global_step'] - history['start']['global_step'] > 0
    assert crops == receipt['crops_consumed']
    assert receipt['last_losses'] == losses[-10:]
    assert math.isclose(duration, receipt['actual_optimizer_seconds'], rel_tol=1e-10, abs_tol=1e-8)
    assert receipt['full_recipe_completed'] == (cursor['epoch'] >= receipt['recipe']['max_epochs'])
    return {'start': history['start'], 'end': cursor, 'steps': len(history['steps']), 'crops': crops, 'all_history_transitions_verified': True, 'all_parent_ids_in_frozen_training_fold': True, 'recipe_contract_pool_unchanged': True, 'targets_truncated': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['results', 'history', 'baseline', 'split', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ['results', 'history']:
        parser.add_argument('--' + name + '-sha256', required=True)
        parser.add_argument('--' + name + '-size', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--revision-id', required=True)
    args = parser.parse_args()
    UUID(args.run_id)
    UUID(args.revision_id)
    results = verified_json(args.results, args.results_size, args.results_sha256)
    history = verified_json(args.history, args.history_size, args.history_sha256)
    baseline = json.loads(args.baseline.read_bytes())
    split = verified_json(args.split, 237988, '81156e7833415b05b39659828d3fc4b8b006b6f2b24e1d4665de654efae6ae3f')
    receipt = results['outputs']['train']['receipt']['value']
    assert receipt['saved_artifacts']['history'] == {'sha256': args.history_sha256, 'size_bytes': args.history_size}
    proof = {'run_id': args.run_id, 'revision_id': args.revision_id, **check(receipt, history, baseline, split), 'receipt_and_history_bytes_independently_verified': True, 'scientific_acceptance_established': False, 'receipt': receipt, 'limitations': 'Baseline must already be independently byte-verified. No local detector, training, checkpoint deserialization or optimizer execution. Crop batch ordering was not independently reconstructed from all prepared archive members; exact managed source and immutable pool binding enforce it. Learning-rate positivity and frozen recipe identity checked; schedule was not independently reconstructed.'}
    args.output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k not in ['receipt', 'limitations']}))


if __name__ == '__main__':
    main()
