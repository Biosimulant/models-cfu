"""Inspect retained MCP-owned detection geometry; no model execution or relabeling."""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from zipfile import ZipFile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.artifacts_dir
ZIP = ROOT / 'native-validation-step10551-verified-predictions-20261003.zip'
raw = ZIP.read_bytes()
assert len(raw) == 10959730
assert hashlib.sha256(raw).hexdigest() == '852d5346393a0339705e82419b74d890c7cef14219c7558699b9fe6e38154693'


def maximum_with_cover(predictions, references):
    p = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    r = np.asarray(references, dtype=np.float64).reshape(-1, 4)
    overlap = np.maximum(0, np.minimum(p[:, None, 2:], r[None, :, 2:]) - np.maximum(p[:, None, :2], r[None, :, :2])).prod(axis=2)
    area_p = np.maximum(0, p[:, 2:] - p[:, :2]).prod(axis=1)
    area_r = np.maximum(0, r[:, 2:] - r[:, :2]).prod(axis=1)
    union = area_p[:, None] + area_r[None, :] - overlap
    ratio = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    adjacency = [np.flatnonzero(row >= .5).tolist() for row in ratio]
    pair_p, pair_r = [-1] * len(p), [-1] * len(r)
    for root in range(len(p)):
        if pair_p[root] >= 0:
            continue
        queue, visited_p, previous_r = deque([root]), {root}, {}
        endpoint = None
        while queue and endpoint is None:
            u = queue.popleft()
            for v in adjacency[u]:
                if v in previous_r:
                    continue
                previous_r[v] = u
                if pair_r[v] < 0:
                    endpoint = v
                    break
                nxt = pair_r[v]
                if nxt not in visited_p:
                    visited_p.add(nxt)
                    queue.append(nxt)
        if endpoint is not None:
            v = endpoint
            while v >= 0:
                u = previous_r[v]
                old = pair_p[u]
                pair_p[u], pair_r[v] = v, u
                v = old
    # A vertex cover of the same size as the matching certifies maximal cardinality.
    reach_p = {u for u, v in enumerate(pair_p) if v < 0}
    reach_r, queue = set(), deque(reach_p)
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v == pair_p[u] or v in reach_r:
                continue
            reach_r.add(v)
            matched = pair_r[v]
            if matched >= 0 and matched not in reach_p:
                reach_p.add(matched)
                queue.append(matched)
    cover_p = set(range(len(p))) - reach_p
    assert all(u in cover_p or v in reach_r for u, neighbors in enumerate(adjacency) for v in neighbors)
    matching = sum(v >= 0 for v in pair_p)
    assert matching == len(cover_p) + len(reach_r)
    assert len({v for v in pair_p if v >= 0}) == matching
    assert all(v in adjacency[u] and pair_r[v] == u for u, v in enumerate(pair_p) if v >= 0)
    return matching, sum(map(len, adjacency))


checked, edges, configurations = 0, 0, []
with ZipFile(ZIP) as archive:
    for member in archive.namelist():
        value = json.loads(archive.read(member))
        rows = {row['sample_id']: row for row in value['metrics']['plates']}
        total_tp = 0
        for plate in value['plates']:
            result, n_edges = maximum_with_cover([d['bbox_xyxy'] for d in plate['detections']], plate['reference_boxes_xyxy'])
            assert result == rows[plate['sample_id']]['tp'], (member, plate['sample_id'], result, rows[plate['sample_id']]['tp'])
            total_tp += result
            checked += 1
            edges += n_edges
        assert total_tp == value['metrics']['tp']
        configurations.append({'configuration_id': value['configuration']['id'], 'plates_checked': len(value['plates']), 'maximum_cardinality_tp': total_tp})
assert len(configurations) == 32 and checked == 1184
comparison = []
for step in [8744, 10551]:
    c = json.loads((ROOT / f'native-validation-step{step}-verified-comparison-20261003.json').read_text())
    with ZipFile(ROOT / f'native-validation-step{step}-verified-predictions-20261003.zip') as archive:
        selected = json.loads(archive.read(c['selected_configuration']['id'] + '.json'))
    rows = selected['metrics']['plates']
    worst = sorted(rows, key=lambda row: row['symmetric_error'], reverse=True)
    fields = ['sample_id', 'group_id', 'reference_count', 'predicted_count', 'absolute_error', 'symmetric_error', 'tp', 'fp', 'fn']
    comparison.append({'global_step': step, 'configuration': c['selected_configuration'], 'selected_p95_symmetric_error': selected['metrics']['p95_symmetric_error'], 'plates_above_20pct_error': sum(row['symmetric_error'] > .2 for row in rows), 'worst_plates': [{k: row[k] for k in fields} for row in worst[:7]]})
proof = {'kind': 'independent_maximum_matching_certificate_and_validation_tail_review', 'run_id': 'ed6e58ff-5044-4801-93f9-2a9f0a5044ad', 'prediction_zip_size_bytes': len(raw), 'prediction_zip_sha256': hashlib.sha256(raw).hexdigest(), 'configuration_count': 32, 'plate_configuration_pairs_checked': checked, 'eligible_iou_edges_checked': edges, 'all_reported_tp_equal_independent_maximum': True, 'minimum_vertex_cover_certificate_equal_matching_size_for_every_plate': True, 'configurations': configurations, 'selected_checkpoint_comparison': comparison, 'scope': 'Independent arithmetic and original-frame geometry inspection of previously byte-verified MCP-owned artifacts; no inference, checkpoint deserialization, training, annotation edits, or final-test outcome access.', 'limits': 'Counts on difficult sparse plates remain inaccurate. Aggregate matching validity does not establish accuracy at each plate, external laboratory generalization, or scientific release acceptance. Earlier and later selected configurations have different frozen-grid confidence thresholds; their comparison is not a controlled same-threshold training-effect estimate.'}
text = json.dumps(proof, indent=2) + '\n'
args.output.write_text(text)
print(json.dumps({k: v for k, v in proof.items() if k not in ['configurations', 'selected_checkpoint_comparison']}))
print(json.dumps(comparison))
