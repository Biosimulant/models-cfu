"""Validation-only original-frame contracts, independent of detector execution."""
from __future__ import annotations

import hashlib
import io
import json
import math

from PIL import Image, ImageOps

from .geometry import index_coco
from .native_source import pinned_bytes

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
MAX_CANDIDATES = 20_000


def validation_members(frozen, plan):
    manifest = frozen['native_dataset_manifest']
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != plan['frozen_dataset_sha256']:
        raise ValueError('Validation requires the exact frozen dataset')
    members = [m for m in manifest['members'] if m['role'] == 'image']
    groups = {}
    for row in members:
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Every source image needs its frozen split')
        old = groups.setdefault(row['group_id'], row['split'])
        if old != row['split']:
            raise ValueError('Related originals cross frozen splits')
    rows = sorted((r for r in members if r['split'] == 'validation'), key=lambda r:r['sample_id'])
    if rows != plan['validation_images'] or len(rows) != 37:
        raise ValueError('Validation must cover every exact frozen validation image')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated validation image identity')
    return rows


def validation_references(document, frozen, rows):
    # Other-split outcome values are not examined by index_coco.
    names = {r['sample_id'] for r in frozen['native_dataset_manifest']['members'] if r['role'] == 'image'}
    selected = {r['sample_id'] for r in rows}
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('This runner cannot inspect test or training outcomes')
    images, annotations = index_coco(document, names, selected_names=selected)
    references = {}
    for name in selected:
        boxes = []
        for a in annotations[name]:
            x, y, w, h = a['bbox']
            # Source edge crossings and zero-area references remain unchanged.
            boxes.append([x, y, x+w, y+h])
        references[name] = {'coco_image':images[name], 'reference_boxes_xyxy':boxes,
                            'source_annotation_ids':[a['id'] for a in annotations[name]]}
    return references


def original_image(path, pin, coco):
    raw = pinned_bytes(path, pin, MAX_SOURCE_BYTES)
    with Image.open(io.BytesIO(raw)) as encoded:
        if encoded.format != 'JPEG' or encoded.width*encoded.height > MAX_SOURCE_PIXELS:
            raise ValueError('Validation original is outside the offline image contract')
        orientation = encoded.getexif().get(274, 1)
        if isinstance(orientation, bool) or orientation not in range(1, 9):
            raise ValueError('Validation original has an invalid EXIF orientation')
        normalized = ImageOps.exif_transpose(encoded)
        try:
            if normalized.size != (coco['width'], coco['height']):
                raise ValueError('Normalized original dimensions differ from the annotation frame')
            normalized.load()
            return normalized.convert('RGB')
        finally:
            normalized.close()


def original_detection(row, *, ratio, window, width, height, confidence):
    if (len(row) != 6 or any(not math.isfinite(float(v)) for v in row)
            or not 0 <= row[4] <= 1 or not 0 <= row[5] <= 1
            or not math.isfinite(ratio) or ratio <= 0):
        raise ValueError('Native detector output is nonfinite or outside its single-class contract')
    cx, cy, bw, bh, obj, cls = map(float, row)
    if bw <= 0 or bh <= 0:
        raise ValueError('Native detector returned nonpositive geometry')
    score = obj*cls
    if score < confidence:
        return None
    wx0, wy0, wx1, wy1 = window
    if not 0 <= wx0 < wx1 <= width or not 0 <= wy0 < wy1 <= height:
        raise ValueError('Inference window is outside the original frame')
    left = max(wx0, wx0+(cx-bw/2)/ratio)
    top = max(wy0, wy0+(cy-bh/2)/ratio)
    right = min(wx1, wx0+(cx+bw/2)/ratio)
    bottom = min(wy1, wy0+(cy+bh/2)/ratio)
    if right <= left or bottom <= top:
        return None  # Padded-region predictions have no original-frame area.
    return {'bbox_xyxy':[left, top, right, bottom], 'score':score, 'class_id':0}


def canonical_detections(values, width, height):
    result = []
    for d in values:
        bbox = list(map(float, d['bbox_xyxy']))
        score = float(d['score'])
        if (len(bbox) != 4 or any(not math.isfinite(v) for v in bbox)
                or not 0 <= bbox[0] < bbox[2] <= width
                or not 0 <= bbox[1] < bbox[3] <= height
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Postprocessing returned an invalid original-frame detection')
        result.append({'bbox_xyxy':bbox, 'score':score, 'class_id':0})
    return sorted(result, key=lambda d:(-d['score'], d['bbox_xyxy']))


def candidate_grid():
    candidates = []
    for confidence in [.05, .1, .2, .3, .4, .5, .6, .7]:
        methods = [('raw', 'NMS', 'IOU', .5), ('tiled', 'NMS', 'IOU', .5),
                   ('tiled', 'GREEDYNMM', 'IOS', .5), ('tiled', 'GREEDYNMM', 'IOS', .7)]
        for mode, method, metric, threshold in methods:
            candidates.append({'id':f'{mode}-{method}-{metric}-{threshold:.2f}-conf{confidence:.2f}',
                               'mode':mode, 'confidence':confidence, 'method':method,
                               'match_metric':metric, 'match_threshold':threshold})
    return candidates


def choose_validation_candidate(reports):
    if not reports:
        raise ValueError('No recorded validation candidates')
    def rank(pair):
        config, report = pair
        return (report['failures'], not report['acceptance_passed'],
                report['wape'] if report['wape'] is not None else math.inf,
                report['p95_symmetric_error'],
                -report['recall'] if report['recall'] is not None else math.inf,
                -report['precision'] if report['precision'] is not None else math.inf, config['id'])
    return min(reports, key=rank)


def locked_test_references(document, frozen, rows):
    """Read test outcomes only after the caller verifies its retained pre-test lock."""
    if len(rows) != 37 or any(r['split'] != 'test' for r in rows):
        raise ValueError('Locked test must cover exactly37 frozen test originals')
    names = {r['sample_id'] for r in frozen['native_dataset_manifest']['members'] if r['role'] == 'image'}
    images, annotations = index_coco(document,names,selected_names={r['sample_id'] for r in rows})
    return {r['sample_id']:{'coco_image':images[r['sample_id']],
            'reference_boxes_xyxy':[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]
                                    for a in annotations[r['sample_id']]],
            'source_annotation_ids':[a['id'] for a in annotations[r['sample_id']]]} for r in rows}
