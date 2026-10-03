"""Unchanged original-frame detection contracts from the locked evaluator."""
import math


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
