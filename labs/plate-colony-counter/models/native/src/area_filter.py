"""Image-wide prediction-only relative-area filter; no reference or sample inputs."""
import math
from statistics import median

RATIOS = (0.0, 0.1, 0.2, 0.25, 0.5)
ANCHOR_SCORE = 0.7
MIN_ANCHORS = 3


def filter_detections(detections, ratio):
    if isinstance(ratio, bool) or ratio not in RATIOS:
        raise ValueError('Filter ratio is outside the predeclared five-candidate grid')
    areas = []
    for detection in detections:
        box = detection['bbox_xyxy']
        score = detection['score']
        if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in [*box, score])
                or not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('Filter requires finite positive-area detections and scores')
        area = (box[2] - box[0]) * (box[3] - box[1])
        if not math.isfinite(area):
            raise ValueError('Detection area overflow')
        areas.append(area)
    anchors = [area for area, d in zip(areas, detections) if d['score'] >= ANCHOR_SCORE]
    anchor = median(anchors) if len(anchors) >= MIN_ANCHORS else None
    threshold = ratio * anchor if anchor is not None else 0.0
    kept = [i for i, area in enumerate(areas) if area >= threshold]
    return [dict(detections[i]) for i in kept], {
        'anchor_score': ANCHOR_SCORE, 'minimum_anchor_count': MIN_ANCHORS,
        'anchor_count': len(anchors), 'anchor_median_area_px2': anchor,
        'minimum_area_px2': threshold, 'minimum_relative_area': ratio,
        'fallback_no_filter': anchor is None, 'kept_indices': kept,
        'removed_indices': [i for i in range(len(detections)) if i not in set(kept)],
    }
