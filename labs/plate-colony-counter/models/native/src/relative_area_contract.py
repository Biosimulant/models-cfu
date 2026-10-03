"""The exact predeclared validation extension; no annotations or sample rules."""
from .area_filter import ANCHOR_SCORE, MIN_ANCHORS, RATIOS, filter_detections

BASE_CONFIGURATION = {'id': 'tiled-GREEDYNMM-IOS-0.50-conf0.40', 'mode': 'tiled',
                      'confidence': .4, 'method': 'GREEDYNMM', 'match_metric': 'IOS', 'match_threshold': .5}


def filter_rule():
    return {'ratios': list(RATIOS), 'anchor_score': ANCHOR_SCORE, 'minimum_anchor_count': MIN_ANCHORS,
            'baseline_configuration_id': BASE_CONFIGURATION['id'], 'fallback': 'No filtering below minimum anchors'}


def filtered_grid():
    rule = filter_rule()
    return [{**BASE_CONFIGURATION, 'id': f"{BASE_CONFIGURATION['id']}-relative-area{ratio:.2f}",
             'filter': {**rule, 'minimum_relative_area': ratio}} for ratio in RATIOS]


def apply_config_filter(detections, config):
    if 'filter' not in config:
        return detections
    if config not in filtered_grid():
        raise ValueError('Post-merge filter differs from the five predeclared configurations')
    return filter_detections(detections, config['filter']['minimum_relative_area'])[0]
