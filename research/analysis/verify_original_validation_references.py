"""Inspect MCP-owned validation artifacts against original COCO bytes; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def pinned(path, size, digest):
    raw = path.read_bytes()
    assert len(raw) == size, (path.name, len(raw), size)
    assert hashlib.sha256(raw).hexdigest() == digest, path.name
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prediction-size', type=int, required=True)
    parser.add_argument('--prediction-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--global-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts_dir
    document = json.loads(pinned(root / 'verified-original-annotations.json', 6386192,
        '062f08c159004e89b96f93bea05d95403b01b1fcfb73384f6eda21ba9842e12c'))
    split = json.loads(pinned(root / 'verified-frozen-original-split.json', 237988,
        '81156e7833415b05b39659828d3fc4b8b006b6f2b24e1d4665de654efae6ae3f'))
    selected = {m['sample_id']: m['group_id'] for m in
                split['native_dataset_manifest']['members']
                if m['role'] == 'image' and m['split'] == 'validation'}
    assert len(selected) == 37
    images = {i['id']: i for i in document['images'] if i['file_name'] in selected}
    assert len(images) == 37 and {i['file_name'] for i in images.values()} == set(selected)
    boxes = {name: [] for name in selected}
    ids = set()
    for annotation in document['annotations']:
        if annotation['image_id'] not in images:
            continue  # Do not inspect other folds' reference values.
        assert annotation['id'] not in ids
        ids.add(annotation['id'])
        x, y, w, h = annotation['bbox']
        boxes[images[annotation['image_id']]['file_name']].append([x, y, x + w, y + h])
    assert sum(map(len, boxes.values())) == 5267
    by_name = {i['file_name']: i for i in images.values()}
    pinned(args.predictions, args.prediction_size, args.prediction_sha256)
    checked = references = 0
    with ZipFile(args.predictions) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 32
        for member in archive.namelist():
            assert '/' not in member and '\\' not in member and member.endswith('.json')
            value = json.loads(archive.read(member))
            plates = value['plates']
            assert len(plates) == 37 and {p['sample_id'] for p in plates} == set(selected)
            for plate in plates:
                name = plate['sample_id']
                assert plate['reference_boxes_xyxy'] == boxes[name], (member, name)
                assert plate['group_id'] == selected[name]
                assert plate['image_width'] == by_name[name]['width']
                assert plate['image_height'] == by_name[name]['height']
                references += len(boxes[name])
                checked += 1
    assert checked == 1184 and references == 168544
    proof = {
        'kind': 'original_coco_validation_reference_equality_audit',
        'run_id': args.run_id, 'global_step': args.global_step,
        'original_coco_file_id': 'd4e1520c-7d74-4a8a-bfcf-e16dbe4a6129',
        'original_coco_sha256': '062f08c159004e89b96f93bea05d95403b01b1fcfb73384f6eda21ba9842e12c',
        'prediction_zip_sha256': args.prediction_sha256,
        'prediction_zip_size_bytes': args.prediction_size,
        'unique_validation_plates': 37, 'original_validation_annotations': 5267,
        'plate_configuration_pairs_checked': checked,
        'reference_boxes_checked_with_repeats_across_configurations': references,
        'all_original_boxes_dimensions_and_group_ids_exact': True,
        'source_edge_crossings_and_zero_area_boxes_preserved': True,
        'limits': 'Original source labels are retained, not independently judged biologically correct. No other-fold reference values inspected, final-test predictions accessed, detector inference, training, checkpoint deserialization or annotation edits.',
    }
    args.output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
