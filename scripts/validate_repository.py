"""Verify relocated source/manifests without loading weights or running inference."""
from pathlib import Path
import argparse
import hashlib
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular source file: {path}')
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def validate(migration=False):
    lab_root = ROOT / 'labs/plate-colony-counter'
    lab = yaml.safe_load((lab_root / 'lab.yaml').read_text())
    assert lab['package'] == 'plate-colony-counter'
    assert lab['version'] == '0.1.1'
    assert [p['name'] for p in lab['io']['inputs']] == ['image']
    assert [p['name'] for p in lab['io']['outputs']] == ['count', 'annotated_image', 'receipt']
    model = lab['models'][0]
    module_root = lab_root / model['path']
    pin = identity(module_root / 'artifacts/inference-contract.json')
    assert pin == {'sha256': model['parameters']['contract_sha256'],
                   'size_bytes': model['parameters']['contract_size_bytes']}
    contract = json.loads((module_root / 'artifacts/inference-contract.json').read_bytes())
    source_names = {str(p.relative_to(module_root)) for p in (module_root / 'src').glob('*.py')}
    assert set(contract['authored_code']) == source_names
    for name, expected in contract['authored_code'].items():
        assert identity(module_root / name) == expected, f'Frozen inference source changed: {name}'
    assert contract['configuration']['id'] == 'tiled-GREEDYNMM-IOS-0.50-conf0.40-relative-area0.25'
    assert contract['evidence']['original_scientific_gates_passed'] is False
    manifests = sorted((ROOT / 'labs').rglob('model.yaml'))
    for path in manifests:
        manifest = yaml.safe_load(path.read_text())
        assert manifest.get('package') and manifest.get('version'), str(path)
    checked = 0
    if migration:
        manifest = json.loads((ROOT / 'docs/migration-manifest.json').read_text())
        for record in manifest['files']:
            assert identity(ROOT / record['path']) == {k: record[k] for k in ('sha256', 'size_bytes')}, record['path']
            checked += 1
    print(json.dumps({'model_manifests': len(manifests), 'frozen_source_files': len(source_names),
                      'migration_files_checked': checked, 'scientific_execution': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migration', action='store_true', help='Verify every initial migration byte identity.')
    validate(parser.parse_args().migration)
