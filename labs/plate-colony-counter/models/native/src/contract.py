"""Verify a separately assembled immutable inference release and its assets."""
import json
from pathlib import Path

from .image import identity
from .inference_core import candidate_grid
from .relative_area_contract import filtered_grid

ASSETS = {'source_archive': ('yolox-source.zip', 64 * 1024 * 1024),
          'source_receipt': ('yolox-source.json', 1024 * 1024),
          'sahi_archive': ('sahi-source.zip', 64 * 1024 * 1024),
          'sahi_receipt': ('sahi-source.json', 1024 * 1024),
          'weights': ('ema-weights.pth', 64 * 1024 * 1024)}


def regular_bytes(path, maximum, pin=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('Release asset must be a bounded regular file')
    raw = path.read_bytes()
    if pin is not None and identity(raw) != {k: pin[k] for k in ('sha256', 'size_bytes')}:
        raise ValueError('Release asset differs from its immutable identity')
    return raw


def verify(module_root, contract_pin):
    module_root = Path(module_root).resolve(strict=True)
    raw = regular_bytes(module_root / 'artifacts/inference-contract.json', 128 * 1024, contract_pin)
    contract = json.loads(raw)
    formats = {'cfu-locked-inference-release-v1': candidate_grid(),
               'cfu-locked-inference-release-v2-relative-area': filtered_grid()}
    if (contract['format'] not in formats
            or contract['configuration'] not in formats[contract['format']]
            or contract['coordinate_basis'] != 'EXIF-normalized original image pixels'
            or contract['input_size'] != 640 or contract['overlap'] != .2
            or contract['batch_size'] != 8 or contract['minimum_confidence'] != .05
            or contract['maximum_candidates'] != 20_000):
        raise ValueError('Inference release changes the locked detector semantics')
    code = contract['authored_code']
    names = {str(p.relative_to(module_root)) for p in (module_root / 'src').glob('*.py')}
    if set(code) != names:
        raise ValueError('Inference release source bundle differs')
    for name in sorted(names):
        if not (module_root / name).resolve(strict=True).is_relative_to(module_root):
            raise ValueError('Inference source escaped the release')
        regular_bytes(module_root / name, 128 * 1024, code[name])
    paths = {}
    for name, (filename, maximum) in ASSETS.items():
        path = module_root / 'artifacts' / filename
        if not path.resolve(strict=True).is_relative_to(module_root):
            raise ValueError('Inference asset escaped the release')
        regular_bytes(path, maximum, contract[name]); paths[name] = str(path)
    return contract, paths
