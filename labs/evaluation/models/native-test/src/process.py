"""Run the pinned native task in a fresh interpreter after dependency resolution."""

import json
from pathlib import Path
import subprocess
import sys

MAX_RECEIPT_BYTES = 1024 * 1024


def absolute_inputs(values):
    return {name: str(Path(value).resolve(strict=True)) for name, value in values.items()}


def read_receipt(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_RECEIPT_BYTES:
        raise RuntimeError('Native worker has no bounded regular receipt file')
    receipt = json.loads(path.read_bytes())
    if (not isinstance(receipt, dict) or receipt.get('stage') != 'native_gpu_training_compatibility_benchmark'
            or receipt.get('scientific_acceptance_established') is not False
            or receipt.get('test_outcomes_inspected') is not False
            or not isinstance(receipt.get('cases'), list)
            or not any(c.get('status') == 'measured' for c in receipt['cases'])):
        raise RuntimeError('Native worker receipt is incomplete or changes the benchmark contract')
    return receipt


def launch(request, root, module_root):
    root, module_root = Path(root).resolve(), Path(module_root).resolve()
    request_path = root/'worker-request.json'
    request_path.write_text(json.dumps(request))
    request_path.chmod(0o600)
    receipt_path = root/'worker-receipt.json'
    log_path = root/'worker.log'
    with log_path.open('wb') as log:
        result = subprocess.run([sys.executable, '-m', 'src.worker', str(request_path), str(receipt_path)],
                                cwd=module_root, stdout=log, stderr=subprocess.STDOUT, timeout=540, check=False)
    if result.returncode:
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 12000))
            tail = log.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Fresh native benchmark worker exited {result.returncode}: {tail}')
    return read_receipt(receipt_path)
