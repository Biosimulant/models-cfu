"""Finite uploaded-image adapter; release settings are fixed, never caller-tuned."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec
from PIL import Image

from .contract import regular_bytes, verify
from .image import MAX_ANNOTATED_BYTES, identity
from .inference_core import canonical_detections

SCHEMA = {'stage': 'str', 'count_unit': 'str', 'input': 'json', 'weights': 'json',
          'configuration': 'json', 'detections': 'json', 'environment': 'json',
          'inference_contract': 'json', 'provenance': 'json', 'seconds': 'float',
          'warnings': 'json', 'annotated_image': 'json'}


def outputs_checked(outputs, root, contract_pin, contract):
    receipt = outputs['receipt']; root = Path(root).resolve(strict=True)
    if (type(outputs['count']) is not int or outputs['count'] < 0
            or receipt['stage'] != 'actual_locked_uploaded_image_inference'
            or receipt['count_unit'] != 'colonies'
            or receipt['inference_contract'] != contract_pin
            or receipt['weights'] != contract['weights']
            or receipt['configuration'] != contract['configuration']):
        raise RuntimeError('Inference returned an invalid count or changed release identity')
    width, height = receipt['input']['normalized_dimensions']
    detections = canonical_detections(receipt['detections'], width, height)
    if detections != receipt['detections'] or len(detections) != outputs['count']:
        raise RuntimeError('Count differs from the inspectable original-frame detections')
    path = Path(outputs['annotated_path'])
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise RuntimeError('Annotated output escaped the invocation')
    raw = regular_bytes(path, MAX_ANNOTATED_BYTES)
    if identity(raw) != {k: receipt['annotated_image'][k] for k in ('sha256', 'size_bytes')}:
        raise RuntimeError('Annotated output differs from its retained receipt')
    with Image.open(path) as encoded:
        if (encoded.format != 'PNG'
                or list(encoded.size) != [receipt['annotated_image']['display_dimensions'][0],
                                         receipt['annotated_image']['display_dimensions'][1] + 64]):
            raise RuntimeError('Annotated output differs from its declared display frame')
        encoded.load()
    return outputs


class CountVisibleColonies(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, contract_sha256='', contract_size_bytes=0, work_seconds=180):
        if type(work_seconds) is not int or not 30 <= work_seconds <= 600:
            raise ValueError('Inference work time must be30..600seconds')
        self.pin = {'sha256': contract_sha256, 'size_bytes': contract_size_bytes}
        self.work_seconds = work_seconds
        self._result = None

    def inputs(self):
        return {'image': SignalSpec.scalar(dtype='str', value_type='file', format='image')}

    def outputs(self):
        return {'count': SignalSpec.scalar(dtype='int', emitted_unit='1'),
                'receipt': SignalSpec.record(schema=SCHEMA, emitted_unit='1'),
                'annotated_path': SignalSpec.scalar(dtype='str', value_type='file', format='png')}

    def execute(self, inputs, *, context):
        self._result = None
        if set(inputs) != {'image'}:
            raise ValueError('Inference accepts one uploaded image; weights/configuration are immutable')
        if (type(self.pin['size_bytes']) is not int or not 0 < self.pin['size_bytes'] <= 128 * 1024
                or not isinstance(self.pin['sha256'], str) or len(self.pin['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in self.pin['sha256'])):
            raise ValueError('Inference requires an exact assembled release contract')
        module_root = Path(__file__).resolve().parent.parent
        contract, _ = verify(module_root, self.pin)
        value = inputs['image'].value if isinstance(inputs['image'], BioSignal) else inputs['image']
        path = Path(value)
        if path.is_symlink(): raise ValueError('Uploaded image cannot be a symbolic link')
        path = path.resolve(strict=True)
        output_dir = Path.cwd() / 'outputs'
        output_dir.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='cfu-inference-', dir=output_dir)); root.chmod(0o700)
        request = root / 'request.json'; result = root / 'result.json'; log = root / 'worker.log'
        request.write_text(json.dumps({'image': str(path), 'root': str(root.resolve()),
                                       'contract_pin': self.pin, 'work_seconds': self.work_seconds}))
        request.chmod(0o600)
        with log.open('wb') as stream:
            done = subprocess.run([sys.executable, '-m', 'src.worker', str(request), str(result)],
                                  cwd=module_root, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=self.work_seconds + 90, check=False)
        if done.returncode:
            # No default plausible count and no embedded user paths in public failure text.
            raise RuntimeError('Colony inference failed; no count was produced')
        self._result = outputs_checked(json.loads(regular_bytes(result, 4 * 1024 * 1024)),
                                       root, self.pin, contract)
        return self._result

    def visualize(self):
        if self._result is None:
            return []
        result = self._result
        return [
            {'schema_version': '1', 'render': 'image',
             'data': {'source': {'kind': 'artifact', 'path': result['annotated_path']},
                      'alt': f"Plate image with {result['count']} detected visible colonies outlined",
                      'caption': 'Inspect the outlined detections before using the count.'}},
            {'schema_version': '1', 'render': 'text',
             'data': {'text': f"Detected visible colonies: {result['count']}. "
                      + ' '.join(result['receipt']['warnings'])}},
        ]
