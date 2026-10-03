"""Locked original-frame detector execution; no annotation input or parameter grid."""
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time

from .contract import verify
from .image import annotate, decode
from .inference_core import original_detection
from .hosted_source import extract_hosted_source as extract_native_source
from .postprocess import merge
from .preprocessing import load_preproc
from .sahi_source import extract_sahi


def decoded_predictions(pixels, model, config, preproc, slice_bboxes, numpy, torch, deadline):
    height, width = pixels.shape[:2]
    bounds = ([[0, 0, width, height]] if config['mode'] == 'raw' else
              slice_bboxes(image_height=height, image_width=width, slice_height=640, slice_width=640,
                           overlap_height_ratio=.2, overlap_width_ratio=.2, auto_slice_resolution=False))
    predictions = []
    for left in range(0, len(bounds), 8):
        if time.monotonic() >= deadline:
            raise TimeoutError('Inference work budget exhausted')
        chosen = bounds[left:left + 8]; prepared = []; ratios = []
        for x0, y0, x1, y1 in chosen:
            tensor, ratio = preproc(pixels[y0:y1, x0:x1], (640, 640))
            prepared.append(tensor); ratios.append(ratio)
        batch = torch.from_numpy(numpy.stack(prepared)).cuda()
        with torch.inference_mode(): decoded = model(batch)
        if (decoded.ndim != 3 or decoded.shape[0] != len(chosen) or decoded.shape[2] != 6
                or not torch.isfinite(decoded).all().item()):
            raise RuntimeError('Detector returned invalid single-class decoded tensors')
        for values, ratio, window in zip(decoded.cpu().tolist(), ratios, chosen):
            for value in values:
                detection = original_detection(value, ratio=ratio, window=window,
                                               width=width, height=height, confidence=.05)
                if detection is not None: predictions.append(detection)
            if len(predictions) > 20_000:
                raise RuntimeError('Candidate bound exceeded; no detections were silently truncated')
        del batch, decoded
    return predictions


def run(request, module_root):
    started = time.monotonic(); deadline = started + request['work_seconds']
    contract, assets = verify(module_root, request['contract_pin'])
    root = Path(request['root']).resolve(strict=True)
    image, input_identity = decode(request['image'])
    native_dir = root / 'native-source'; sahi_dir = root / 'sahi-source'
    native = extract_native_source(assets['source_archive'], assets['source_receipt'], native_dir,
                                  archive_pin=contract['source_archive'], manifest_pin=contract['source_receipt'])
    sahi = extract_sahi(assets['sahi_archive'], assets['sahi_receipt'], sahi_dir,
                        archive_pin=contract['sahi_archive'], manifest_pin=contract['sahi_receipt'])
    if any(n == 'yolox' or n.startswith('yolox.') or n == 'sahi' or n.startswith('sahi.') for n in sys.modules):
        raise RuntimeError('Inference must run in a fresh source-isolated interpreter')
    sys.path[:0] = [str(native_dir), str(sahi_dir)]
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import cv2
        import numpy
        import torch
        import torchvision
        from yolox.exp import Exp
        from sahi.slicing import get_slice_bboxes
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import NMSPostprocess, GreedyNMMPostprocess

        environment = {'python': platform.python_version(), 'numpy': numpy.__version__,
                       'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                       'cuda_build': str(torch.version.cuda), 'cudnn': torch.backends.cudnn.version()}
        versions = {name: importlib.metadata.version(name) for name in contract['required_package_versions']}
        if (environment != contract['required_environment']
                or any(versions[name].split('+')[0] != expected
                       for name, expected in contract['required_package_versions'].items())
                or cv2.__version__ != '4.10.0' or not torch.cuda.is_available()):
            raise RuntimeError('Hosted inference differs from the pinned measured CUDA environment')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
        state = torch.load(assets['weights'], map_location='cpu', weights_only=True)
        if (state['format'] != 'cfu-native-ema-weights-v1'
                or state['contract_sha256'] != contract['training_contract_sha256']
                or state['training_pool_sha256'] != contract['training_pool_sha256']
                or state['recipe'] != contract['training_recipe'] or state['environment'] != environment
                or state['progress'] != contract['weights_progress']):
            raise ValueError('Hosted weights differ from the selected native training state')
        exp = Exp(); exp.depth = .33; exp.width = .375; exp.num_classes = 1
        model = exp.get_model(); model.load_state_dict(state['model'], strict=True); model.eval().cuda()
        if any(not torch.isfinite(v).all().item() for v in model.state_dict().values() if v.is_floating_point()):
            raise ValueError('Hosted learned weights are nonfinite')
        pixels = numpy.asarray(image)[:, :, ::-1].copy()
        predictions = decoded_predictions(pixels, model, contract['configuration'], load_preproc(native_dir),
                                          get_slice_bboxes, numpy, torch, deadline)
        if time.monotonic() >= deadline: raise TimeoutError('Inference budget exhausted before merging')
        width, height = image.size
        detections = merge(predictions, contract['configuration'], width, height,
                           ObjectPrediction, NMSPostprocess, GreedyNMMPostprocess)
        annotated_path = root / 'annotated-colonies.png'
        rendered = annotate(image, detections, annotated_path)
        receipt = {'stage': 'actual_locked_uploaded_image_inference', 'count_unit': 'colonies',
                   'input': input_identity, 'weights': contract['weights'],
                   'configuration': contract['configuration'], 'detections': detections,
                   'environment': {**environment, 'package_versions': versions, 'opencv_api': cv2.__version__},
                   'inference_contract': request['contract_pin'], 'seconds': time.monotonic() - started,
                   'provenance': {'evidence': contract['evidence'], 'weights_progress': state['progress'],
                                  'native_source_commit': native['source_commit'], 'sahi_version': sahi['version'],
                                  'coordinate_basis': contract['coordinate_basis'],
                                  'source_archives': {n: contract[n] for n in ['source_archive', 'sahi_archive']}},
                   'warnings': ['Experimental model: the held-out 95th-percentile count error was 22.22%, above the approved 20% limit.',
                                'Research assistance: inspect every detection before using the count.',
                                'New laboratories/acquisition conditions, blank controls and merged growth are unvalidated.',
                                'Scores are detector scores, not calibrated count uncertainty.',
                                'No species, CFU/mL, clinical or sterility inference.'],
                   'annotated_image': rendered}
        return {'count': len(detections), 'receipt': receipt, 'annotated_path': str(annotated_path)}
    finally:
        image.close()
        for path in [str(native_dir), str(sahi_dir)]: sys.path.remove(path)
