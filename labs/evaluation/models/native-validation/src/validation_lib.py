"""Actual native inference and prespecified validation-only candidate comparison."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

from .core import (MAX_CANDIDATES, canonical_detections, candidate_grid,
                   choose_validation_candidate, original_detection, original_image,
                   validation_members, validation_references)
from .metrics import summarize
from .native_source import extract_native_source, pinned_bytes
from .preprocessing import load_preproc
from .sahi_source import extract_sahi


def identity(path):
    p = Path(path)
    return {'size_bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def merge(values, config, width, height, ObjectPrediction, NMSPostprocess, GreedyNMMPostprocess):
    selected = [v for v in values if v['score'] >= config['confidence']]
    if not selected:return []
    objects = [ObjectPrediction(bbox=v['bbox_xyxy'],category_id=0,category_name='colony',
                                score=v['score'],shift_amount=[0,0],full_shape=[height,width]) for v in selected]
    cls = NMSPostprocess if config['method'] == 'NMS' else GreedyNMMPostprocess
    merged = cls(match_threshold=config['match_threshold'],match_metric=config['match_metric'],class_agnostic=True)(objects)
    return canonical_detections([{'bbox_xyxy':v.bbox.to_xyxy(), 'score':v.score.value} for v in merged],width,height)


def run(inputs, plan, root, *, weights_pin, work_seconds):
    root = Path(root);started = time.monotonic();deadline = started+work_seconds
    for name,pin in plan['authored_code'].items():
        pinned_bytes(Path(__file__).parent/name,pin,128*1024)
    frozen = json.loads(pinned_bytes(inputs['frozen_split'],plan['frozen_split'],1024*1024))
    rows = validation_members(frozen,plan)
    document = json.loads(pinned_bytes(inputs['annotations'],plan['annotations'],16*1024*1024))
    references = validation_references(document,frozen,rows)
    for i,row in enumerate(rows):
        pinned_bytes(inputs[f'image_{i:02d}'],row,20*1024*1024)
    pinned_bytes(inputs['weights'],weights_pin,64*1024*1024)
    native = extract_native_source(inputs['source_archive'],inputs['source_receipt'],root/'native-source',
                                  archive_pin=plan['source_archive'],manifest_pin=plan['source_receipt'])
    sahi = extract_sahi(inputs['sahi_archive'],inputs['sahi_receipt'],root/'sahi-source',
                       archive_pin=plan['sahi_archive'],manifest_pin=plan['sahi_receipt'])
    if any(n == 'yolox' or n.startswith('yolox.') or n == 'sahi' or n.startswith('sahi.') for n in sys.modules):
        raise RuntimeError('Validation cannot reuse cached third-party source modules')
    sys.path[:0] = [str(root/'native-source'),str(root/'sahi-source')]
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import numpy
        import cv2
        import torch
        import torchvision
        from yolox.exp import Exp
        preproc = load_preproc(root/'native-source')
        from sahi.slicing import get_slice_bboxes
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import NMSPostprocess, GreedyNMMPostprocess

        environment = {'python':platform.python_version(),'numpy':numpy.__version__,'torch':str(torch.__version__),
                       'torchvision':str(torchvision.__version__),'cuda_build':str(torch.version.cuda),
                       'cudnn':torch.backends.cudnn.version()}
        if environment != plan['required_environment'] or importlib.metadata.version('numpy') != numpy.__version__:
            raise RuntimeError('Validation differs from the pinned measured native environment')
        package_versions={name:importlib.metadata.version(name) for name in plan['required_package_versions']}
        if any(package_versions[name].split('+')[0] != version for name,version in plan['required_package_versions'].items()):
            raise RuntimeError('Validation dependency installation differs from exact declared package versions')
        if cv2.__version__ != '4.10.0':raise RuntimeError('Validation requires the source-compatible pinned headless OpenCV API')
        runtime_environment={**environment,'package_versions':package_versions,'opencv_api':cv2.__version__}
        if not torch.cuda.is_available():raise RuntimeError('Native validation requires managed CUDA')
        torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
        state = torch.load(inputs['weights'],map_location='cpu',weights_only=True)
        if (state['format'] != 'cfu-native-ema-weights-v1'
                or state['contract_sha256'] != plan['training_contract_sha256']
                or state['training_pool_sha256'] != plan['training_pool_sha256']
                or state['recipe'] != plan['training_recipe'] or state['environment'] != environment):
            raise ValueError('Validation requires retained weights from the exact native training contract')
        exp = Exp();exp.depth=.33;exp.width=.375;exp.num_classes=1
        model = exp.get_model();model.load_state_dict(state['model'],strict=True);model.eval().cuda()
        if any(not torch.isfinite(v).all().item() for v in model.state_dict().values() if v.is_floating_point()):
            raise ValueError('Learned detector weights are nonfinite')
        cache = []
        for i,row in enumerate(rows):
            name = row['sample_id'];reference = references[name];coco = reference['coco_image']
            plate = {'sample_id':name,'group_id':row['group_id'],'image_width':coco['width'],
                     'image_height':coco['height'],'reference_boxes_xyxy':reference['reference_boxes_xyxy'],
                     'source_annotation_ids':reference['source_annotation_ids'],'original':row}
            try:
                if time.monotonic() >= deadline:raise TimeoutError('Validation work budget exhausted before plate')
                image = original_image(inputs[f'image_{i:02d}'],row,coco)
                try:
                    pixels = numpy.asarray(image)[:,:,::-1].copy()
                finally:image.close()
                w,h = coco['width'],coco['height']
                windows = get_slice_bboxes(image_height=h,image_width=w,slice_height=640,slice_width=640,
                                          overlap_height_ratio=.2,overlap_width_ratio=.2,auto_slice_resolution=False)
                predictions = {'raw':[],'tiled':[]};plate_started=time.monotonic()
                for mode,bounds in [('raw',[[0,0,w,h]]),('tiled',windows)]:
                    for left in range(0,len(bounds),8):
                        if time.monotonic() >= deadline:raise TimeoutError('Validation work budget exhausted during plate')
                        chosen=bounds[left:left+8];prepared=[];ratios=[]
                        for x0,y0,x1,y1 in chosen:
                            tensor,ratio=preproc(pixels[y0:y1,x0:x1],(640,640))
                            prepared.append(tensor);ratios.append(ratio)
                        batch=torch.from_numpy(numpy.stack(prepared)).cuda()
                        with torch.inference_mode():decoded=model(batch)
                        if decoded.ndim != 3 or decoded.shape[2] != 6 or not torch.isfinite(decoded).all().item():
                            raise RuntimeError('Native single-class inference returned invalid decoded tensors')
                        outputs=decoded.cpu().tolist()
                        for values,ratio,window in zip(outputs,ratios,chosen):
                            for value in values:
                                detection=original_detection(value,ratio=ratio,window=window,width=w,height=h,confidence=.05)
                                if detection is not None:predictions[mode].append(detection)
                            if len(predictions[mode]) > MAX_CANDIDATES:
                                raise RuntimeError('Plate exceeds candidate bound; no detections are silently truncated')
                        del batch,decoded
                plate.update(status='completed',predictions=predictions,windows=windows,
                             inference_seconds=time.monotonic()-plate_started)
            except Exception as exc:
                plate.update(status='failed',predictions={'raw':[],'tiled':[]},
                             failure={'type':type(exc).__name__,'message':str(exc)})
            cache.append(plate)
        cache_path=root/'validation-decoded-cache.json'
        cache_path.write_text(json.dumps({'stage':'actual_validation_native_decoded_predictions','weights':weights_pin,
                                         'minimum_confidence':.05,'plates':cache},allow_nan=False))
        reports=[];predictions_path=root/'validation-predictions.zip'
        with zipfile.ZipFile(predictions_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for config in candidate_grid():
                records=[]
                for plate in cache:
                    record={k:plate[k] for k in ['sample_id','group_id','image_width','image_height','reference_boxes_xyxy']}
                    try:
                        if plate['status'] != 'completed':raise RuntimeError(plate['failure']['message'])
                        if time.monotonic() >= deadline:raise TimeoutError('Validation work budget exhausted during merging')
                        detections=merge(plate['predictions'][config['mode']],config,plate['image_width'],plate['image_height'],
                                         ObjectPrediction,NMSPostprocess,GreedyNMMPostprocess)
                        record.update(status='completed',detections=detections,predicted_count=len(detections))
                    except Exception as exc:
                        record.update(status='failed',detections=[],predicted_count=None,
                                      failure={'type':type(exc).__name__,'message':str(exc)})
                    records.append(record)
                report=summarize(records,expected_sample_ids=[r['sample_id'] for r in rows])
                reports.append((config,report))
                archive.writestr(config['id']+'.json',json.dumps({'configuration':config,'plates':records,'metrics':report},allow_nan=False))
        selected,selected_report=choose_validation_candidate(reports)
        comparison_path=root/'validation-candidate-comparison.json'
        comparison={'stage':'actual_validation_candidate_comparison','weights':weights_pin,'weights_progress':state['progress'],
                    'frozen_dataset_id':plan['frozen_dataset_id'],'frozen_dataset_sha256':plan['frozen_dataset_sha256'],
                    'environment':runtime_environment,'source':native,'sahi':sahi,'candidate_grid':candidate_grid(),
                    'candidates':[{'configuration':c,'metrics':r} for c,r in reports],
                    'selected_configuration':selected,'selected_validation_metrics':selected_report,
                    'validation_selection_complete':all(r['failures']==0 for _,r in reports),
                    'test_outcomes_inspected':False,'final_test_completed':False,'scientific_acceptance_established':False}
        comparison_path.write_text(json.dumps(comparison,indent=2,allow_nan=False))
        for p in [cache_path,predictions_path,comparison_path]:
            if p.stat().st_size > 64*1024*1024:raise RuntimeError('Validation output exceeds retained64MiB bound')
            p.chmod(0o600)
        return {'receipt':{'stage':'actual_native_validation','plate_count':len(rows),'weights':weights_pin,
                           'environment':runtime_environment,'selected_configuration':selected,
                           'selected_validation_metrics':selected_report,
                           'validation_selection_complete':comparison['validation_selection_complete'],'test_outcomes_inspected':False,
                           'scientific_acceptance_established':False,'artifacts':{n:identity(p) for n,p in
                            [('decoded_cache',cache_path),('predictions',predictions_path),('comparison',comparison_path)]}},
                'cache_path':str(cache_path),'predictions_path':str(predictions_path),'comparison_path':str(comparison_path)}
    finally:
        for path in [str(root/'native-source'),str(root/'sahi-source')]:sys.path.remove(path)
