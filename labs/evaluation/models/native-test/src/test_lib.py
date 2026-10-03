"""Single-configuration locked test, saved-model replay and recreation evaluation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from .core import (MAX_CANDIDATES, canonical_detections, locked_test_references,
                   original_detection, original_image)
from .lock import digest, test_members, verify
from .metrics import recreation_check, replay_check, summarize
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


def checked_baseline(baseline, lock, records, phase):
    if (baseline['stage'] != 'actual_locked_test_evaluation' or baseline['phase'] != 'primary'
            or baseline['lock_digest'] != digest(lock)
            or baseline['configuration'] != lock['configuration']
            or baseline['weights'] != {k:lock['weights'][k] for k in ('sha256','size_bytes')}
            or baseline['evaluation_bundle'] != lock['evaluation_bundle']):
        raise ValueError('Comparison baseline differs from the locked primary test')
    reference_keys = ['sample_id','group_id','image_width','image_height','reference_boxes_xyxy']
    def reference_rows(values):
        return [{k:r[k] for k in reference_keys} for r in sorted(values,key=lambda r:r['sample_id'])]
    if reference_rows(baseline['plates']) != reference_rows(records):
        raise ValueError('Comparison baseline changed the frozen original references')
    measured = summarize(baseline['plates'],expected_sample_ids=[r['sample_id'] for r in records])
    if measured != baseline['metrics']:
        raise ValueError('Baseline metrics differ from actual retained baseline predictions')
    if phase not in {'replay','recreation'}:
        raise ValueError('Only replay/recreation compare a prior test')
    return measured


def run(inputs, plan, root, *, weights_pin, work_seconds, lock_pin, lock_digest,
        evaluation_bundle, phase, baseline_pin=None):
    root = Path(root);started = time.monotonic();deadline = started+work_seconds
    for name,pin in evaluation_bundle['authored_code'].items():
        pinned_bytes(Path(__file__).parent.parent/name,pin,128*1024)
    frozen = json.loads(pinned_bytes(inputs['frozen_split'],plan['frozen_split'],1024*1024))
    lock = json.loads(pinned_bytes(inputs['test_lock'],lock_pin,8*1024*1024))
    config = verify(lock,expected_sha256=lock_digest,plan=plan,frozen=frozen,
                    evaluation_bundle=evaluation_bundle,weights_file=lock['weights'])
    if phase not in {'primary','replay','recreation'}:
        raise ValueError('Unknown locked evaluation phase')
    if phase != 'recreation' and weights_pin != {k:lock['weights'][k] for k in ('sha256','size_bytes')}:
        raise ValueError('Primary/replay must consume the exact locked learned weights')
    if ((phase == 'primary' and ('baseline' in inputs or baseline_pin is not None))
            or (phase != 'primary' and ('baseline' not in inputs or baseline_pin is None))):
        raise ValueError('Replay/recreation require one exact retained primary test baseline')
    baseline = (json.loads(pinned_bytes(inputs['baseline'],baseline_pin,64*1024*1024))
                if phase != 'primary' else None)
    rows = test_members(frozen,plan)
    # Verify actual input/source/weight bytes, not only declared pins, before outcomes.
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
        # Actual source, learned state and measured runtime now match the lock.
        document = json.loads(pinned_bytes(inputs['annotations'],plan['annotations'],16*1024*1024))
        references = locked_test_references(document,frozen,rows)
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
                bounds = [[0,0,w,h]] if config['mode'] == 'raw' else windows
                for mode,bounds in [(config['mode'],bounds)]:
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
        cache_path=root/'locked-test-decoded-cache.json'
        cache_path.write_text(json.dumps({'stage':'actual_locked_test_native_decoded_predictions','weights':weights_pin,
                                         'minimum_confidence':.05,'plates':cache},allow_nan=False))
        records=[]
        for plate in cache:
            record={k:plate[k] for k in ['sample_id','group_id','image_width','image_height','reference_boxes_xyxy']}
            try:
                if plate['status'] != 'completed':raise RuntimeError(plate['failure']['message'])
                if time.monotonic() >= deadline:raise TimeoutError('Locked test work budget exhausted during merging')
                detections=merge(plate['predictions'][config['mode']],config,plate['image_width'],plate['image_height'],
                                 ObjectPrediction,NMSPostprocess,GreedyNMMPostprocess)
                record.update(status='completed',detections=detections,predicted_count=len(detections))
            except Exception as exc:
                record.update(status='failed',detections=[],predicted_count=None,
                              failure={'type':type(exc).__name__,'message':str(exc)})
            records.append(record)
        report=summarize(records,expected_sample_ids=[r['sample_id'] for r in rows])
        comparison_result={}
        if baseline is not None:
            baseline_metrics=checked_baseline(baseline,lock,records,phase)
            if baseline['environment'] != runtime_environment:
                raise ValueError('Replay/recreation baseline changed the measured dependency environment')
            comparison_result=(replay_check(baseline_metrics,report) if phase == 'replay'
                               else recreation_check(baseline_metrics,report))
            if phase == 'replay':
                def boxes(values):
                    return [(r['sample_id'],r['detections']) for r in sorted(values,key=lambda r:r['sample_id'])]
                comparison_result['identical_original_frame_detections'] = boxes(baseline['plates']) == boxes(records)
        predictions_path=root/'locked-test-predictions.json'
        comparison={'stage':'actual_locked_test_evaluation','phase':phase,'lock_digest':lock_digest,
                    'configuration':config,'weights':weights_pin,'weights_progress':state['progress'],
                    'frozen_dataset_id':plan['frozen_dataset_id'],'frozen_dataset_sha256':plan['frozen_dataset_sha256'],
                    'environment':runtime_environment,'source':native,'sahi':sahi,'evaluation_bundle':evaluation_bundle,
                    'plates':records,'metrics':report,'comparison':comparison_result,
                    'test_outcomes_inspected':True,'final_test_completed':report['failures']==0,
                    'independent_training_lineage_verified_by_this_worker':False,
                    'scientific_acceptance_established':False}
        predictions_path.write_text(json.dumps(comparison,allow_nan=False))
        comparison_path=root/'locked-test-summary.json'
        summary={k:v for k,v in comparison.items() if k != 'plates'}
        comparison_path.write_text(json.dumps(summary,indent=2,allow_nan=False))
        for p in [cache_path,predictions_path,comparison_path]:
            if p.stat().st_size > 64*1024*1024:raise RuntimeError('Validation output exceeds retained64MiB bound')
            p.chmod(0o600)
        return {'receipt':{'stage':'actual_locked_test_evaluation','phase':phase,'lock_digest':lock_digest,
                           'plate_count':len(rows),'weights':weights_pin,'environment':runtime_environment,
                           'configuration':config,'metrics':report,'comparison':comparison_result,
                           'final_test_completed':report['failures']==0,'test_outcomes_inspected':True,
                           'scientific_acceptance_established':False,'artifacts':{n:identity(p) for n,p in
                            [('decoded_cache',cache_path),('predictions',predictions_path),('comparison',comparison_path)]}},
                'cache_path':str(cache_path),'predictions_path':str(predictions_path),'comparison_path':str(comparison_path)}
    finally:
        for path in [str(root/'native-source'),str(root/'sahi-source')]:sys.path.remove(path)
