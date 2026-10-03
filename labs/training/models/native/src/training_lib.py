"""Native training on every frozen training crop, with retained resumable state."""

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

from .data import TrainingArchive, loss_targets
from .pool import index_training_crops, next_batch
from .source import extract_native_source, pinned_bytes
from .training_state import augmentation_seed, mirrored_rows, recipe_digest, validate_progress

MAX_CHECKPOINT_BYTES=64*1024*1024


def file_identity(path):
    path=Path(path)
    return {'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_verification(inputs, plan):
    raw=pinned_bytes(inputs['prepared_verification'],plan['prepared_verification'],1024*1024)
    receipt=json.loads(raw)['outputs']['aggregate']['receipt']['value']
    summary=receipt['summary']
    if (receipt['stage']!='whole_set_prepared_crop_identity_verification'
            or summary['source_plates']!=332 or summary['derived_tiles']!=10943
            or summary['cross_split_identical_crops']!=0 or summary['original_plates_excluded']!=0
            or summary['test_outcomes_inspected'] is not False
            or receipt['verified_inputs']['frozen_split']!=plan['frozen_split']):
        raise ValueError('Training requires the exact actual whole-set verification receipt')
    return receipt


def tensors(rows, archives, torch, numpy, augment_hsv, *, seed, step, augment):
    images=numpy.full((len(rows),3,640,640),114,dtype=numpy.float32)
    labels=[loss_targets(t) for _,t in rows]
    capacity=max(1,max(map(len,labels)))
    targets=numpy.zeros((len(rows),capacity,5),dtype=numpy.float32)
    for i,(archive_index,tile) in enumerate(rows):
        with archives[archive_index].image(tile) as image:
            pixels=numpy.asarray(image)[:,:,::-1].copy()
        if augment:
            aug_seed=augmentation_seed(seed,step,tile)
            numpy.random.seed(aug_seed)
            augment_hsv(pixels)
            if random.Random(aug_seed).random()<.5:
                pixels=pixels[:,::-1,:].copy()
                labels[i]=mirrored_rows(labels[i],tile['width'])
        images[i,:,:tile['height'],:tile['width']]=pixels.transpose(2,0,1)
        if labels[i]:targets[i,:len(labels[i])]=labels[i]
    return torch.from_numpy(images).cuda(),torch.from_numpy(targets).cuda(),[len(r) for r in labels]


def run(inputs, plan, root, *, work_seconds, checkpoint_pin, weights_pin):
    started=time.monotonic();root=Path(root)
    for name,pin in plan['authored_code'].items():
        pinned_bytes(Path(__file__).parent/name,pin,128*1024)
    checked_verification(inputs,plan)
    source=extract_native_source(inputs['source_archive'],inputs['source_receipt'],root/'native',
                                archive_pin=plan['source_archive'],manifest_pin=plan['source_receipt'])
    frozen=json.loads(pinned_bytes(inputs['frozen_split'],plan['frozen_split'],1024*1024))
    archives=[]
    try:
        for i,pin in enumerate(plan['archives']):
            archive_plan={'archive':pin,'manifest':pin['manifest'],'preparation_batch':i,
                          'frozen_split':plan['frozen_split'],'frozen_dataset_id':plan['frozen_dataset_id'],
                          'frozen_dataset_sha256':plan['frozen_dataset_sha256']}
            archives.append(TrainingArchive(inputs[f'archive_{i:02d}'],inputs['frozen_split'],archive_plan,require_training=False))
        pool=index_training_crops(archives,frozen['native_dataset_manifest']['members'])
        pool_sha=recipe_digest([{'parent_file_id':t['parent_file_id'],'file_name':t['file_name'],'sha256':t['sha256']} for _,t in pool])
        sys.path.insert(0,str(root/'native'))
        try:
            if any(n=='yolox' or n.startswith('yolox.') for n in sys.modules):
                raise RuntimeError('Training would reuse an unverified cached native module')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            import numpy
            import torch
            import torchvision
            from yolox.exp import Exp
            from yolox.utils import ModelEMA

            recipe=plan['recipe'];seed=recipe['seed'];batch=recipe['batch_size'];max_epochs=recipe['max_epochs']
            environment={'python':platform.python_version(),'numpy':str(numpy.__version__),'torch':str(torch.__version__),
                         'torchvision':str(torchvision.__version__),'cuda_build':str(torch.version.cuda),'cudnn':torch.backends.cudnn.version()}
            if environment!=recipe['required_environment'] or importlib.metadata.version('numpy')!=numpy.__version__:
                raise RuntimeError('Native training differs from its actually measured pinned environment')
            if not torch.cuda.is_available():raise RuntimeError('Native training requires CUDA; CPU execution is forbidden')
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
            random.seed(seed);numpy.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            exp=Exp();exp.depth=.33;exp.width=.375;exp.num_classes=1
            exp.input_size=(640,640);exp.test_size=(640,640)
            exp.max_epoch=max_epochs;exp.warmup_epochs=1;exp.no_aug_epochs=5
            model=exp.get_model().cuda();optimizer=exp.get_optimizer(batch)
            scheduler=exp.get_lr_scheduler(exp.basic_lr_per_img*batch,math.ceil(len(pool)/batch))
            amp=recipe['precision']=='amp_float16'
            scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128,growth_interval=1000000)
            ema=ModelEMA(model,decay=.9998)
            progress={'epoch':0,'cursor':0,'global_step':0}
            contract_sha=recipe_digest(plan)
            if checkpoint_pin:
                state=torch.load(io.BytesIO(pinned_bytes(inputs['checkpoint'],checkpoint_pin,MAX_CHECKPOINT_BYTES)),map_location='cpu',weights_only=True)
                saved=torch.load(io.BytesIO(pinned_bytes(inputs['weights'],weights_pin,MAX_CHECKPOINT_BYTES)),map_location='cpu',weights_only=True)
                if (state['format']!='cfu-native-training-checkpoint-v1' or state['contract_sha256']!=contract_sha
                        or state['training_pool_sha256']!=pool_sha or state['environment']!=environment
                        or state['weights_identity']!=weights_pin or saved['contract_sha256']!=contract_sha
                        or saved['format']!='cfu-native-ema-weights-v1'
                        or saved['progress']!=state['progress'] or saved['training_pool_sha256']!=pool_sha):
                    raise ValueError('Native checkpoint changes its frozen training trajectory or EMA binding')
                progress=state['progress'];validate_progress(progress,len(pool),batch,max_epochs)
                model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
                scaler.load_state_dict(state['scaler']);ema.ema.load_state_dict(saved['model'],strict=True)
                ema.updates=state['ema_updates']
                if ema.updates!=progress['global_step']:raise ValueError('EMA updates differ from actual optimizer progress')
                torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            start_progress=dict(progress);history=[];initial=next(model.parameters()).detach().clone()
            model.train();torch.cuda.reset_peak_memory_stats()
            spec=importlib.util.spec_from_file_location('_cfu_verified_native_augmentation',root/'native/yolox/data/data_augment.py')
            augmentation=importlib.util.module_from_spec(spec);spec.loader.exec_module(augmentation)
            while progress['epoch']<max_epochs and time.monotonic()-started<work_seconds:
                before=dict(progress)
                selected,(next_epoch,next_cursor)=next_batch(len(pool),seed,progress['epoch'],progress['cursor'],batch)
                rows=[pool[i] for i in selected]
                last_phase=progress['epoch']>=max_epochs-exp.no_aug_epochs
                model.head.use_l1=last_phase
                torch.cuda.synchronize();tick=time.monotonic()
                images,targets,counts=tensors(rows,archives,torch,numpy,augmentation.augment_hsv,
                                             seed=seed,step=progress['global_step'],augment=not last_phase)
                lr=scheduler.update_lr(progress['global_step']+1)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=amp,dtype=torch.float16):
                    output=model(images,targets);loss=output['total_loss']
                if not torch.isfinite(loss).item():raise RuntimeError('Native training loss is nonfinite')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise RuntimeError('Native training gradient is nonfinite; no crop is silently skipped')
                scaler.step(optimizer);scaler.update();ema.update(model)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                progress={'epoch':next_epoch,'cursor':next_cursor,'global_step':progress['global_step']+1}
                history.append({'before':before,'after':dict(progress),'seconds':elapsed,'loss':float(loss.item()),'lr':float(lr),
                                'target_counts':counts,'crop_sha256':[t['sha256'] for _,t in rows],
                                'parent_file_ids':[t['parent_file_id'] for _,t in rows],'l1_enabled':last_phase})
                del images,targets,output,loss
            validate_progress(progress,len(pool),batch,max_epochs)
            if not history or torch.equal(initial,next(model.parameters()).detach()):
                raise RuntimeError('Native training task made no real optimizer progress')
            if any(not torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()):
                raise RuntimeError('Native training would save nonfinite weights')
            weights_path=root/'ema-weights.pth';checkpoint_path=root/'training-checkpoint.pth';history_path=root/'training-history.json'
            torch.save({'format':'cfu-native-ema-weights-v1','model':ema.ema.state_dict(),'contract_sha256':contract_sha,
                        'training_pool_sha256':pool_sha,'progress':progress,'environment':environment,'recipe':recipe},weights_path)
            weights_identity=file_identity(weights_path)
            torch.save({'format':'cfu-native-training-checkpoint-v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),'ema_updates':ema.updates,'contract_sha256':contract_sha,
                        'training_pool_sha256':pool_sha,'progress':progress,'environment':environment,'weights_identity':weights_identity,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint_path)
            history_path.write_text(json.dumps({'stage':'actual_native_training_steps','start':start_progress,'end':progress,
                                               'steps':history,'targets_truncated':0},allow_nan=False))
            if any(p.stat().st_size>MAX_CHECKPOINT_BYTES for p in [weights_path,checkpoint_path,history_path]):
                raise RuntimeError('Native training artifact exceeds the64MiB retained-file bound')
            for p in [weights_path,checkpoint_path,history_path]:p.chmod(0o600)
            crops=sum(len(h['crop_sha256']) for h in history);seconds=sum(h['seconds'] for h in history)
            receipt={'stage':'native_frozen_training_chunk','source':source,'contract_sha256':contract_sha,
                     'training_pool':{'source_plates':len({t['sample_id'] for _,t in pool}),'crops':len(pool),'sha256':pool_sha,'frozen_dataset_id':plan['frozen_dataset_id'],
                                      'all_frozen_training_parents_covered':True,'optimizer_validation_or_test_crops':0},
                     'environment':{**environment,'device':torch.cuda.get_device_name(0),'precision':recipe['precision'],
                                    'deterministic_algorithms':True,'execution_isolation':'fresh managed Python subprocess'},
                     'recipe':recipe,'start_progress':start_progress,'end_progress':progress,'steps_executed':len(history),
                     'full_recipe_completed':progress['epoch']==max_epochs,'crops_consumed':crops,'actual_optimizer_seconds':seconds,
                     'end_to_end_crops_per_second':crops/seconds,'targets_truncated':0,'weights_changed':True,
                     'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'last_losses':[h['loss'] for h in history[-10:]],
                     'saved_artifacts':{'checkpoint':file_identity(checkpoint_path),'weights':weights_identity,'history':file_identity(history_path)},
                     'test_outcomes_inspected':False,'scientific_acceptance_established':False}
            return {'receipt':receipt,'checkpoint_path':str(checkpoint_path),'weights_path':str(weights_path),'history_path':str(history_path)}
        finally:
            sys.path.remove(str(root/'native'))
    finally:
        for archive in archives:archive.close()
