"""Finite recorded native training; parameters bind exact retained resume artifacts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .process import absolute_inputs

SCHEMA={'stage':'str','source':'json','contract_sha256':'str','training_pool':'json','environment':'json','recipe':'json',
        'start_progress':'json','end_progress':'json','steps_executed':'int','full_recipe_completed':'bool',
        'crops_consumed':'int','actual_optimizer_seconds':'float','end_to_end_crops_per_second':'float',
        'targets_truncated':'int','weights_changed':'bool','peak_cuda_allocated_bytes':'int','peak_cuda_reserved_bytes':'int',
        'last_losses':'json','saved_artifacts':'json','test_outcomes_inspected':'bool','scientific_acceptance_established':'bool'}


def input_path(inputs,name,required=True):
    value=inputs.get(name);value=value.value if isinstance(value,BioSignal) else value
    if not value and not required:return None
    if not isinstance(value,str) or not value:raise ValueError('Training requires retained File '+name)
    return value


def launch_training(request,root,module_root,work_seconds):
    root=Path(root).resolve();request_path=root/'training-request.json';result_path=root/'training-result.json'
    request_path.write_text(json.dumps(request));request_path.chmod(0o600)
    log_path=root/'training-worker.log'
    with log_path.open('wb') as log:
        result=subprocess.run([sys.executable,'-m','src.train_worker',str(request_path),str(result_path)],
                              cwd=module_root,stdout=log,stderr=subprocess.STDOUT,timeout=work_seconds+180,check=False)
    if result.returncode:
        with log_path.open('rb') as log:
            log.seek(max(0,log_path.stat().st_size-12000));tail=log.read().decode('utf-8',errors='replace')
        raise RuntimeError(f'Native training worker exited {result.returncode}: {tail}')
    if result_path.is_symlink() or not result_path.is_file() or not 0<result_path.stat().st_size<=1024*1024:
        raise RuntimeError('Native training has no bounded regular result')
    outputs=json.loads(result_path.read_bytes());receipt=outputs['receipt']
    if (receipt['stage']!='native_frozen_training_chunk' or receipt['steps_executed']<1
            or receipt['scientific_acceptance_established'] is not False or receipt['test_outcomes_inspected'] is not False):
        raise RuntimeError('Native training returned incomplete or invalid evidence')
    for name in ['checkpoint_path','weights_path','history_path']:
        path=Path(outputs[name]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():raise RuntimeError('Native training artifact escapes its own task')
        outputs[name]=str(path)
    return outputs


class TrainNativeDetector(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self,work_seconds=300,checkpoint_sha256='',checkpoint_size_bytes=0,weights_sha256='',weights_size_bytes=0):
        if isinstance(work_seconds,bool) or not isinstance(work_seconds,int) or not 60<=work_seconds<=1500:
            raise ValueError('Training work time must be60..1500seconds')
        self.work_seconds=work_seconds
        self.checkpoint_pin=self.pin(checkpoint_sha256,checkpoint_size_bytes)
        self.weights_pin=self.pin(weights_sha256,weights_size_bytes)
        if bool(self.checkpoint_pin)!=bool(self.weights_pin):raise ValueError('Resume requires both checkpoint and bound EMA weights')

    @staticmethod
    def pin(sha,size):
        if sha=='' and size==0:return None
        if (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)
                or isinstance(size,bool) or not isinstance(size,int) or not 0<size<=64*1024*1024):
            raise ValueError('Training resume requires an exact bounded SHA/length pin')
        return {'sha256':sha,'size_bytes':size}

    def inputs(self):
        names={'source_archive':'zip','source_receipt':'json','frozen_split':'json','prepared_verification':'json',
               **{f'archive_{i:02d}':'zip' for i in range(22)}}
        specs={n:SignalSpec.scalar(dtype='str',value_type='file',format=kind) for n,kind in names.items()}
        specs.update({n:SignalSpec.scalar(dtype='str',value_type='file',format='pth',required=False) for n in ['checkpoint','weights']})
        return specs

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                **{n:SignalSpec.scalar(dtype='str',value_type='file',format=fmt)
                   for n,fmt in [('checkpoint_path','pth'),('weights_path','pth'),('history_path','json')]}}

    def execute(self,inputs,*,context):
        plan=json.loads((Path(__file__).resolve().parent.parent/'training-plan.json').read_text())
        values={n:input_path(inputs,n,required=n not in {'checkpoint','weights'}) for n in self.inputs()}
        optional={n:v for n,v in values.items() if n in {'checkpoint','weights'} and v}
        if bool(optional)!=bool(self.checkpoint_pin) or (optional and set(optional)!={'checkpoint','weights'}):
            raise ValueError('Resume File inputs and declared pins must match exactly')
        values=absolute_inputs({n:v for n,v in values.items() if v})
        root=Path(tempfile.mkdtemp(prefix='cfu-training-',dir=Path.cwd()));root.chmod(0o700)
        request={'inputs':values,'plan':plan,'work_seconds':self.work_seconds,
                 'checkpoint_pin':self.checkpoint_pin,'weights_pin':self.weights_pin}
        return launch_training(request,root,Path(__file__).resolve().parent.parent,self.work_seconds)
