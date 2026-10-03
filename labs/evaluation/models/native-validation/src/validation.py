"""Finite validation adapter; exact learned-weight pins are required per run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .process import absolute_inputs

SCHEMA = {'stage':'str','plate_count':'int','weights':'json','environment':'json',
          'selected_configuration':'json','selected_validation_metrics':'json',
          'validation_selection_complete':'bool','test_outcomes_inspected':'bool',
          'scientific_acceptance_established':'bool','artifacts':'json'}


def checked_weight_pin(sha, size):
    if (not isinstance(sha,str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
            or isinstance(size,bool) or not isinstance(size,int) or not 0 < size <= 64*1024*1024):
        raise ValueError('Validation needs the exact retained learned-weight SHA/length')
    return {'sha256':sha,'size_bytes':size}


def launch(request, root, module_root, work_seconds):
    root = Path(root).resolve();req=root/'validation-request.json';result=root/'validation-result.json'
    req.write_text(json.dumps(request));req.chmod(0o600)
    log_path=root/'validation-worker.log'
    with log_path.open('wb') as log:
        completed=subprocess.run([sys.executable,'-m','src.worker',str(req),str(result)],cwd=module_root,
                                 stdout=log,stderr=subprocess.STDOUT,timeout=work_seconds+180,check=False)
    if completed.returncode:
        with log_path.open('rb') as log:
            log.seek(max(0,log_path.stat().st_size-12000));tail=log.read().decode('utf-8',errors='replace')
        raise RuntimeError(f'Native validation worker exited {completed.returncode}: {tail}')
    if result.is_symlink() or not result.is_file() or not 0 < result.stat().st_size <= 1024*1024:
        raise RuntimeError('Validation has no bounded regular terminal result')
    outputs=json.loads(result.read_bytes());receipt=outputs['receipt']
    if (receipt['stage'] != 'actual_native_validation' or receipt['plate_count'] != 37
            or receipt['test_outcomes_inspected'] is not False
            or receipt['scientific_acceptance_established'] is not False):
        raise RuntimeError('Validation returned invalid scope or acceptance claims')
    for name in ['cache_path','predictions_path','comparison_path']:
        original=Path(outputs[name])
        if original.is_symlink():raise RuntimeError('Validation artifact must be a regular owned output')
        p=original.resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file():raise RuntimeError('Validation artifact escaped its task')
        outputs[name]=str(p)
    return outputs


class ValidateNativeDetector(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self,weights_sha256='',weights_size_bytes=0,work_seconds=1200):
        if isinstance(work_seconds,bool) or not isinstance(work_seconds,int) or not 60 <= work_seconds <= 1500:
            raise ValueError('Validation work time must be60..1500seconds')
        # Empty defaults allow workspace authoring; execution never accepts them.
        self.weights_sha256=weights_sha256;self.weights_size_bytes=weights_size_bytes;self.work_seconds=work_seconds

    def inputs(self):
        names={'source_archive':'zip','source_receipt':'json','sahi_archive':'zip','sahi_receipt':'json',
               'frozen_split':'json','annotations':'json','weights':'pth',**{f'image_{i:02d}':'jpg' for i in range(37)}}
        return {n:SignalSpec.scalar(dtype='str',value_type='file',format=fmt) for n,fmt in names.items()}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                **{n:SignalSpec.scalar(dtype='str',value_type='file',format=fmt) for n,fmt in
                   [('cache_path','json'),('predictions_path','zip'),('comparison_path','json')]}}

    def execute(self,inputs,*,context):
        pin=checked_weight_pin(self.weights_sha256,self.weights_size_bytes)
        plan=json.loads((Path(__file__).resolve().parent.parent/'validation-plan.json').read_text())
        values={n:(v.value if isinstance(v,BioSignal) else v) for n,v in inputs.items()}
        if set(values) != set(self.inputs()):raise ValueError('Validation input set differs from its exact contract')
        values=absolute_inputs(values)
        root=Path(tempfile.mkdtemp(prefix='cfu-validation-',dir=Path.cwd()));root.chmod(0o700)
        request={'inputs':values,'plan':plan,'root':str(root.resolve()),'weights_pin':pin,'work_seconds':self.work_seconds}
        return launch(request,root,Path(__file__).resolve().parent.parent,self.work_seconds)
