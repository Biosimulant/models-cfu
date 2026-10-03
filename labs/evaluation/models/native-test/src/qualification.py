"""Finite locked-test adapter; no final scientific acceptance is self-certified."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .process import absolute_inputs

SCHEMA = {'stage':'str','phase':'str','lock_digest':'str','plate_count':'int','weights':'json',
          'environment':'json','configuration':'json','metrics':'json','comparison':'json',
          'final_test_completed':'bool','test_outcomes_inspected':'bool',
          'scientific_acceptance_established':'bool','artifacts':'json'}


def checked_pin(sha,size,maximum):
    if (not isinstance(sha,str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
            or isinstance(size,bool) or not isinstance(size,int) or not 0 < size <= maximum):
        raise ValueError('Locked evaluation needs exact bounded retained artifact SHA/length')
    return {'sha256':sha,'size_bytes':size}


def bundle_identity(root):
    root = Path(root)
    def identity(path):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 128*1024:
            raise ValueError('Locked evaluation source/manifest must be bounded regular files')
        body = path.read_bytes()
        return {'sha256':hashlib.sha256(body).hexdigest(),'size_bytes':len(body)}
    return {'entrypoint':'src.qualification:QualifyNativeDetector',
            'model_manifest':identity(root/'model.yaml'),
            'authored_code':{str(p.relative_to(root)):identity(p) for p in sorted((root/'src').glob('*.py'))}}


def checked_request_inputs(values,phase):
    names={'source_archive','source_receipt','sahi_archive','sahi_receipt','frozen_split',
           'annotations','weights','test_lock',*{f'image_{i:02d}' for i in range(37)}}
    if phase != 'primary': names.add('baseline')
    if set(values) != names:
        raise ValueError('Locked evaluation input set differs from its phase contract')
    return absolute_inputs(values)


def launch(request,root,module_root,work_seconds):
    root = Path(root).resolve();req=root/'test-request.json';result=root/'test-result.json'
    req.write_text(json.dumps(request));req.chmod(0o600)
    log_path=root/'test-worker.log'
    with log_path.open('wb') as log:
        completed=subprocess.run([sys.executable,'-m','src.worker',str(req),str(result)],cwd=module_root,
                                 stdout=log,stderr=subprocess.STDOUT,timeout=work_seconds+180,check=False)
    if completed.returncode:
        with log_path.open('rb') as log:
            log.seek(max(0,log_path.stat().st_size-12000));tail=log.read().decode('utf-8',errors='replace')
        raise RuntimeError(f'Locked test worker exited {completed.returncode}: {tail}')
    if result.is_symlink() or not result.is_file() or not 0 < result.stat().st_size <= 1024*1024:
        raise RuntimeError('Locked test has no bounded regular terminal result')
    outputs=json.loads(result.read_bytes());receipt=outputs['receipt']
    if (receipt['stage'] != 'actual_locked_test_evaluation' or receipt['plate_count'] != 37
            or receipt['phase'] != request['phase'] or receipt['lock_digest'] != request['lock_digest']
            or receipt['test_outcomes_inspected'] is not True
            or receipt['scientific_acceptance_established'] is not False):
        raise RuntimeError('Locked test returned invalid scope or acceptance claims')
    for name in ['cache_path','predictions_path','comparison_path']:
        original=Path(outputs[name])
        if original.is_symlink():raise RuntimeError('Locked test artifact must be a regular owned output')
        p=original.resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file():raise RuntimeError('Locked test artifact escaped its task')
        outputs[name]=str(p)
    return outputs


class QualifyNativeDetector(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self,phase='primary',weights_sha256='',weights_size_bytes=0,
                 lock_sha256='',lock_size_bytes=0,lock_digest='',
                 baseline_sha256='',baseline_size_bytes=0,work_seconds=1200):
        if phase not in {'primary','replay','recreation'}:
            raise ValueError('Unknown locked evaluation phase')
        if isinstance(work_seconds,bool) or not isinstance(work_seconds,int) or not 60 <= work_seconds <= 1500:
            raise ValueError('Locked evaluation work time must be60..1500seconds')
        self.phase=phase;self.weights_sha256=weights_sha256;self.weights_size_bytes=weights_size_bytes
        self.lock_sha256=lock_sha256;self.lock_size_bytes=lock_size_bytes;self.lock_digest=lock_digest
        self.baseline_sha256=baseline_sha256;self.baseline_size_bytes=baseline_size_bytes
        self.work_seconds=work_seconds

    def inputs(self):
        names={'source_archive':'zip','source_receipt':'json','sahi_archive':'zip','sahi_receipt':'json',
               'frozen_split':'json','annotations':'json','weights':'pth','test_lock':'json','baseline':'json',
               **{f'image_{i:02d}':'jpg' for i in range(37)}}
        return {n:SignalSpec.scalar(dtype='str',value_type='file',format=fmt) for n,fmt in names.items()}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                **{n:SignalSpec.scalar(dtype='str',value_type='file',format='json') for n in
                   ['cache_path','predictions_path','comparison_path']}}

    def execute(self,inputs,*,context):
        weights=checked_pin(self.weights_sha256,self.weights_size_bytes,64*1024*1024)
        lock=checked_pin(self.lock_sha256,self.lock_size_bytes,8*1024*1024)
        checked_pin(self.lock_digest,1,1)  # Canonical lock digest is independently plan-bound.
        baseline=None
        if self.phase != 'primary':
            baseline=checked_pin(self.baseline_sha256,self.baseline_size_bytes,64*1024*1024)
        elif self.baseline_sha256 != '' or self.baseline_size_bytes != 0:
            raise ValueError('Primary test cannot consume prior test results')
        module_root=Path(__file__).resolve().parent.parent
        plan=json.loads((module_root/'validation-plan.json').read_bytes())
        values={n:(v.value if isinstance(v,BioSignal) else v) for n,v in inputs.items()}
        values=checked_request_inputs(values,self.phase)
        root=Path(tempfile.mkdtemp(prefix='cfu-locked-test-',dir=Path.cwd()));root.chmod(0o700)
        request={'inputs':values,'plan':plan,'root':str(root.resolve()),'weights_pin':weights,
                 'lock_pin':lock,'lock_digest':self.lock_digest,'evaluation_bundle':bundle_identity(module_root),
                 'phase':self.phase,'baseline_pin':baseline,'work_seconds':self.work_seconds}
        return launch(request,root,module_root,self.work_seconds)
