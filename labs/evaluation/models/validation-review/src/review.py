"""Managed visual review of actual validation predictions; no new inference or test access."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

SCHEMA={'stage':'str','plate_count':'int','configuration':'json','weights':'json',
        'validation_origin':'json','validation_metrics':'json','environment':'json',
        'test_outcomes_inspected':'bool','scientific_acceptance_established':'bool','artifact':'json'}
FORMATS={'frozen_split':'json','comparison':'json','predictions':'zip',
         **{f'image_{i:02d}':'jpg' for i in range(37)}}


class RenderValidationReview(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self,validation_origin_json=''):
        self.origin_json=validation_origin_json

    def inputs(self):
        return {n:SignalSpec.scalar(dtype='str',value_type='file',format=f) for n,f in FORMATS.items()}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                'review_path':SignalSpec.scalar(dtype='str',value_type='file',format='zip')}

    def execute(self,inputs,*,context):
        if set(inputs)!=set(FORMATS):raise ValueError('Review accepts only original validation photos and retained validation evidence')
        if not isinstance(self.origin_json,str) or not 0<len(self.origin_json)<=16384:
            raise ValueError('Review requires bounded account-verified validation origin metadata')
        values={n:str(Path(v.value if isinstance(v,BioSignal) else v).resolve(strict=True)) for n,v in inputs.items()}
        module_root=Path(__file__).resolve().parent.parent
        root=Path(tempfile.mkdtemp(prefix='cfu-validation-review-',dir=Path.cwd()));root.chmod(0o700)
        request=root/'request.json';result=root/'result.json';log=root/'worker.log'
        request.write_text(json.dumps({'inputs':values,'root':str(root.resolve()),
                                       'validation_origin':json.loads(self.origin_json)}));request.chmod(0o600)
        with log.open('wb') as stream:
            done=subprocess.run([sys.executable,'-m','src.worker',str(request),str(result)],
                                cwd=module_root,stdout=stream,stderr=subprocess.STDOUT,timeout=240,check=False)
        if done.returncode:
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
            raise RuntimeError(f'Validation visual review failed: {tail}')
        if result.is_symlink() or not result.is_file() or not 0<result.stat().st_size<=1024*1024:
            raise RuntimeError('Review requires a bounded regular result')
        outputs=json.loads(result.read_bytes());receipt=outputs['receipt'];path=Path(outputs['review_path'])
        if (receipt['stage']!='managed_actual_validation_visual_review' or receipt['plate_count']!=37
                or receipt['test_outcomes_inspected'] is not False
                or receipt['scientific_acceptance_established'] is not False
                or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(root.resolve())
                or not 0<path.stat().st_size<=64*1024*1024):
            raise RuntimeError('Visual review changes its scope or bounded artifact contract')
        return outputs
