"""Finite managed pre-test lock; passing validation evidence is mandatory."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .lock import checked_pin

SCHEMA={'stage':'str','lock_digest':'str','configuration':'json','weights':'json',
        'validation_origin':'json','validation_metrics':'json','test_plate_count':'int',
        'test_outcomes_inspected':'bool','scientific_acceptance_established':'bool',
        'environment':'json','artifact':'json'}
FORMATS={'frozen_split':'json','comparison':'json','predictions':'zip','weights':'pth'}


def checked_origin(origin,weights):
    from uuid import UUID
    UUID(origin['run_id']);UUID(origin['revision_id'])
    for name in ('comparison','predictions'):
        checked_pin(origin[name],file_required=True)
    checked_pin(weights,file_required=True)
    return {'validation_origin':origin,'weights_file':weights}


class FreezeFinalTest(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self,validation_origin_json='',weights_file_json=''):
        self.origin_json=validation_origin_json;self.weights_json=weights_file_json

    def inputs(self):
        return {n:SignalSpec.scalar(dtype='str',value_type='file',format=f) for n,f in FORMATS.items()}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                'lock_path':SignalSpec.scalar(dtype='str',value_type='file',format='json')}

    def execute(self,inputs,*,context):
        if set(inputs)!=set(FORMATS):raise ValueError('Lock accepts only frozen metadata, validation artifacts and weights')
        if (not isinstance(self.origin_json,str) or not isinstance(self.weights_json,str)
                or not 0<len(self.origin_json)<=16384 or not 0<len(self.weights_json)<=16384):
            raise ValueError('Lock requires bounded account-verified Run/File metadata parameters')
        metadata=checked_origin(json.loads(self.origin_json),json.loads(self.weights_json))
        values={n:str(Path(v.value if isinstance(v,BioSignal) else v).resolve(strict=True)) for n,v in inputs.items()}
        module_root=Path(__file__).resolve().parent.parent
        root=Path(tempfile.mkdtemp(prefix='cfu-final-test-lock-',dir=Path.cwd()));root.chmod(0o700)
        request=root/'request.json';result=root/'result.json';log=root/'worker.log'
        request.write_text(json.dumps({'inputs':values,'root':str(root.resolve()),**metadata}));request.chmod(0o600)
        with log.open('wb') as stream:
            completed=subprocess.run([sys.executable,'-m','src.worker',str(request),str(result)],
                                     cwd=module_root,stdout=stream,stderr=subprocess.STDOUT,timeout=240,check=False)
        if completed.returncode:
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
            raise RuntimeError(f'Managed pre-test lock failed: {tail}')
        if result.is_symlink() or not result.is_file() or not 0<result.stat().st_size<=1024*1024:
            raise RuntimeError('Lock requires a bounded regular terminal result')
        outputs=json.loads(result.read_bytes());receipt=outputs['receipt'];path=Path(outputs['lock_path'])
        if (receipt['stage']!='managed_validation_selected_pre_test_lock'
                or receipt['test_plate_count']!=37 or receipt['test_outcomes_inspected'] is not False
                or receipt['scientific_acceptance_established'] is not False
                or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve())
                or not path.is_file() or not 0<path.stat().st_size<=8*1024*1024):
            raise RuntimeError('Lock output changes its bounded pre-test scope')
        return outputs
