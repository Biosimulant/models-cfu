"""Managed CPU software compatibility check, with no CFU data or learned weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

SCHEMA = {'stage':'str', 'checks':'json', 'environment':'json', 'sources':'json',
          'dependency_license':'json', 'scientific_acceptance_established':'bool',
          'test_outcomes_inspected':'bool', 'synthetic_software_fixtures_only':'bool'}
FORMATS = {'source_archive':'zip', 'source_receipt':'json',
           'sahi_archive':'zip', 'sahi_receipt':'json'}


class CheckInferenceImports(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):
        return {n:SignalSpec.scalar(dtype='str',value_type='file',format=f) for n,f in FORMATS.items()}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                'receipt_path':SignalSpec.scalar(dtype='str',value_type='file',format='json')}

    def execute(self,inputs,*,context):
        if set(inputs) != set(FORMATS):raise ValueError('Import check accepts only four pinned source inputs')
        values={n:str(Path(v.value if isinstance(v,BioSignal) else v).resolve(strict=True))
                for n,v in inputs.items()}
        module_root=Path(__file__).resolve().parent.parent
        root=Path(tempfile.mkdtemp(prefix='cfu-import-smoke-',dir=Path.cwd()));root.chmod(0o700)
        request=root/'request.json';result=root/'receipt.json';log=root/'worker.log'
        request.write_text(json.dumps({'inputs':values,'root':str(root.resolve())}));request.chmod(0o600)
        with log.open('wb') as stream:
            completed=subprocess.run([sys.executable,'-m','src.worker',str(request),str(result)],
                                     cwd=module_root,stdout=stream,stderr=subprocess.STDOUT,
                                     timeout=120,check=False)
        if completed.returncode:
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
            raise RuntimeError(f'CPU inference import check failed: {tail}')
        if result.is_symlink() or not result.is_file() or not 0 < result.stat().st_size <= 1024*1024:
            raise RuntimeError('Import check requires a bounded regular receipt')
        receipt=json.loads(result.read_bytes())
        if (receipt.get('stage') != 'managed_cpu_inference_import_software_check'
                or receipt.get('scientific_acceptance_established') is not False
                or receipt.get('test_outcomes_inspected') is not False
                or receipt.get('synthetic_software_fixtures_only') is not True):
            raise RuntimeError('Import check receipt changes its software-only scope')
        return {'receipt':receipt,'receipt_path':str(result.resolve())}
