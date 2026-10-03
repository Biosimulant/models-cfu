"""Synthetic six-file transport/integrity fixtures, never CFU accuracy evidence."""

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'models/aggregate/src'
spec = importlib.util.spec_from_file_location('cfu_whole_set_fixture', SRC/'__init__.py', submodule_search_locations=[str(SRC)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
aggregate = importlib.import_module('cfu_whole_set_fixture.aggregate')


def fixture(tmp_path, leak=False):
    members, inputs, pins = [], {}, []

    def retain(name, value):
        path = tmp_path/(name+'.json')
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        inputs[name] = str(path)
        return {'file_id':'synthetic-'+name,'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

    for i in range(6):
        split = 'train' if i < 5 else 'validation'
        member = {'sample_id':f'plate-{i}','file_id':f'parent-{i}','group_id':f'group-{i}','split':split,'role':'image'}
        members.append(member)
        row = {'sample_id':member['sample_id'],'parent_file_id':member['file_id'],'group_id':member['group_id'],'split':split,
               'file_name':f'crop-{i}.jpg','sha256':('a' if leak else str(i))*64,'size_bytes':100,'preparation_batch':i}
        pins.append(retain(f'index_{i:02d}', [row]))
    members.append({'sample_id':'held','file_id':'held','group_id':'held','split':'test','role':'image'})
    frozen = retain('frozen_split', {'native_dataset_manifest':{'members':members}})
    return inputs, {'frozen_split':frozen,'indices':pins,'expected_tiles_by_batch':{str(i):1 for i in range(6)}}


def test_finite_record_transport_computes_all_six_files(tmp_path, monkeypatch):
    from biosimulant import BioModule, BioWorld, ExecutionPolicy

    values, plan = fixture(tmp_path)
    model = tmp_path/'model'
    (model/'src').mkdir(parents=True)
    (model/'aggregate-plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(aggregate, '__file__', str(model/'src/aggregate.py'))
    component = aggregate.VerifyWholeSet()

    class Source(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

        def inputs(self):
            return {}

        def outputs(self):
            return component.inputs()

        def execute(self, inputs, *, context):
            return values

    world = BioWorld(communication_step=.01)
    world.add_biomodule('source', Source())
    world.add_biomodule('aggregate', component)
    for name in values:
        world.connect('source.'+name, 'aggregate.'+name)
    world.run(duration=.01)
    receipt = world.get_outputs('aggregate')['receipt'].value
    assert receipt['summary']['source_plates'] == 6
    assert receipt['summary']['derived_tiles'] == 6
    assert receipt['summary']['unique_crop_byte_digests'] == 6
    assert receipt['summary']['cross_split_identical_crops'] == 0
    assert receipt['scientific_acceptance_established'] is False


@pytest.mark.parametrize('tamper', ['missing','truncated','digest','index_count','cross_split_hash'])
def test_missing_changed_incomplete_or_leaked_files_fail_before_receipt(tmp_path, tamper):
    inputs, plan = fixture(tmp_path, leak=tamper == 'cross_split_hash')
    if tamper == 'missing':
        inputs.pop('index_05')
    elif tamper == 'truncated':
        Path(inputs['index_05']).write_bytes(b'[]')
    elif tamper == 'digest':
        plan['indices'][0]['sha256'] = 'f'*64
    elif tamper == 'index_count':
        plan['indices'].pop()
    with pytest.raises(ValueError):
        aggregate.combine(inputs, plan)
