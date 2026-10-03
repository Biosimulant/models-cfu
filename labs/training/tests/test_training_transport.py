"""Synthetic execution/pin contract checks; no optimizer, predictions or qualification."""

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SRC=Path(__file__).resolve().parents[1]/'models/native/src'
spec=importlib.util.spec_from_file_location('cfu_training_transport_fixture',SRC/'__init__.py',submodule_search_locations=[str(SRC)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
training=importlib.import_module('cfu_training_transport_fixture.training')


@pytest.mark.parametrize('kwargs',[{'work_seconds':True},{'work_seconds':59},{'work_seconds':1501},
                                  {'checkpoint_sha256':'a'*64,'checkpoint_size_bytes':100},
                                  {'checkpoint_sha256':'g'*64,'checkpoint_size_bytes':100},
                                  {'checkpoint_sha256':'a'*64,'checkpoint_size_bytes':64*1024*1024+1},
                                  {'weights_sha256':'a'*64,'weights_size_bytes':100}])
def test_unbounded_or_partial_resume_contract_cannot_launch(kwargs):
    with pytest.raises(ValueError):training.TrainNativeDetector(**kwargs)


def test_a_failed_child_cannot_publish_a_leftover_training_result(tmp_path,monkeypatch):
    (tmp_path/'training-result.json').write_text(json.dumps({'synthetic':'stale'}))

    def fail(args,**kwargs):
        kwargs['stdout'].write(b'synthetic training failure')
        return subprocess.CompletedProcess(args,7)

    monkeypatch.setattr(training.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='exited 7: synthetic training failure'):
        training.launch_training({'synthetic':True},tmp_path,tmp_path,300)


def test_declared_mixed_numeric_record_is_transported_by_pinned_sdk():
    from biosimulant import BioModule,BioWorld,ExecutionPolicy

    fields={'stage':'synthetic_transport_only','source':{},'contract_sha256':'synthetic','training_pool':{},'environment':{},'recipe':{},
            'start_progress':{},'end_progress':{},'steps_executed':1,'full_recipe_completed':False,'crops_consumed':0,
            'actual_optimizer_seconds':0.,'end_to_end_crops_per_second':0.,'targets_truncated':0,'weights_changed':False,
            'peak_cuda_allocated_bytes':0,'peak_cuda_reserved_bytes':0,'last_losses':[],'saved_artifacts':{},
            'test_outcomes_inspected':False,'scientific_acceptance_established':False}

    class Source(BioModule):
        execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

        def inputs(self):return {}

        def outputs(self):return {'receipt':training.TrainNativeDetector().outputs()['receipt']}

        def execute(self,inputs,*,context):return {'receipt':fields}

    world=BioWorld(communication_step=.01);world.add_biomodule('synthetic',Source());world.run(duration=.01)
    assert world.get_outputs('synthetic')['receipt'].value==fields
