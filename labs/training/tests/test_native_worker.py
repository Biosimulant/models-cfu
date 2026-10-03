"""Synthetic subprocess failure/receipt checks; no training or timing assertions."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SRC = Path(__file__).resolve().parents[1]/'models/native/src'
spec = importlib.util.spec_from_file_location('cfu_process_fixture', SRC/'process.py')
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)


def test_child_failure_cannot_return_a_leftover_plausible_receipt(tmp_path, monkeypatch):
    (tmp_path/'worker-receipt.json').write_text(json.dumps({'stage':'synthetic-stale'}))

    def fail(args, **kwargs):
        kwargs['stdout'].write(b'synthetic child failure')
        return subprocess.CompletedProcess(args, 7)

    monkeypatch.setattr(process.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='exited 7: synthetic child failure'):
        process.launch({'synthetic':True}, tmp_path, tmp_path)


def test_child_timeout_does_not_emit_a_receipt(tmp_path, monkeypatch):
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])

    monkeypatch.setattr(process.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        process.launch({'synthetic':True}, tmp_path, tmp_path)


@pytest.mark.parametrize('tamper', ['missing','oversize','symlink','no_measured_case','changes_claim'])
def test_unbounded_missing_or_invalid_child_receipt_is_rejected(tmp_path, tamper):
    path=tmp_path/'receipt.json'
    value={'stage':'native_gpu_training_compatibility_benchmark','cases':[],
           'scientific_acceptance_established':False,'test_outcomes_inspected':False}
    if tamper == 'changes_claim':
        value['scientific_acceptance_established']=True
    if tamper == 'oversize':
        path.write_bytes(b'x'*(process.MAX_RECEIPT_BYTES+1))
    elif tamper == 'symlink':
        target=tmp_path/'other.json';target.write_text(json.dumps(value));path.symlink_to(target)
    elif tamper != 'missing':
        path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        process.read_receipt(path)


def test_file_paths_keep_their_identity_after_child_working_directory_changes(tmp_path, monkeypatch):
    parent=tmp_path/'parent';child=tmp_path/'child'
    parent.mkdir();child.mkdir()
    source=parent/'source.zip';source.write_bytes(b'synthetic retained bytes')
    monkeypatch.chdir(parent)
    values=process.absolute_inputs({'source_archive':'source.zip'})
    monkeypatch.chdir(child)
    assert Path(values['source_archive']).read_bytes()==b'synthetic retained bytes'
    with pytest.raises(FileNotFoundError):
        process.absolute_inputs({'source_archive':'missing.zip'})
