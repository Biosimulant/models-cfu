"""Synthetic resume/augmentation fixtures, never checkpoint replay or accuracy."""

import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('cfu_state_fixture',Path(__file__).resolve().parents[1]/'models/native/src/training_state.py')
state=importlib.util.module_from_spec(spec);spec.loader.exec_module(state)


@pytest.mark.parametrize('progress',[{'epoch':0,'cursor':0,'global_step':0},{'epoch':0,'cursor':12,'global_step':3},
                                    {'epoch':1,'cursor':0,'global_step':4},{'epoch':2,'cursor':8,'global_step':10},
                                    {'epoch':40,'cursor':0,'global_step':160}])
def test_partial_and_complete_epoch_progress_accounts_for_short_final_batch(progress):
    assert state.validate_progress(progress,13,4,40)==tuple(progress.values())


@pytest.mark.parametrize('progress',[{'epoch':0,'cursor':1,'global_step':0},{'epoch':1,'cursor':0,'global_step':3},
                                    {'epoch':40,'cursor':4,'global_step':161},{'epoch':True,'cursor':0,'global_step':4},
                                    {'epoch':0,'cursor':13,'global_step':4}])
def test_changed_cursor_or_optimizer_count_fails_before_resume(progress):
    with pytest.raises(ValueError):state.validate_progress(progress,13,4,40)


def test_horizontal_flip_preserves_dense_target_capacity_and_is_an_involution():
    rows=[[0.,2.,5.,4.,2.]]*128+[[0.,639.5,10.,1.,3.]]
    mirrored=state.mirrored_rows(rows,640)
    assert len(mirrored)==129
    assert mirrored[0][1]==638. and mirrored[-1][1]==.5
    assert state.mirrored_rows(mirrored,640)==rows


def test_augmentation_seed_is_independent_of_worker_chunks_and_changes_by_step():
    tile={'parent_file_id':'synthetic-parent','file_name':'synthetic-crop.jpg'}
    assert state.augmentation_seed(20261001,10,tile)==state.augmentation_seed(20261001,10,dict(tile))
    assert state.augmentation_seed(20261001,10,tile)!=state.augmentation_seed(20261001,11,tile)
