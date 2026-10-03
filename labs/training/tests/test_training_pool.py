"""Synthetic coverage/resume fixtures, never training or scientific acceptance."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('cfu_pool_fixture',Path(__file__).resolve().parents[1]/'models/native/src/pool.py')
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)


def fixture():
    originals=[{'role':'image','sample_id':n,'file_id':n,'group_id':n,'split':split}
               for n,split in [('a','train'),('b','train'),('v','validation'),('t','test')]]
    archives=[SimpleNamespace(tiles=[{'sample_id':n,'parent_file_id':n,'group_id':n,'split':'train',
                                     'file_name':n+'.jpg','sha256':n*64}]) for n in ['a','b']]
    return archives,originals


def test_full_training_coverage_retains_all_members_without_validation_or_test():
    archives,originals=fixture()
    rows=pool.index_training_crops(archives,originals)
    assert {t['sample_id'] for _,t in rows}=={'a','b'}


@pytest.mark.parametrize('tamper',['missing','duplicate','parent_in_two_archives','validation','parent','group'])
def test_incomplete_duplicate_or_changed_optimizer_pool_fails(tamper):
    archives,originals=fixture()
    if tamper=='missing': archives.pop()
    elif tamper=='duplicate':archives[0].tiles.append(copy.deepcopy(archives[0].tiles[0]))
    elif tamper=='parent_in_two_archives':
        tile=copy.deepcopy(archives[0].tiles[0]);tile['file_name']='different.jpg';archives[1].tiles.append(tile)
    elif tamper=='validation':archives[0].tiles[0]['split']='validation'
    elif tamper=='parent':archives[0].tiles[0]['parent_file_id']='changed'
    else:archives[0].tiles[0]['group_id']='changed'
    with pytest.raises(ValueError):pool.index_training_crops(archives,originals)


def test_mid_epoch_resume_and_short_last_batch_equal_uninterrupted_epochs():
    size,seed,batch=13,20261001,4
    expected=pool.epoch_order(size,seed,0)+pool.epoch_order(size,seed,1)
    epoch,cursor=0,0;actual=[]
    for _ in range(2):
        # Represent two separately executed chunks; only epoch/cursor persist.
        for _ in range(4):
            rows,(epoch,cursor)=pool.next_batch(size,seed,epoch,cursor,batch)
            actual+=rows
    assert actual==expected and (epoch,cursor)==(2,0)
    assert len(set(actual[:size]))==size


@pytest.mark.parametrize('cursor,batch',[(-1,4),(13,4),(True,4),(0,0),(0,False)])
def test_invalid_resume_cannot_skip_or_repeat_training_crops(cursor,batch):
    with pytest.raises(ValueError):pool.next_batch(13,20261001,0,cursor,batch)
