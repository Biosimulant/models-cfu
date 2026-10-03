"""Full frozen training coverage and resumable epoch order, independent of Torch."""

import random


def index_training_crops(archives, originals):
    eligible={m['sample_id']:m for m in originals if m['role']=='image' and m['split']=='train'}
    rows, seen, parents, parent_batches=[],set(),set(),{}
    for archive_index, archive in enumerate(archives):
        for tile in archive.tiles:
            name=tile['sample_id']
            parent=eligible.get(name)
            if (not parent or tile['split']!='train' or tile['parent_file_id']!=parent['file_id']
                    or any(tile[k]!=parent[k] for k in ('group_id','split'))):
                raise ValueError('Optimizer pool differs from a frozen training parent')
            identity=(tile['parent_file_id'],tile['file_name'])
            if identity in seen:
                raise ValueError('A training crop appears twice in the selected pool')
            if name in parent_batches and parent_batches[name]!=archive_index:
                raise ValueError('A training parent occurs in more than one selected archive')
            seen.add(identity);parents.add(name);parent_batches[name]=archive_index
            rows.append((archive_index,tile))
    if parents!=set(eligible):
        raise ValueError('Training pool does not cover every frozen training original')
    rows.sort(key=lambda pair:(pair[1]['sha256'],pair[1]['parent_file_id'],pair[1]['file_name']))
    return rows


def epoch_order(size, seed, epoch):
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (size,seed,epoch)) or size<1 or epoch<0:
        raise ValueError('Invalid epoch ordering contract')
    order=list(range(size))
    random.Random(f'cfu-epoch-order-v1:{seed}:{epoch}').shuffle(order)
    return order


def next_batch(size, seed, epoch, cursor, batch_size):
    if (isinstance(cursor,bool) or not isinstance(cursor,int) or not 0<=cursor<size
            or isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1):
        raise ValueError('Invalid retained training cursor or batch size')
    order=epoch_order(size,seed,epoch)
    end=min(size,cursor+batch_size)
    return order[cursor:end], ((epoch+1,0) if end==size else (epoch,end))
