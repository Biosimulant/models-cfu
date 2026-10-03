"""Pure checkpoint progress and augmentation geometry contracts."""

import hashlib
import json
import math


def recipe_digest(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_progress(progress, size, batch_size, max_epochs):
    epoch,cursor,step=(progress[k] for k in ('epoch','cursor','global_step'))
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (epoch,cursor,step))
            or not 0<=epoch<=max_epochs or not 0<=cursor<size or step<0
            or cursor%batch_size or (epoch==max_epochs and cursor)):
        raise ValueError('Checkpoint progress cannot represent the frozen training order')
    if step!=epoch*math.ceil(size/batch_size)+cursor//batch_size:
        raise ValueError('Checkpoint optimizer steps differ from its epoch/crop cursor')
    return epoch,cursor,step


def mirrored_rows(rows, width):
    if isinstance(width,bool) or not isinstance(width,int) or width<=0:
        raise ValueError('Invalid horizontal augmentation frame')
    output=[]
    for cls,x,y,w,h in rows:
        if not 0<=x-w/2<=x+w/2<=width:
            raise ValueError('Target differs from its horizontal augmentation frame')
        output.append([cls,width-x,y,w,h])
    return output


def augmentation_seed(seed, step, tile):
    key=f"cfu-training-augmentation-v1:{seed}:{step}:{tile['parent_file_id']}:{tile['file_name']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')
