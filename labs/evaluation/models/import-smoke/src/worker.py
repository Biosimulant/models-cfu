"""Fresh interpreter verifies and exercises the actual pinned inference imports."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .native_source import extract_native_source, pinned_bytes
from .preprocessing import load_preproc
from .sahi_source import extract_sahi


def run(inputs,root,plan):
    root=Path(root)
    for name,pin in plan['authored_code'].items():
        pinned_bytes(Path(__file__).parent/name,pin,128*1024)
    native=extract_native_source(inputs['source_archive'],inputs['source_receipt'],root/'native-source',
                                archive_pin=plan['source_archive'],manifest_pin=plan['source_receipt'])
    sahi=extract_sahi(inputs['sahi_archive'],inputs['sahi_receipt'],root/'sahi-source',
                      archive_pin=plan['sahi_archive'],manifest_pin=plan['sahi_receipt'])
    if any(n=='yolox' or n.startswith('yolox.') or n=='sahi' or n.startswith('sahi.') for n in sys.modules):
        raise RuntimeError('Import check must not reuse cached third-party modules')
    versions={n:importlib.metadata.version(n) for n in plan['required_package_versions']}
    if any(versions[n].split('+')[0] != v for n,v in plan['required_package_versions'].items()):
        raise RuntimeError('Import check package versions differ from the declared inference dependencies')
    distribution=importlib.metadata.distribution('pybboxes')
    notices={}
    for file in distribution.files or []:
        if Path(str(file)).name.upper() in {'LICENSE','LICENSE.TXT','LICENSE.MD'}:
            path=Path(distribution.locate_file(file))
            if path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 64*1024:
                raw=path.read_bytes();text=raw.decode('utf-8')
                notices[str(file)]={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'text':text}
    if not any('MIT License' in v['text'] and 'Permission is hereby granted' in v['text'] for v in notices.values()):
        raise RuntimeError('Installed pybboxes has no retained primary MIT license notice')
    sys.path[:0]=[str(root/'native-source'),str(root/'sahi-source')]
    import numpy as np
    import cv2
    import torch
    import torchvision
    from yolox.exp import Exp
    preproc=load_preproc(root/'native-source')
    from sahi.slicing import get_slice_bboxes
    from sahi.prediction import ObjectPrediction
    from sahi.postprocess.combine import NMSPostprocess, GreedyNMMPostprocess
    from .validation_lib import merge

    # All pixel values and boxes below are explicit synthetic software fixtures.
    pixels=np.zeros((160,320,3),dtype=np.uint8);pixels[:]=[11,22,33]
    prepared,ratio=preproc(pixels,(640,640))
    if (ratio != 2 or prepared.shape != (3,640,640) or prepared.dtype != np.float32
            or not prepared.flags.c_contiguous
            or not np.array_equal(prepared[:,0,0],np.array([11,22,33],dtype=np.float32))
            or not np.all(prepared[:,320:,:]==114)):
        raise RuntimeError('Actual native preprocessing differs from its BGR, padding and tensor contract')
    windows=get_slice_bboxes(image_height=960,image_width=1280,slice_height=640,slice_width=640,
                            overlap_height_ratio=.2,overlap_width_ratio=.2,auto_slice_resolution=False)
    if (not windows or any(not (0<=x0<x1<=1280 and 0<=y0<y1<=960 and x1-x0==640 and y1-y0==640)
                           for x0,y0,x1,y1 in windows)
            or not any(x0==0 and y0==0 for x0,y0,_,_ in windows)
            or not any(x1==1280 and y1==960 for _,_,x1,y1 in windows)):
        raise RuntimeError('Actual SAHI slicing violates the synthetic bounds fixture')
    fixture=[{'bbox_xyxy':[10,10,30,30],'score':.9},
             {'bbox_xyxy':[10,10,30,30],'score':.8},
             {'bbox_xyxy':[100,100,120,120],'score':.7}]
    merging={}
    for method,metric,threshold in [('NMS','IOU',.5),('GREEDYNMM','IOS',.5),('GREEDYNMM','IOS',.7)]:
        config={'confidence':.05,'method':method,'match_metric':metric,'match_threshold':threshold}
        rows=merge(fixture,config,1280,960,ObjectPrediction,NMSPostprocess,GreedyNMMPostprocess)
        if len(rows)!=2 or sorted(r['bbox_xyxy'] for r in rows)!=[[10,10,30,30],[100,100,120,120]]:
            raise RuntimeError('Actual SAHI merging fails duplicate/disjoint synthetic software fixtures')
        merging[f'{method}-{metric}-{threshold}']=rows
    torch.set_num_threads(4);torch.manual_seed(20261001)
    exp=Exp();exp.depth=.33;exp.width=.375;exp.num_classes=1
    model=exp.get_model().eval()
    with torch.inference_mode():decoded=model(torch.from_numpy(prepared).unsqueeze(0))
    if decoded.ndim!=3 or decoded.shape[0]!=1 or decoded.shape[2]!=6 or not torch.isfinite(decoded).all().item():
        raise RuntimeError('Actual CPU native single-class forward has invalid decoded outputs')
    return {'stage':'managed_cpu_inference_import_software_check',
            'checks':{'preprocessing':{'shape':list(prepared.shape),'ratio':ratio,'padding':114},
                      'slicing':{'windows':windows},'merging':merging,
                      'random_weight_cpu_forward':{'shape':list(decoded.shape),'finite':bool(torch.isfinite(decoded).all().item())}},
            'environment':{'python':platform.python_version(),'package_versions':versions,
                           'opencv_api':cv2.__version__,'torch':str(torch.__version__),
                           'torchvision':str(torchvision.__version__),'device':'cpu'},
            'sources':{'native':native,'sahi':sahi},
            'dependency_license':{'package':'pybboxes','version':versions['pybboxes'],'license':'MIT','notices':notices},
            'synthetic_software_fixtures_only':True,'test_outcomes_inspected':False,
            'scientific_acceptance_established':False}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_bytes())
    plan=json.loads((Path(__file__).resolve().parent.parent/'import-plan.json').read_bytes())
    result=run(request['inputs'],request['root'],plan)
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False));Path(sys.argv[2]).chmod(0o600)
