"""Checksum-bound original-frame overlays and offline interactive validation review."""
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import stat
import sys
import zipfile
from uuid import UUID

from .core import candidate_grid, original_image, validation_members
from .metrics import summarize
from .native_source import pinned_bytes

HTML='''<!doctype html><meta charset="utf-8"><title>CFU validation review</title>
<style>body{font:16px system-ui;background:#101b25;color:#edf4fa;max-width:1200px;margin:24px auto;padding:0 20px}h1{font-size:28px}a{color:#a3d4ff}select,button{font:inherit;padding:8px;background:#203446;color:inherit;border:1px solid #59748c;border-radius:5px}label{margin-right:16px}svg{width:100%;max-height:75vh;background:#070d13;margin-top:16px}.box:hover{stroke-width:5;cursor:crosshair}#detail{min-height:24px;color:#c4d9ed}.legend{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0}.green{color:#48de8b}.orange{color:#ffae42}.pink{color:#ff69db}.blue{color:#77cfff}#summary{line-height:1.7}</style>
<h1>CFU validation detections</h1><p id="scope"></p><p id="gates"></p>
<select id="plate" aria-label="Validation plate"></select>
<p id="summary"></p><label><input id="predictions" type="checkbox" checked>Detections</label>
<label><input id="references" type="checkbox" checked>Original annotations</label>
<a id="annotated" href="#">Annotated image</a> · <a href="review.json">Full detections and provenance</a>
<div class="legend"><span class="green">Matched detection</span><span class="orange">Unmatched detection</span><span class="pink">Missed annotation</span><span class="blue">Matched annotation</span></div>
<div id="detail">Hover over a box for its original coordinates and detection score.</div><svg id="scene" aria-label="Original plate with inspectable colony boxes"></svg>
<p>Boxes use the original annotation frame after EXIF orientation. Display images are resized derivatives. Detection scores are model scores, not calibrated probabilities. A zero-area reference is shown as a cross; original coordinates and labels remain in the JSON.</p>
<p>Research assistance for visible colonies in the ADBC source domain; human review required. Final test, replay and independent training recreation remain separate qualification steps.</p>
<p>Source photographs and annotations: Norbert Solymosi and Sára Nagy, <a href="https://doi.org/10.6084/m9.figshare.22022540.v3">ADBC dataset v3</a>, CC BY 4.0. These are validation plates.</p>
<script>const data=__DATA__;const ns='http://www.w3.org/2000/svg';
const byId=id=>document.getElementById(id);const picker=byId('plate');
data.plates.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.sample_id;picker.append(o)});
byId('scope').textContent=`${data.plates.length} validation plates · Training step ${data.weights_progress.global_step} · ${data.configuration.id}`;
const m=data.metrics;byId('gates').textContent=`Validation gates ${m.acceptance_passed?'passed; final qualification pending':'not passed'}. Count error ${(100*m.wape).toFixed(2)}%; 95th-percentile error ${(100*m.p95_symmetric_error).toFixed(2)}%; precision ${(100*m.precision).toFixed(2)}%; recall ${(100*m.recall).toFixed(2)}%.`;
function draw(){const p=data.plates[Number(picker.value)];const scene=byId('scene');scene.replaceChildren();scene.setAttribute('viewBox',`0 0 ${p.image_width} ${p.image_height}`);
const image=document.createElementNS(ns,'image');image.setAttribute('href',p.normalized_image);image.setAttribute('width',p.image_width);image.setAttribute('height',p.image_height);image.setAttribute('preserveAspectRatio','none');scene.append(image);
byId('summary').textContent=`${p.sample_id}: ${p.predicted_count} detections, ${p.reference_count} original annotations; ${p.tp} matched, ${p.fp} unmatched detections, ${p.fn} missed annotations. Count symmetric error ${(100*p.symmetric_error).toFixed(2)}%.`;
byId('annotated').href=p.annotated_image;const pm=new Set(p.matches.map(x=>x.prediction_index));const rm=new Set(p.matches.map(x=>x.reference_index));
function box(b,color,label){const [x0,y0,x1,y1]=b;const e=document.createElementNS(ns,x1===x0||y1===y0?'path':'rect');if(e.tagName==='path')e.setAttribute('d',`M${x0-5},${y0-5}L${x0+5},${y0+5}M${x0+5},${y0-5}L${x0-5},${y0+5}`);else{e.setAttribute('x',x0);e.setAttribute('y',y0);e.setAttribute('width',x1-x0);e.setAttribute('height',y1-y0)}e.setAttribute('fill','none');e.setAttribute('stroke',color);e.setAttribute('stroke-width','2');e.setAttribute('vector-effect','non-scaling-stroke');e.setAttribute('class','box');const title=document.createElementNS(ns,'title');title.textContent=label;e.append(title);e.onmouseenter=()=>byId('detail').textContent=label;scene.append(e)}
if(byId('references').checked)p.reference_boxes_xyxy.forEach((b,i)=>box(b,rm.has(i)?'#77cfff':'#ff69db',`Annotation ${i}: [${b.join(', ')}]`));
if(byId('predictions').checked)p.detections.forEach((d,i)=>box(d.bbox_xyxy,pm.has(i)?'#48de8b':'#ffae42',`Detection ${i}; score ${d.score.toFixed(4)}; [${d.bbox_xyxy.join(', ')}]`));}
picker.onchange=draw;byId('predictions').onchange=draw;byId('references').onchange=draw;draw();</script>'''


def checked_origin(origin):
    UUID(origin['run_id']);UUID(origin['revision_id'])
    for name in ('comparison','predictions'):
        pin=origin[name];UUID(pin['file_id']);sha=pin['sha256'];size=pin['size_bytes']
        if (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)
                or isinstance(size,bool) or not isinstance(size,int) or not 0<size<=64*1024*1024):
            raise ValueError('Review requires exact bounded original validation artifact identities')


def selected_evidence(inputs,plan,origin):
    checked_origin(origin)
    frozen=json.loads(pinned_bytes(inputs['frozen_split'],plan['frozen_split'],1024*1024))
    rows=validation_members(frozen,plan)
    comparison=json.loads(pinned_bytes(inputs['comparison'],origin['comparison'],64*1024*1024))
    if (comparison['stage']!='actual_validation_candidate_comparison'
            or comparison['frozen_dataset_sha256']!=plan['frozen_dataset_sha256']
            or comparison['frozen_dataset_id']!=plan['frozen_dataset_id']
            or comparison['candidate_grid']!=candidate_grid()
            or comparison['validation_selection_complete'] is not True
            or comparison['test_outcomes_inspected'] is not False
            or comparison['final_test_completed'] is not False
            or comparison['scientific_acceptance_established'] is not False
            or comparison['selected_configuration'] not in candidate_grid()):
        raise ValueError('Review needs actual frozen validation-only evidence')
    raw=pinned_bytes(inputs['predictions'],origin['predictions'],64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos=archive.infolist();expected={c['id']+'.json' for c in candidate_grid()}
        if len(infos)!=32 or {x.filename for x in infos}!=expected or sum(x.file_size for x in infos)>128*1024*1024:
            raise ValueError('Review requires the original complete32-configuration prediction archive')
        name=comparison['selected_configuration']['id']+'.json';info=archive.getinfo(name)
        if (info.flag_bits&1 or stat.S_IFMT(info.external_attr>>16) not in (0,stat.S_IFREG)
                or not 0<info.file_size<=32*1024*1024):
            raise ValueError('Selected validation document is not bounded regular JSON')
        payload=json.loads(archive.read(info))
    if payload['configuration']!=comparison['selected_configuration']:
        raise ValueError('Selected predictions changed configuration')
    plates=payload['plates'];by_name={r['sample_id']:r for r in rows}
    if any(p['group_id']!=by_name[p['sample_id']]['group_id'] for p in plates):
        raise ValueError('Selected predictions changed frozen groups')
    metrics=summarize(plates,expected_sample_ids=by_name)
    if metrics!=payload['metrics'] or metrics!=comparison['selected_validation_metrics'] or metrics['failures']:
        raise ValueError('Review requires complete unchanged metrics derived from retained predictions')
    return rows,comparison,plates,metrics


def render_photo(image,plate,measured):
    from PIL import Image, ImageDraw, ImageFont
    width,height=image.size;ratio=min(1,1600/max(width,height))
    size=(max(1,round(width*ratio)),max(1,round(height*ratio)))
    display=image.resize(size,Image.Resampling.LANCZOS);background=io.BytesIO();display.save(background,format='JPEG',quality=90)
    annotated=Image.new('RGB',(size[0],size[1]+90),'#101b25');annotated.paste(display,(0,0));display.close()
    draw=ImageDraw.Draw(annotated);pm={x['prediction_index'] for x in measured['matches']};rm={x['reference_index'] for x in measured['matches']}
    def box(b,color):
        x0,y0,x1,y1=b;x0*=size[0]/width;x1*=size[0]/width;y0*=size[1]/height;y1*=size[1]/height
        if x0==x1 or y0==y1:
            draw.line((x0-4,y0-4,x0+4,y0+4),fill=color,width=2);draw.line((x0+4,y0-4,x0-4,y0+4),fill=color,width=2)
        else:draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
    for i,b in enumerate(plate['reference_boxes_xyxy']):box(b,'#77cfff' if i in rm else '#ff69db')
    for i,d in enumerate(plate['detections']):box(d['bbox_xyxy'],'#48de8b' if i in pm else '#ffae42')
    font=ImageFont.load_default(size=16)
    lines=[plate['sample_id']+': '+str(measured['predicted_count'])+' detections / '+str(measured['reference_count'])+' original annotations',
           'Green: matched detection; orange: unmatched detection; pink: missed annotation; blue: matched annotation',
           'Validation review only. Source: Solymosi and Nagy, ADBC v3, CC BY 4.0.']
    for i,line in enumerate(lines):draw.text((8,size[1]+6+i*25),line,font=font,fill='white')
    output=io.BytesIO();annotated.save(output,format='JPEG',quality=90);annotated.close()
    return background.getvalue(),output.getvalue()


def run(request,contract,module_root):
    module_root=Path(module_root);root=Path(request['root']);inputs=request['inputs']
    for name,pin in contract['authored_code'].items():pinned_bytes(module_root/'src'/name,pin,128*1024)
    if set(inputs)!={'frozen_split','comparison','predictions',*{f'image_{i:02d}' for i in range(37)}}:
        raise ValueError('Review rejects test/annotation inputs or missing original validation photos')
    packages={n:importlib.metadata.version(n) for n in contract['required_package_versions']}
    if packages!=contract['required_package_versions']:raise RuntimeError('Review dependency versions differ from the pinned managed contract')
    rows,comparison,plates,metrics=selected_evidence(inputs,contract['validation_plan'],request['validation_origin'])
    predictions={p['sample_id']:p for p in plates};measurements={p['sample_id']:p for p in metrics['plates']}
    review={'stage':'actual_validation_visual_review','validation_origin':request['validation_origin'],
            'configuration':comparison['selected_configuration'],'weights':comparison['weights'],
            'weights_progress':comparison['weights_progress'],'environment':comparison['environment'],
            'source':comparison['source'],'sahi':comparison['sahi'],'metrics':metrics,
            'test_outcomes_inspected':False,'scientific_acceptance_established':False,'plates':[]}
    path=root/'validation-review.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,row in enumerate(rows):
            plate=predictions[row['sample_id']];measured=measurements[row['sample_id']]
            image=original_image(inputs[f'image_{i:02d}'],row,{'width':plate['image_width'],'height':plate['image_height']})
            try:background,annotated=render_photo(image,plate,measured)
            finally:image.close()
            normal_name=f'normalized/{i:02d}.jpg';annotated_name=f'annotated/{i:02d}.jpg'
            archive.writestr(normal_name,background);archive.writestr(annotated_name,annotated)
            review['plates'].append({**plate,**measured,'original':row,'normalized_image':normal_name,'annotated_image':annotated_name})
        raw=json.dumps(review,allow_nan=False,separators=(',',':'))
        archive.writestr('review.json',raw)
        archive.writestr('index.html',HTML.replace('__DATA__',raw.replace('<','\\u003c')))
    if not 0<path.stat().st_size<=64*1024*1024:raise ValueError('Review ZIP exceeds its artifact bound')
    identity={'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt={'stage':'managed_actual_validation_visual_review','plate_count':len(rows),
             'configuration':review['configuration'],'weights':review['weights'],'validation_origin':request['validation_origin'],
             'validation_metrics':{k:v for k,v in metrics.items() if k not in {'plates','bootstrap'}},
             'environment':{'python':platform.python_version(),'package_versions':packages},
             'test_outcomes_inspected':False,'scientific_acceptance_established':False,'artifact':identity}
    return {'receipt':receipt,'review_path':str(path.resolve())}


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_bytes());module_root=Path(__file__).resolve().parent.parent
    contract=json.loads((module_root/'review-plan.json').read_bytes())
    result=run(request,contract,module_root)
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False));Path(sys.argv[2]).chmod(0o600)
