"""Inspect an exact SAHI wheel from publisher bytes without executing its code."""

import email.parser
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile

from biosimulant import BioModule,ExecutionPolicy,SignalSpec

VERSION='0.11.21'
WHEEL_SHA256='31fa3dab37284e5cf0e404e74505781523a7faab59f9b2e1f847a436d9b6cf11'
LICENSE_SHA256='7b2e5b979948fb176f483b0e9404a237f94c485387755a32258f7bd6830fffe1'
REQUIRED=['sahi/__init__.py','sahi/slicing.py','sahi/prediction.py','sahi/postprocess/combine.py']
SCHEMA={'stage':'str','package':'str','version':'str','provenance':'json','source_inspection_passed':'bool',
        'license_observed':'json','required_source':'json','member_count':'int','third_party_code_executed':'bool'}


def fetch(url,maximum,*,expected_host):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or parsed.hostname!=expected_host or parsed.username or parsed.password:
        raise ValueError('Source URL is outside the declared publisher contract')
    request=urllib.request.Request(url,headers={'User-Agent':'Biosimulant-CFU-source-inspection/0.1'})
    with urllib.request.urlopen(request,timeout=30) as response:
        if urllib.parse.urlparse(response.url).hostname!=expected_host:raise ValueError('Source redirect changed publisher host')
        raw=response.read(maximum+1)
    if len(raw)>maximum:raise ValueError('Publisher source exceeds its byte bound')
    return raw


def inspect(raw, *, retain_unqualified=False):
    records=[];licenses=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        if len(entries)>2000 or len({e.filename for e in entries})!=len(entries):raise ValueError('Source wheel has repeated/excessive members')
        if sum(e.file_size for e in entries)>32*1024*1024:raise ValueError('Source wheel decoded bytes exceed bound')
        for entry in entries:
            path=PurePosixPath(entry.filename);mode=(entry.external_attr>>16)&0xffff
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.filename or (str(path)+'/' if entry.is_dir() else str(path))!=entry.filename
                    or stat.S_ISLNK(mode) or entry.file_size>8*1024*1024):
                raise ValueError('Source wheel contains unsafe or unbounded members')
            if entry.is_dir():continue
            body=archive.read(entry)
            records.append({'path':entry.filename,'size_bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
            if path.name.lower() in {'license','license.txt','license.md'}:
                text=body.decode('utf-8')
                licenses.append({'path':entry.filename,'sha256':hashlib.sha256(body).hexdigest(),'content':text})
        names={r['path'] for r in records}
        if not set(REQUIRED)<=names:raise ValueError('Wheel lacks required sliced-inference source')
        metadata_name=next((n for n in names if n.endswith('.dist-info/METADATA')),None)
        if not metadata_name:raise ValueError('Wheel lacks its publisher package/version metadata')
        metadata=email.parser.BytesParser().parsebytes(archive.read(metadata_name))
        if metadata.get('Name','').lower()!='sahi' or metadata.get('Version')!=VERSION:
            raise ValueError('Wheel package/version differs from the declared exact source')
        approved=[l for l in licenses if l['sha256']==LICENSE_SHA256 and all(t in ' '.join(l['content'].split()) for t in ['MIT License','Permission is hereby granted, free of charge','IN NO EVENT SHALL THE AUTHORS'])]
        if not approved and not retain_unqualified:raise ValueError('Permissive source license has not been observed in publisher wheel')
        return {'members':records,'source_inspection_passed':bool(approved),
                'inspection_findings':[] if approved else ['Primary license text requires review; downstream execution is prohibited.'],
                'license_observed':{'license':'MIT' if approved else 'unqualified','records':licenses},
                'required_source':[r for r in records if r['path'] in REQUIRED],
                'requires_dist':metadata.get_all('Requires-Dist',[]),'metadata_path':metadata_name}


class InspectSahiSource(BioModule):
    execution_policy=ExecutionPolicy.ONCE_BEFORE_RUN

    def inputs(self):return {}

    def outputs(self):
        return {'receipt':SignalSpec.record(schema=SCHEMA,emitted_unit='1'),
                'source_archive_path':SignalSpec.scalar(dtype='str',value_type='file',format='zip'),
                'source_manifest_path':SignalSpec.scalar(dtype='str',value_type='file',format='json')}

    def execute(self,inputs,*,context):
        url=f'https://pypi.org/pypi/sahi/{VERSION}/json'
        metadata_raw=fetch(url,1024*1024,expected_host='pypi.org');metadata=json.loads(metadata_raw)
        candidates=[f for f in metadata['urls'] if f['packagetype']=='bdist_wheel' and f['filename'].endswith('py3-none-any.whl')]
        if len(candidates)!=1:raise ValueError('Source has no unique pure-Python publisher wheel')
        selected=candidates[0]
        raw=fetch(selected['url'],16*1024*1024,expected_host='files.pythonhosted.org')
        sha=hashlib.sha256(raw).hexdigest()
        if len(raw)!=selected['size'] or sha!=selected['digests']['sha256'] or sha!=WHEEL_SHA256:raise ValueError('Source wheel differs from publisher length/digest')
        manifest=inspect(raw,retain_unqualified=True)
        provenance={'metadata_url':url,'metadata_sha256':hashlib.sha256(metadata_raw).hexdigest(),
                    'wheel_url':selected['url'],'wheel_file_name':selected['filename'],'wheel_size_bytes':len(raw),'wheel_sha256':sha}
        manifest.update({'package':'sahi','version':VERSION,'provenance':provenance,'third_party_code_executed':False})
        root=Path(tempfile.mkdtemp(prefix='cfu-sahi-source-',dir=Path.cwd()));root.chmod(0o700)
        archive_path=root/'sahi-source-wheel.zip';manifest_path=root/'sahi-source-manifest.json'
        archive_path.write_bytes(raw);manifest_path.write_text(json.dumps(manifest,indent=2));archive_path.chmod(0o600);manifest_path.chmod(0o600)
        receipt={'stage':'exact_sahi_publisher_wheel_inspection','package':'sahi','version':VERSION,'provenance':provenance,
                 'source_inspection_passed':manifest['source_inspection_passed'],'license_observed':manifest['license_observed'],
                 'required_source':manifest['required_source'],'member_count':len(manifest['members']),'third_party_code_executed':False}
        return {'receipt':receipt,'source_archive_path':str(archive_path),'source_manifest_path':str(manifest_path)}
