"""Extract reviewed, pinned SAHI source without executing archive content."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from .native_source import pinned_bytes

VERSION = '0.11.21'
WHEEL_SHA = '31fa3dab37284e5cf0e404e74505781523a7faab59f9b2e1f847a436d9b6cf11'
LICENSE_SHA = '7b2e5b979948fb176f483b0e9404a237f94c485387755a32258f7bd6830fffe1'


def extract_sahi(archive_path, manifest_path, destination, *, archive_pin, manifest_pin):
    raw = pinned_bytes(archive_path, archive_pin, 16*1024*1024)
    m = json.loads(pinned_bytes(manifest_path, manifest_pin, 1024*1024))
    if (archive_pin['sha256'] != WHEEL_SHA or m.get('source_inspection_passed') is not True
            or m.get('package') != 'sahi' or m.get('version') != VERSION
            or m['provenance']['wheel_sha256'] != archive_pin['sha256']
            or m['provenance']['wheel_size_bytes'] != len(raw)
            or m['license_observed']['license'] != 'MIT'
            or not any(r['sha256'] == LICENSE_SHA for r in m['license_observed']['records'])):
        raise ValueError('SAHI requires the exact reviewed passing source receipt')
    records = m['members']
    if not 0 < len(records) <= 2000 or len({r['path'] for r in records}) != len(records):
        raise ValueError('SAHI member receipt is repeated or unbounded')
    expected = {r['path']:r for r in records}
    regular = {}
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist()) > 2000:
            raise ValueError('SAHI archive has too many members')
        seen = set()
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or (str(path)+'/' if info.is_dir() else str(path)) != info.filename
                    or stat.S_ISLNK(info.external_attr >> 16) or info.file_size > 8*1024*1024):
                raise ValueError('SAHI member is unsafe or unbounded')
            if info.is_dir():continue
            if info.filename in seen or info.filename not in expected:
                raise ValueError('SAHI archive differs from its member receipt')
            seen.add(info.filename);total += info.file_size
            if total > 32*1024*1024:raise ValueError('SAHI decoded bytes exceed their bound')
            body = z.read(info);pin = expected[info.filename]
            if len(body) != pin['size_bytes'] or hashlib.sha256(body).hexdigest() != pin['sha256']:
                raise ValueError('SAHI member bytes changed')
            regular[info.filename] = body
        if seen != set(expected):raise ValueError('SAHI source is incomplete')
    licenses = {r['path']:r for r in m['license_observed']['records']}
    if any(hashlib.sha256(regular[n]).hexdigest() != r['sha256'] for n,r in licenses.items()):
        raise ValueError('SAHI license receipt differs from actual primary bytes')
    destination = Path(destination)
    if destination.exists():raise ValueError('SAHI extraction must use a fresh private directory')
    destination.mkdir(mode=0o700)
    try:
        for name,body in regular.items():
            if name.startswith('sahi/') or name in licenses:
                p = destination / name;p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                p.write_bytes(body);p.chmod(0o600)
    except Exception:
        shutil.rmtree(destination)
        raise
    return {'version':VERSION, 'source_sha256':WHEEL_SHA, 'license_sha256':LICENSE_SHA,
            'verified_members':len(regular), 'third_party_code_executed_by_extraction':False}
