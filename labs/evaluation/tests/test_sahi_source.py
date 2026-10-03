"""Synthetic publisher-wheel transport/integrity fixtures, never licensed real-source evidence."""

import importlib.util
import io
from pathlib import Path
import stat
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('cfu_sahi_source_fixture',Path(__file__).resolve().parents[1]/'models/sahi-source/src/acquire.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)


def wheel(*,extra=None,license_ok=True,version='0.11.21'):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name in source.REQUIRED:z.writestr(name,b'# synthetic source, never executed')
        z.writestr('sahi-0.11.21.dist-info/METADATA',f'Name: sahi\nVersion: {version}\nRequires-Dist: synthetic-only\n')
        license_text='MIT License\nPermission is hereby granted, free of charge\nIN NO EVENT SHALL THE AUTHORS' if license_ok else 'unverified license'
        z.writestr('sahi-0.11.21.dist-info/LICENSE',license_text)
        if extra:extra(z)
    return stream.getvalue()


@pytest.mark.parametrize('change',['traversal','symlink','duplicate','version','license'])
def test_unverified_changed_or_unsafe_wheel_cannot_pass_source_inspection(change):
    def extra(z):
        if change=='traversal':z.writestr('../outside.py',b'unsafe')
        elif change=='duplicate':z.writestr(source.REQUIRED[0],b'repeated')
        elif change=='symlink':
            info=zipfile.ZipInfo('sahi/link.py');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,b'outside')
    with pytest.raises(ValueError):source.inspect(wheel(extra=extra,license_ok=change!='license',version='wrong' if change=='version' else source.VERSION))


@pytest.mark.parametrize('url',['http://pypi.org/x','https://unapproved.example/x','https://user:pass@pypi.org/x'])
def test_changed_publisher_host_or_credentials_fail_before_network(url):
    with pytest.raises(ValueError):source.fetch(url,100,expected_host='pypi.org')


def test_unqualified_license_is_retained_for_review_without_execution_permission():
    result=source.inspect(wheel(license_ok=False),retain_unqualified=True)
    assert result['source_inspection_passed'] is False
    assert result['license_observed']['license']=='unqualified'
    assert result['license_observed']['records'][0]['content']=='unverified license'
    assert result['requires_dist']==['synthetic-only']
    assert result['inspection_findings']


@pytest.mark.parametrize('change',['traversal','symlink','version'])
def test_retention_mode_never_weakens_source_identity_or_path_guards(change):
    def extra(z):
        if change=='traversal':z.writestr('../outside.py',b'unsafe')
        elif change=='symlink':
            info=zipfile.ZipInfo('sahi/link.py');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,b'outside')
    with pytest.raises(ValueError):
        source.inspect(wheel(extra=extra,license_ok=False,version='wrong' if change=='version' else source.VERSION),retain_unqualified=True)


def test_rewrapped_license_whitespace_is_not_a_false_rejection(monkeypatch):
    import hashlib
    stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(wheel())) as original,zipfile.ZipFile(stream,'w') as changed:
        for name in original.namelist():
            body=original.read(name)
            if name.endswith('/LICENSE'):body=body.replace(b'SHALL THE AUTHORS',b'SHALL THE\nAUTHORS')
            changed.writestr(name,body)
    raw=stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text=archive.read('sahi-0.11.21.dist-info/LICENSE')
    monkeypatch.setattr(source,'LICENSE_SHA256',hashlib.sha256(text).hexdigest())
    assert source.inspect(raw)['source_inspection_passed'] is True


def test_plausible_license_words_without_the_reviewed_identity_are_unqualified():
    assert source.inspect(wheel(),retain_unqualified=True)['source_inspection_passed'] is False
