"""Accept a regular-file derivative only with the exact original source receipt.

The original extractor and detector bytes remain unchanged. This wrapper verifies
the derivative's ancestry before the original extractor verifies every ZIP member.
"""
import hashlib
import json

from .native_source import COMMIT, PRIMARY_DIGESTS, extract_native_source, pinned_bytes

ORIGINAL_ARCHIVE = {'sha256': '647bba7d16cc7e13feb5f2beca0296f86186c944f5a8fa9ccefc2629758bccba',
                    'size_bytes': 4507433}
ORIGINAL_RECEIPT = {'sha256': '994c43d110e431f124bb83e8d4934d026ac58cfcb640463707600222b8e2c118',
                    'size_bytes': 36374}
DERIVATIVE_FORMAT = 'cfu-yolox-regular-hosted-source-v1'


def canonical_receipt(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def selected_members(original):
    return sorted((m for m in original['members'] if m['kind'] == 'regular_file'
                   and (m['path'] in PRIMARY_DIGESTS
                        or m['path'].startswith(('yolox/', 'exps/')))), key=lambda m: m['path'])


def verify_ancestor(manifest):
    lineage = manifest['hosted_derivative']
    if (lineage['format'] != DERIVATIVE_FORMAT
            or lineage['original_archive'] != ORIGINAL_ARCHIVE
            or lineage['original_receipt'] != ORIGINAL_RECEIPT):
        raise ValueError('Hosted source derivative has an unapproved ancestor')
    original = lineage['original_manifest']
    raw = canonical_receipt(original)
    if (len(raw) != ORIGINAL_RECEIPT['size_bytes']
            or hashlib.sha256(raw).hexdigest() != ORIGINAL_RECEIPT['sha256']):
        raise ValueError('Hosted source derivative changed the original receipt')
    if (original['source_inspection_passed'] is not True or original['commit'] != COMMIT
            or original['source_archive_sha256'] != ORIGINAL_ARCHIVE['sha256']
            or original['source_archive_size_bytes'] != ORIGINAL_ARCHIVE['size_bytes']
            or original['primary_records_verified'] != PRIMARY_DIGESTS
            or manifest['members'] != selected_members(original)):
        raise ValueError('Hosted source derivative changed the verified source subset')
    return original


def extract_hosted_source(archive_path, manifest_path, destination, *, archive_pin, manifest_pin):
    manifest = json.loads(pinned_bytes(manifest_path, manifest_pin, 1024 * 1024))
    derived = 'hosted_derivative' in manifest
    if derived:
        verify_ancestor(manifest)
    elif archive_pin != ORIGINAL_ARCHIVE or manifest_pin != ORIGINAL_RECEIPT:
        raise ValueError('Original hosted source differs from the approved acquisition')
    report = extract_native_source(archive_path, manifest_path, destination,
                                   archive_pin=archive_pin, manifest_pin=manifest_pin)
    if derived:
        report.update(source_package=DERIVATIVE_FORMAT,
                      original_acquisition_archive=ORIGINAL_ARCHIVE,
                      verified_original_receipt=ORIGINAL_RECEIPT,
                      original_code_bytes_unchanged=True)
    return report
