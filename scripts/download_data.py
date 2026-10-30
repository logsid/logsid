#!/usr/bin/env python3
"""Download, verify, and unpack the original LogSID dataset ZIP archives."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import time
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('Apache', 'BGL', 'OpenSSH', 'HealthApp', 'AIT-mail', 'Zanbil', 'Mac', 'Thunderbird')
CHUNK_SIZE = 1 << 20
SHARED_FILES = ('taxonomy.csv', 'statistics.csv', 'templates.csv', 'THIRD_PARTY_NOTICES.txt')


def validate_integrity(name, info):
    if (not isinstance(info, dict) or type(info.get('bytes')) is not int or info['bytes'] < 0
            or not isinstance(info.get('sha256'), str)
            or re.fullmatch(r'[0-9a-fA-F]{64}', info['sha256']) is None):
        raise ValueError(f'Dataset manifest has invalid integrity metadata for {name}')


def validate_url(url, *, allow_query=False):
    try:
        parsed = urlparse(url) if isinstance(url, str) else None
        valid = (parsed is not None and parsed.scheme in ('https', 'http', 'file')
                 and not parsed.fragment and (allow_query or not parsed.query)
                 and not parsed.username and not parsed.password
                 and (parsed.scheme == 'file' or bool(parsed.netloc))
                 and not any(character.isspace() or ord(character) < 32 for character in url))
    except ValueError:
        valid = False
    if not valid:
        # Never include a complete download URL: its query may grant read access.
        raise ValueError('Download URL must use HTTP(S) or file with no fragment or embedded credentials')
    return url


def load_manifest(path, base_url=None):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise ValueError('Dataset manifest must contain a files mapping')
    override_urls = base_url is not None
    if override_urls:
        manifest['base_url'] = base_url
    validate_url(manifest.get('base_url'))
    for name, info in manifest['files'].items():
        if (not isinstance(name, str) or not name or name in ('.', '..')
                or any(character in name for character in ('/', '\\', ':', '\x00'))
                or Path(name).name != name):
            raise ValueError('Dataset manifest contains an unsafe filename')
        validate_integrity(name, info)
    if not isinstance(manifest.get('archives'), dict):
        raise ValueError('Dataset manifest must contain an archives mapping')
    for source, info in manifest['archives'].items():
        if (source not in SOURCES or not isinstance(info, dict)
                or info.get('filename') != source + '.zip'):
            raise ValueError('Dataset manifest contains an unsafe archive filename or unknown source')
        validate_integrity(info['filename'], info)
        if 'url' in info:
            validate_url(info['url'], allow_query=True)
            if override_urls:
                info.pop('url')
        if 'parts' in info:
            if not isinstance(info['parts'], list) or not info['parts']:
                raise ValueError('Split archives require a nonempty ordered parts list')
            names = set()
            for part in info['parts']:
                name = part.get('filename') if isinstance(part, dict) else None
                if (not isinstance(name, str) or name in names
                        or re.fullmatch(re.escape(info['filename']) + r'\.\d{3}', name) is None):
                    raise ValueError('Dataset manifest contains an unsafe or duplicate archive part')
                names.add(name)
                validate_integrity(name, part)
                if part['bytes'] == 0:
                    raise ValueError('Archive parts must not be empty')
                if 'url' in part:
                    validate_url(part['url'], allow_query=True)
                    if override_urls:
                        part.pop('url')
            if sum(part['bytes'] for part in info['parts']) != info['bytes']:
                raise ValueError('Archive parts do not add up to the complete ZIP size')
    for name in SHARED_FILES:
        if name not in manifest['files']:
            raise ValueError(f'Dataset manifest is missing shared metadata: {name}')
    return manifest


def verify_file(path, info):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular dataset file: {path}')
    if path.stat().st_size != info['bytes']:
        raise ValueError(f'Dataset size mismatch: {path.name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b''):
            digest.update(block)
    if digest.hexdigest() != info['sha256'].lower():
        raise ValueError(f'Dataset SHA-256 mismatch: {path.name}')


def transfer(url, partial, expected_size, timeout):
    """Resume a partial transfer if the server honors Range, else restart it."""
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f'Expected a regular partial file: {partial}')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        status = response.getcode()
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if (match is None or int(match[1]) != offset or int(match[3]) != expected_size
                    or int(match[2]) < offset or int(match[2]) >= expected_size):
                raise ValueError('Server returned an inconsistent Content-Range')
            mode = 'ab'
        else:
            # Includes file:// fixtures and HTTP servers without Range support.
            mode, offset = 'wb', 0
        with partial.open(mode) as output:
            received = offset
            while block := response.read(CHUNK_SIZE):
                received += len(block)
                if received > expected_size:
                    raise ValueError('Download exceeds the size recorded in the manifest')
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if received != expected_size:
            raise http.client.IncompleteRead(b'', expected_size - received)


def download_file(base_url, name, info, data_dir, *, url=None, verify_only=False, retries=2, timeout=30):
    destination = Path(data_dir) / name
    if destination.exists() or destination.is_symlink():
        # Never replace an existing final file, including one with invalid bytes.
        verify_file(destination, info)
        return 'verified'
    if verify_only:
        raise FileNotFoundError(f'Missing dataset file: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f'Expected a regular partial file: {partial}')
    if partial.exists() and partial.stat().st_size == info['bytes']:
        try:
            verify_file(partial, info)
        except ValueError:
            partial.unlink()
        else:
            os.replace(partial, destination)
            return 'downloaded'
    url = validate_url(url if url is not None else base_url.rstrip('/') + '/' + quote(name, safe=''),
                       allow_query=True)
    for attempt in range(retries + 1):
        try:
            try:
                transfer(url, partial, info['bytes'], timeout)
            except ValueError:
                raise ValueError(f'Invalid download response for {name}') from None
            verify_file(partial, info)
            os.replace(partial, destination)
            return 'downloaded'
        except ValueError:
            # Invalid complete bytes cannot be repaired by appending a suffix.
            partial.unlink(missing_ok=True)
            raise
        except (URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException):
            if attempt == retries:
                raise RuntimeError(f'Download interrupted for {name}; rerun to resume') from None
            time.sleep(min(2 ** attempt, 4))
    raise AssertionError('Unreachable transfer state')


def unpack_archive(path, source, files, data_dir):
    """Validate and stage this source's logs, labels, and shared metadata."""
    names = (source + '.log.gz', source + '_labels.jsonl.gz', *SHARED_FILES)
    members = {'data/' + name: name for name in names}
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        seen = set()
        for member in archive.infolist():
            name = member.filename
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (name != member.orig_filename or '\\' in name or '\x00' in name
                    or name not in members.keys() | {'data/'}
                    or name in seen or stat.S_ISLNK(member.external_attr >> 16)
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (name == 'data/' and not member.is_dir())
                    or (name != 'data/' and (member.is_dir() or kind == stat.S_IFDIR))):
                raise ValueError('ZIP contains an unsafe, duplicate, or unexpected member')
            seen.add(name)
            if name in members and member.file_size != files[members[name]]['bytes']:
                raise ValueError(f'ZIP member size mismatch: {members[name]}')
        if not members.keys() <= seen:
            raise ValueError(f'ZIP is missing the original files for {source}')
        # Only fixed filenames reach data_dir; all six files must verify
        # before any new final file is installed.
        with tempfile.TemporaryDirectory(prefix='.extract-', dir=data_dir) as temporary:
            staged = []
            for member_name, name in members.items():
                destination = data_dir / name
                if destination.exists() or destination.is_symlink():
                    verify_file(destination, files[name])
                    continue
                staged_path = Path(temporary) / name
                with archive.open(member_name) as input_stream, staged_path.open('wb') as output:
                    received = 0
                    while block := input_stream.read(CHUNK_SIZE):
                        received += len(block)
                        if received > files[name]['bytes']:
                            raise ValueError(f'ZIP member exceeds the expected size: {name}')
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                verify_file(staged_path, files[name])
                staged.append((staged_path, destination))
            for staged_path, destination in staged:
                if destination.exists() or destination.is_symlink():
                    verify_file(destination, files[destination.name])
                else:
                    os.replace(staged_path, destination)


def download_archive(base_url, info, archive_dir, *, retries=2, timeout=30):
    """Download a ZIP directly or join its verified binary parts in manifest order."""
    name = info['filename']
    if 'parts' not in info:
        return download_file(base_url, name, info, archive_dir, url=info.get('url'),
                             retries=retries, timeout=timeout)
    archive_dir = Path(archive_dir)
    destination = archive_dir / name
    partial = destination.with_name(destination.name + '.partial')

    def remove_parts():
        for part in info['parts']:
            for suffix in ('', '.partial'):
                path = archive_dir / (part['filename'] + suffix)
                if path.is_file() and not path.is_symlink():
                    path.unlink()

    if destination.exists() or destination.is_symlink():
        verify_file(destination, info)
        remove_parts()
        return 'verified'
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f'Expected a regular partial file: {partial}')
    if partial.exists() and partial.stat().st_size == info['bytes']:
        try:
            verify_file(partial, info)
        except ValueError:
            pass
        else:
            os.replace(partial, destination)
            remove_parts()
            return 'joined'
    for part in info['parts']:
        state = download_file(base_url, part['filename'], part, archive_dir,
                              url=part.get('url'), retries=retries, timeout=timeout)
        print(f'{state}: {part["filename"]}', flush=True)
    with partial.open('wb') as output:
        for part in info['parts']:
            with (archive_dir / part['filename']).open('rb') as input_stream:
                while block := input_stream.read(CHUNK_SIZE):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    # Keep the downloaded parts on failure so a retry never requires all
    # transfers again. A complete verified ZIP replaces the combined partial.
    verify_file(partial, info)
    os.replace(partial, destination)
    remove_parts()
    return 'joined'


def download_selected(manifest, sources, data_dir, *, archive_dir=None,
                      verify_only=False, keep_archive=True, retries=2, timeout=30):
    sources = list(dict.fromkeys(sources))
    data_dir = Path(data_dir).expanduser()
    if archive_dir is None:
        archive_dir = Path(os.environ.get('LOGSID_CACHE', '~/.cache/logsid-paper')).expanduser() / 'downloads'
    archive_dir = Path(archive_dir).expanduser()
    names = list(SHARED_FILES)
    for source in sources:
        if source not in SOURCES:
            raise ValueError(f'Unknown dataset source: {source}')
        names.extend((source + '.log.gz', source + '_labels.jsonl.gz'))
    for name in names:
        if name not in manifest['files']:
            raise ValueError(f'Dataset manifest does not contain {name}')
        destination = data_dir / name
        if destination.exists() or destination.is_symlink():
            verify_file(destination, manifest['files'][name])
        elif verify_only:
            raise FileNotFoundError(f'Missing dataset file: {destination}')
    if verify_only:
        for name in names:
            print(f'verified: {name}', flush=True)
        return
    for source in sources:
        if source not in manifest.get('archives', {}):
            raise ValueError(f'Dataset manifest does not contain a ZIP archive for {source}')
    for source in sources:
        info = manifest['archives'][source]
        name = info['filename']
        state = download_archive(manifest['base_url'], info, archive_dir,
                                 retries=retries, timeout=timeout)
        print(f'{state}: {name}', flush=True)
        archive_path = archive_dir / name
        unpack_archive(archive_path, source, manifest['files'], data_dir)
        if not keep_archive:
            archive_path.unlink()
    for name in names:
        print(f'verified: {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--verify-only', action='store_true', help='Check selected local files without downloading')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='Dataset destination directory')
    parser.add_argument('--base-url', help='Override the Zenodo record files URL (also accepts a file:// directory)')
    parser.add_argument('--archive-dir', type=Path, help='ZIP directory (default: $LOGSID_CACHE/downloads or ~/.cache/logsid-paper/downloads)')
    parser.add_argument('--keep-archive', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep verified ZIP archives after extraction (default: enabled)')
    parser.add_argument('--retries', type=int, default=2, help='Retries for interrupted transfers (default: 2)')
    parser.add_argument('--timeout', type=float, default=30, help='Network timeout in seconds (default: 30)')
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0:
        parser.error('--retries must be nonnegative and --timeout must be positive')
    try:
        manifest = load_manifest(ROOT / 'scripts/data_manifest.json', base_url=args.base_url)
        download_selected(manifest, args.sources, args.data_dir, archive_dir=args.archive_dir,
                          verify_only=args.verify_only, keep_archive=args.keep_archive,
                          retries=args.retries, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
