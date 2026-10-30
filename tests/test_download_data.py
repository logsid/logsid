import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_data.py'
SPEC = importlib.util.spec_from_file_location('download_data', SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote = self.root / 'release'
        self.remote.mkdir()
        self.data = self.root / 'data'
        self.archives = self.root / 'downloads'
        self.payloads = {'Apache.log.gz': b'original raw logs',
                         'Apache_labels.jsonl.gz': b'original labels',
                         'BGL.log.gz': b'unselected'}
        self.payloads.update({name: ('shared ' + name).encode() for name in download.SHARED_FILES})
        for name, payload in self.payloads.items():
            (self.remote / name).write_bytes(payload)
        self.manifest = {'release': 'dataset-v1', 'base_url': self.remote.as_uri(),
                         'files': {name: self.integrity(payload) for name, payload in self.payloads.items()},
                         'archives': {}}
        self.make_archive()
        (self.root / 'scripts').mkdir()
        self.manifest_path = self.root / 'scripts/data_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def integrity(payload):
        return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def make_archive(self, extras=(), overrides=None):
        path = self.remote / 'Apache.zip'
        replaced = {name.filename if isinstance(name, zipfile.ZipInfo) else name for name, _ in extras}
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name in ('Apache.log.gz', 'Apache_labels.jsonl.gz'):
                payload = (overrides or {}).get(name, self.payloads[name])
                if 'data/' + name not in replaced:
                    archive.writestr('data/' + name, payload)
            for name in download.SHARED_FILES:
                if 'data/' + name not in replaced:
                    archive.writestr('data/' + name, (overrides or {}).get(name, self.payloads[name]))
            for name, payload in extras:
                archive.writestr(name, payload)
        self.manifest['archives']['Apache'] = {'filename': 'Apache.zip', **self.integrity(path.read_bytes())}

    def make_split_archive(self):
        payload = (self.remote / 'Apache.zip').read_bytes()
        width = len(payload) // 3
        pieces = (payload[:width], payload[width:2 * width], payload[2 * width:])
        parts = []
        for index, piece in enumerate(pieces, 1):
            name = f'Apache.zip.{index:03d}'
            (self.remote / name).write_bytes(piece)
            parts.append({'filename': name, **self.integrity(piece)})
        self.manifest['archives']['Apache']['parts'] = parts
        return parts

    def test_split_archive_joins_exact_urls_and_removes_parts(self):
        parts = self.make_split_archive()
        payloads = {}
        for part in parts:
            part['url'] = 'https://example.invalid/draft/files/' + part['filename'] + '/content?token=fixture-preview'
            payloads[part['url']] = (self.remote / part['filename']).read_bytes()
        with patch.object(download, 'urlopen', side_effect=lambda request, **_: Response(payloads[request.full_url])) as opened:
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertEqual([call.args[0].full_url for call in opened.call_args_list], list(payloads))
        self.assertEqual((self.archives / 'Apache.zip').read_bytes(), (self.remote / 'Apache.zip').read_bytes())
        self.assertEqual(list(self.archives.iterdir()), [self.archives / 'Apache.zip'])
        self.assertEqual((self.data / 'Apache_labels.jsonl.gz').read_bytes(), self.payloads['Apache_labels.jsonl.gz'])

    def test_split_archive_missing_part_resumes_existing_verified_parts(self):
        parts = self.make_split_archive()
        missing = self.remote / parts[1]['filename']
        saved = missing.read_bytes()
        missing.unlink()
        with self.assertRaises(RuntimeError):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertTrue((self.archives / parts[0]['filename']).is_file())
        self.assertFalse((self.archives / 'Apache.zip').exists())
        self.assertFalse(self.data.exists())
        missing.write_bytes(saved)
        with patch.object(download, 'urlopen', wraps=download.urlopen) as opened:
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertEqual(len(opened.call_args_list), 2)
        self.assertEqual(list(self.archives.iterdir()), [self.archives / 'Apache.zip'])

    def test_split_archive_wrong_order_fails_whole_hash_then_recovers_without_network(self):
        parts = self.make_split_archive()
        self.manifest['archives']['Apache']['parts'] = list(reversed(parts))
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertFalse((self.archives / 'Apache.zip').exists())
        self.assertFalse(self.data.exists())
        self.assertTrue(all((self.archives / part['filename']).is_file() for part in parts))
        self.manifest['archives']['Apache']['parts'] = parts
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives)
        self.assertEqual((self.archives / 'Apache.zip').read_bytes(), (self.remote / 'Apache.zip').read_bytes())

    def test_split_archive_base_url_override_and_complete_zip_reuse(self):
        parts = self.make_split_archive()
        for part in parts:
            part['url'] = 'https://example.invalid/content?token=fixture-preview'
        self.manifest_path.write_text(json.dumps(self.manifest))
        manifest = download.load_manifest(self.manifest_path, base_url=self.remote.as_uri())
        download.download_selected(manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives,
                                       keep_archive=False)
        self.assertFalse((self.archives / 'Apache.zip').exists())
        self.assertEqual((self.data / 'Apache.log.gz').read_bytes(), self.payloads['Apache.log.gz'])

    def test_split_manifest_rejects_unsafe_duplicate_and_incomplete_parts(self):
        parts = self.make_split_archive()
        for invalid in ([{**parts[0], 'filename': '../outside'}], [parts[0], parts[0]], parts[:-1]):
            self.manifest['archives']['Apache']['parts'] = invalid
            self.manifest_path.write_text(json.dumps(self.manifest))
            with self.assertRaises(ValueError):
                download.load_manifest(self.manifest_path)

    def test_file_release_downloads_selected_pair_and_verifies_existing_files(self):
        manifest = download.load_manifest(self.manifest_path)
        self.data.mkdir()
        (self.data / 'taxonomy.csv').write_bytes(self.payloads['taxonomy.csv'])
        download.download_selected(manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        for name in ('Apache.log.gz', 'Apache_labels.jsonl.gz'):
            self.assertEqual((self.data / name).read_bytes(), self.payloads[name])
        self.assertFalse((self.data / 'BGL.log.gz').exists())
        self.assertEqual(list(self.data.glob('*.partial')), [])
        self.assertTrue((self.archives / 'Apache.zip').is_file())
        for name in download.SHARED_FILES:
            self.assertEqual((self.data / name).read_bytes(), self.payloads[name])
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download_selected(manifest, ['Apache'], self.data, archive_dir=self.archives, verify_only=True)
            download.download_selected(manifest, ['Apache'], self.data, archive_dir=self.archives)

    def test_existing_wrong_bytes_are_preserved(self):
        self.data.mkdir()
        target = self.data / 'Apache.log.gz'
        target.write_bytes(b'keep this existing file')
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives)
        self.assertEqual(target.read_bytes(), b'keep this existing file')

    def test_corrupt_download_never_becomes_final_file(self):
        path = self.remote / 'Apache.zip'
        path.write_bytes(b'x' * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertFalse((self.data / 'Apache.log.gz').exists())
        self.assertFalse((self.archives / 'Apache.zip').exists())
        self.assertFalse((self.archives / 'Apache.zip.partial').exists())

    def test_verify_only_missing_file_never_downloads(self):
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            with self.assertRaises(FileNotFoundError):
                download.download_selected(self.manifest, ['Apache'], self.data,
                                           archive_dir=self.archives, verify_only=True)
        self.assertFalse(self.data.exists())

    def test_verify_only_does_not_need_archive(self):
        self.data.mkdir()
        for name in ('Apache.log.gz', 'Apache_labels.jsonl.gz', *download.SHARED_FILES):
            (self.data / name).write_bytes(self.payloads[name])
        self.manifest['archives'] = {}
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download_selected(self.manifest, ['Apache'], self.data,
                                       archive_dir=self.archives, verify_only=True)
        self.assertFalse(self.archives.exists())

    def test_inner_gzip_hash_is_checked_before_any_final_file_is_created(self):
        self.make_archive(overrides={'Apache_labels.jsonl.gz': b'x' * len(self.payloads['Apache_labels.jsonl.gz'])})
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertFalse((self.data / 'Apache.log.gz').exists())
        self.assertFalse((self.data / 'Apache_labels.jsonl.gz').exists())

    def test_metadata_hash_is_checked_before_any_final_file_is_created(self):
        self.make_archive(overrides={'taxonomy.csv': b'x' * len(self.payloads['taxonomy.csv'])})
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertFalse((self.data / 'Apache.log.gz').exists())
        self.assertFalse((self.data / 'taxonomy.csv').exists())

    def test_existing_wrong_metadata_is_preserved(self):
        self.data.mkdir()
        target = self.data / 'taxonomy.csv'
        target.write_bytes(b'keep custom taxonomy')
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives)
        self.assertEqual(target.read_bytes(), b'keep custom taxonomy')

    def test_command_reads_program_manifest_and_accepts_base_url_override(self):
        self.manifest['base_url'] = 'https://example.invalid/record/files'
        self.manifest['archives']['Apache']['url'] = 'https://example.invalid/draft/files/Apache.zip/content?token=fixture-preview'
        self.manifest_path.write_text(json.dumps(self.manifest))
        argv = ['download_data.py', '--sources', 'Apache', '--base-url', self.remote.as_uri(),
                '--data-dir', str(self.data), '--archive-dir', str(self.archives), '--retries', '0']
        with patch.object(download, 'ROOT', self.root), patch('sys.argv', argv):
            download.main()
        for name in ('Apache.log.gz', 'Apache_labels.jsonl.gz', *download.SHARED_FILES):
            self.assertEqual((self.data / name).read_bytes(), self.payloads[name])

    def test_per_archive_url_keeps_its_exact_preview_query(self):
        url = 'https://example.invalid/api/records/1/draft/files/Apache.zip/content?token=fixture%2Bpreview&mode=read'
        self.manifest['archives']['Apache']['url'] = url
        self.manifest_path.write_text(json.dumps(self.manifest))
        manifest = download.load_manifest(self.manifest_path)
        with patch.object(download, 'urlopen', return_value=Response((self.remote / 'Apache.zip').read_bytes())) as opened:
            download.download_selected(manifest, ['Apache'], self.data, archive_dir=self.archives, retries=0)
        self.assertEqual(opened.call_args.args[0].full_url, url)

    def test_preview_query_does_not_leak_in_command_errors(self):
        secret = 'fixture-private-query-value'
        self.manifest['archives']['Apache']['url'] = 'https://example.invalid/content?token=' + secret
        self.manifest_path.write_text(json.dumps(self.manifest))
        argv = ['download_data.py', '--sources', 'Apache', '--data-dir', str(self.data),
                '--archive-dir', str(self.archives), '--retries', '0']
        for error in (URLError('failed request ?token=' + secret), ValueError('bad URL ?token=' + secret)):
            with self.subTest(error=type(error).__name__):
                stderr = io.StringIO()
                with patch.object(download, 'ROOT', self.root), patch('sys.argv', argv):
                    with patch.object(download, 'urlopen', side_effect=error), patch('sys.stderr', stderr):
                        with self.assertRaises(SystemExit) as stopped:
                            download.main()
                self.assertEqual(stopped.exception.code, 1)
                self.assertNotIn(secret, stderr.getvalue())
                self.assertNotIn('token=', stderr.getvalue())

    def test_per_archive_url_validation_is_redacted(self):
        for url in ('javascript:fixture-secret', 'https://user:fixture-secret@example.invalid/file',
                    'https://example.invalid/file?token=fixture-secret value'):
            self.manifest['archives']['Apache']['url'] = url
            self.manifest_path.write_text(json.dumps(self.manifest))
            with self.assertRaises(ValueError) as caught:
                download.load_manifest(self.manifest_path)
            self.assertNotIn('fixture-secret', str(caught.exception))

    def test_zip_traversal_symlink_and_unknown_members_are_rejected(self):
        symlink = zipfile.ZipInfo('data/Apache.log.gz')
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        shared_symlink = zipfile.ZipInfo('data/taxonomy.csv')
        shared_symlink.create_system = 3
        shared_symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        for member in ('../outside', '/outside', 'data/../../outside', 'data\\outside',
                       'data/BGL.log.gz', 'data/private.txt', symlink, shared_symlink):
            with self.subTest(member=str(member)):
                self.make_archive(extras=[(member, b'outside')])
                with self.assertRaisesRegex(ValueError, 'unsafe|duplicate|unexpected'):
                    download.unpack_archive(self.remote / 'Apache.zip', 'Apache', self.manifest['files'], self.data)
                self.assertFalse((self.data / 'Apache.log.gz').exists())
                self.assertFalse((self.root / 'outside').exists())

    def test_archive_can_be_removed_explicitly_after_success(self):
        download.download_selected(self.manifest, ['Apache'], self.data, archive_dir=self.archives,
                                   keep_archive=False, retries=0)
        self.assertFalse((self.archives / 'Apache.zip').exists())
        self.assertEqual((self.data / 'Apache.log.gz').read_bytes(), self.payloads['Apache.log.gz'])

    def test_range_resume_and_retry_preserve_received_bytes(self):
        payload = self.payloads['Apache.log.gz']
        first = Response(payload[:5])
        remaining = Response(payload[5:], 206, {'Content-Range': f'bytes 5-{len(payload)-1}/{len(payload)}'})
        with patch.object(download, 'urlopen', side_effect=[first, remaining]) as opened:
            with patch.object(download.time, 'sleep'):
                download.download_file('https://example.invalid/release', 'Apache.log.gz',
                                       self.integrity(payload), self.data, retries=1)
        self.assertEqual((self.data / 'Apache.log.gz').read_bytes(), payload)
        self.assertEqual(opened.call_args_list[1].args[0].get_header('Range'), 'bytes=5-')

    def test_server_without_range_restarts_partial(self):
        payload = self.payloads['Apache.log.gz']
        self.data.mkdir()
        (self.data / 'Apache.log.gz.partial').write_bytes(b'bad prefix')
        with patch.object(download, 'urlopen', return_value=Response(payload)):
            download.download_file('https://example.invalid/release', 'Apache.log.gz',
                                   self.integrity(payload), self.data, retries=0)
        self.assertEqual((self.data / 'Apache.log.gz').read_bytes(), payload)

    def test_complete_partial_is_verified_and_promoted_without_network(self):
        payload = self.payloads['Apache.log.gz']
        self.data.mkdir()
        (self.data / 'Apache.log.gz.partial').write_bytes(payload)
        with patch.object(download, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download_file('https://example.invalid/release', 'Apache.log.gz',
                                   self.integrity(payload), self.data)
        self.assertEqual((self.data / 'Apache.log.gz').read_bytes(), payload)

    def test_path_traversal_manifest_is_rejected(self):
        for name in ('../outside', '/outside', 'nested/file', '..\\outside', 'C:outside', '..', 'a\x00b'):
            with self.subTest(name=name):
                self.manifest['files'] = {name: self.integrity(b'fixture')}
                self.manifest_path.write_text(json.dumps(self.manifest))
                with self.assertRaisesRegex(ValueError, 'unsafe filename'):
                    download.load_manifest(self.manifest_path)

    def test_archive_path_traversal_manifest_is_rejected(self):
        self.manifest['archives']['Apache']['filename'] = '../outside.zip'
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'unsafe archive'):
            download.load_manifest(self.manifest_path)


if __name__ == '__main__':
    unittest.main()
