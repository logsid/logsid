import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from logshield import data, prepare
from logshield.io import ORDER_PROTOCOL, build_order, file_sha256, load_order


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.cache = self.root / 'cache'
        self.data.mkdir()
        (self.data / 'taxonomy.csv').write_text('category_id\nACCOUNT\n')
        (self.data / 'statistics.csv').write_text('Dataset,Logs\nApache,3\nBGL,2\n')
        self.write_source('Apache', ['2020-01-02', '2020-01-01', '2020-01-01'])
        self.write_source('BGL', ['2020-01-01', '2020-01-02'])

    def tearDown(self):
        self.temporary.cleanup()

    def write_source(self, source, timestamps):
        with gzip.open(self.data / (source + '.log.gz'), 'wt') as output:
            output.write('raw\n' * len(timestamps))
        with gzip.open(self.data / (source + '_labels.jsonl.gz'), 'wt') as output:
            for index, stamp in enumerate(timestamps, 1):
                output.write(json.dumps({'LineId': index, 'Timestamp': stamp,
                    'Content': '用户 account', 'EventId': 'E1',
                    'Labels': [{'start': 0, 'end': 2, 'category': 'ACCOUNT'}]}) + '\n')

    def test_incremental_ordering_retains_raw_labels_and_stable_ties(self):
        prepare.prepare_data(self.data, ['Apache'], self.cache)
        order_dir = self.cache / 'order'
        np.testing.assert_array_equal(np.load(order_dir / 'Apache_order.npy'), [1, 2, 0])
        before = (order_dir / 'Apache_order.npy').read_bytes()
        prepare.prepare_data(self.data, ['BGL'], self.cache)
        prepare.prepare_data(self.data, ['Apache', 'BGL'], self.cache)
        self.assertEqual((order_dir / 'Apache_order.npy').read_bytes(), before)
        np.testing.assert_array_equal(np.load(order_dir / 'BGL_order.npy'), [0, 1])
        metadata = json.loads((order_dir / 'metadata.json').read_text())
        self.assertEqual(metadata['protocol'], ORDER_PROTOCOL)
        self.assertEqual(set(metadata['sources']), {'Apache', 'BGL'})
        self.assertEqual({key: metadata['sources']['Apache'][key] for key in ('rows', 'history', 'test80')},
                         {'rows': 3, 'history': 0, 'test80': 3})
        self.assertEqual({path.name for path in order_dir.iterdir()},
                         {'metadata.json', 'Apache_order.npy', 'BGL_order.npy'})
        self.assertEqual(metadata['sources']['BGL']['missing_or_unparseable_timestamps'], 0)
        self.assertFalse(metadata['sources']['Apache']['yearless_timestamps'])
        with sqlite3.connect(self.cache / 'log-data/Apache/records.sqlite3') as connection:
            row = json.loads(connection.execute('SELECT payload FROM records WHERE line_id=1').fetchone()[0])
        self.assertEqual(row['Content'], '用户 account')
        self.assertEqual(row['Labels'], [{'start': 0, 'end': 2, 'category': 'ACCOUNT'}])
        with patch.object(data, 'ROOT', self.root), patch.object(data, 'CACHE', self.cache):
            reader = data.LogFile('Apache')
            try:
                records = list(reader.rows())
            finally:
                reader.close()
        self.assertEqual([row['line_id'] for row in records], [2, 3, 1])
        self.assertTrue(all(set(row) == {'rank', 'line_id', 'source', 'text'} for row in records))

    def test_yearless_rollover_keeps_reversals_ties_and_original_timestamps(self):
        timestamps = ['--12-31T23:59:59.000001', '--12-31T23:59:58', '--12-31T23:59:59.000001',
                      '--01-02T00:00:00', '--01-01T00:00:01', '--01-01T00:00:01',
                      '--02-29T12:00:00', '--03-01T00:00:00', '--02-28T12:00:00']
        (self.data / 'statistics.csv').write_text('Dataset,Logs\nApache,9\nBGL,2\n')
        self.write_source('Apache', timestamps)
        originals = {name: file_sha256(self.data / name)
                     for name in ('Apache.log.gz', 'Apache_labels.jsonl.gz')}
        prepare.prepare_data(self.data, ['Apache'], self.cache)
        order, metadata = load_order(self.cache / 'order', 'Apache', expected_rows=9)
        np.testing.assert_array_equal(order, [1, 0, 2, 4, 5, 3, 8, 6, 7])
        stats = metadata['sources']['Apache']
        self.assertTrue(stats['yearless_timestamps'])
        self.assertEqual(stats['relative_year_rollovers'], 1)
        self.assertEqual(stats['adjacent_timestamp_reversals'], 3)
        self.assertIn('not observed calendar years', stats['relative_year_convention'])
        self.assertEqual({name: file_sha256(self.data / name) for name in originals}, originals)
        with sqlite3.connect(self.cache / 'log-data/Apache/records.sqlite3') as connection:
            stored = [json.loads(row[0])['Timestamp'] for row in connection.execute(
                'SELECT payload FROM records ORDER BY line_id')]
        self.assertEqual(stored, timestamps)

    def test_invalid_timestamp_fails_without_installing_order(self):
        for invalid in (None, '', 'not-a-timestamp', '--02-30T12:00:00'):
            with self.subTest(timestamp=invalid):
                prefix = '--01-01T00:00:00' if isinstance(invalid, str) and invalid.startswith('--') else '2020-01-01'
                self.write_source('Apache', [prefix, invalid, prefix])
                order_dir = self.cache / 'order'
                with self.assertRaisesRegex(ValueError, 'unparseable Timestamp at Apache:2'):
                    prepare.prepare_order(self.data, 'Apache', order_dir)
                self.assertFalse((order_dir / 'Apache_order.npy').exists())
                self.assertFalse((order_dir / 'metadata.json').exists())

    def test_full_date_timezone_order_and_mixed_year_formats(self):
        self.write_source('Apache', ['2020-01-01T01:00:00+01:00', '2019-12-31T23:59:59Z',
                                     '2020-01-01T00:00:00Z'])
        stats = build_order(self.data, 'Apache', self.cache / 'absolute')
        np.testing.assert_array_equal(np.load(self.cache / 'absolute/Apache_order.npy'), [1, 0, 2])
        self.assertFalse(stats['yearless_timestamps'])
        self.write_source('Apache', ['2020-01-01', '--01-01T00:00:00', '2020-01-02'])
        with self.assertRaisesRegex(ValueError, 'Mixed full-date and yearless'):
            build_order(self.data, 'Apache', self.cache / 'mixed')
        self.assertFalse((self.cache / 'mixed/Apache_order.npy').exists())

    def test_order_reuse_rejects_duplicate_rows_even_with_updated_checksum(self):
        prepare.prepare_data(self.data, ['Apache'], self.cache)
        order_dir = self.cache / 'order'
        path = order_dir / 'Apache_order.npy'
        np.save(path, np.array([1, 1, 0], dtype=np.uint32), allow_pickle=False)
        manifest = order_dir / 'metadata.json'
        saved = json.loads(manifest.read_text())
        saved['sources']['Apache']['order_sha256'] = file_sha256(path)
        manifest.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            prepare.prepare_order(self.data, 'Apache', order_dir)

    def test_order_reuse_rejects_changed_source_and_incompatible_protocol(self):
        prepare.prepare_data(self.data, ['Apache'], self.cache)
        order_dir = self.cache / 'order'
        self.write_source('Apache', ['2022-01-01'] * 3)
        with self.assertRaisesRegex(ValueError, 'Dataset or taxonomy changed'):
            prepare.prepare_order(self.data, 'Apache', order_dir)
        manifest = order_dir / 'metadata.json'
        saved = json.loads(manifest.read_text())
        saved['protocol'] = 'chronological-history20-test80'
        manifest.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            prepare.prepare_order(self.data, 'Apache', order_dir)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            load_order(order_dir, 'Apache')

    def test_changed_data_cannot_reuse_complete_or_partial_index(self):
        prepare.prepare_data(self.data, ['Apache'], self.cache)
        self.write_source('Apache', ['2021-01-01'] * 3)
        for status in ('complete', 'preparing'):
            manifest = self.cache / 'log-data/Apache/index.json'
            saved = json.loads(manifest.read_text())
            saved['status'] = status
            manifest.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'different data'):
                prepare.prepare_data(self.data, ['Apache'], self.cache)

    def test_sampler_download_is_checked_before_installation(self):
        payload = b'original sampler source'
        manifest_root = self.root / 'repo'
        (manifest_root / 'logshield').mkdir(parents=True)
        manifest = {'commit': prepare.COMMIT, 'source_files': {prepare.NATIVE_FILE: {
            'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://example.invalid/sampler'}}}
        (manifest_root / 'logshield/sampling_source.json').write_text(json.dumps(manifest))
        destination = self.cache / 'upstream' / ('UNLEASH-' + prepare.COMMIT) / prepare.NATIVE_FILE
        with patch.object(prepare, 'ROOT', manifest_root):
            with patch.object(prepare, 'urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    prepare.prepare_sampler(self.cache)
            self.assertFalse(destination.exists())
            with patch.object(prepare, 'urlopen', return_value=io.BytesIO(payload)):
                self.assertEqual(prepare.prepare_sampler(self.cache), destination)
            self.assertEqual(destination.read_bytes(), payload)
            destination.write_bytes(b'changed locally')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare.prepare_sampler(self.cache)

    def test_backbone_download_pins_revision_and_falls_back_to_bin(self):
        snapshot = self.cache / 'model-snapshot'
        snapshot.mkdir(parents=True)
        for name in ('config.json', 'vocab.json', 'merges.txt'):
            (snapshot / name).write_text('{}')
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            if kwargs['allow_patterns'] == ['pytorch_model.bin']:
                (snapshot / 'pytorch_model.bin').write_bytes(b'mocked weights')
            return str(snapshot)

        with patch('huggingface_hub.snapshot_download', side_effect=download):
            self.assertEqual(prepare.prepare_model(self.cache), snapshot)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call['repo_id'], prepare.MODEL_ID)
            self.assertEqual(call['revision'], prepare.MODEL_REVISION)
            self.assertEqual(call['cache_dir'], str(self.cache / 'huggingface'))


if __name__ == '__main__':
    unittest.main()
