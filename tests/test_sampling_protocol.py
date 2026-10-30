"""Synthetic checks for content-only historical sampling and role isolation."""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from logshield import data


class SamplingProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='logsid-sampling-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = 'Fixture'
        self.total = 15000
        self.order = np.arange(self.total - 1, -1, -1)
        self.events = []
        self.selected_ids = []
        self.scanned_intervals = []
        self.eligible = True
        database = self.root / 'log-data' / self.source / 'records.sqlite3'
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE records(line_id INTEGER PRIMARY KEY,payload TEXT)')
            connection.executemany('INSERT INTO records VALUES (?,?)', (
                (line_id, json.dumps({'Content': self.text(line_id), 'Labels': self.spans(line_id)}))
                for line_id in range(1, self.total + 1)))
        owner = self

        class ContentOnlyLogs:
            def __init__(self, source):
                owner.assertEqual(source, owner.source)
                self.order = owner.order

            def __len__(self):
                return owner.total

            def rows(self, first, stop):
                owner.scanned_intervals.append((first, stop))
                for rank in range(first, stop):
                    line_id = int(self.order[rank]) + 1
                    yield {'source': owner.source, 'rank': rank, 'line_id': line_id,
                           'text': owner.text(line_id)}

            def close(self):
                pass

        def native_sampling(texts, placeholders, shots):
            self.assertEqual(texts, placeholders)
            self.assertEqual(shots, [2048])
            self.assertTrue(all(isinstance(text, str) for text in texts))
            self.events.append('selection')
            # Deliberately return reverse order and include the end of history.
            chosen = list(reversed(texts[-2048:]))
            return {2048: [(text, text) for text in chosen]}

        sampler = {'entropy': lambda text: 1.0,
                   'clean': lambda text: (text if self.eligible else '', None),
                   'sampling': native_sampling}
        original_connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            self.events.append('gold-open')
            connection = original_connect(*args, **kwargs)

            def trace(sql):
                if sql.startswith('SELECT line_id,'):
                    self.selected_ids.extend(map(int, sql.rsplit('(', 1)[1].rstrip(')').split(',')))

            connection.set_trace_callback(trace)
            return connection

        for patcher in (
            mock.patch.object(data, 'CACHE', self.root),
            mock.patch.object(data, 'LogFile', ContentOnlyLogs),
            mock.patch('logshield.sampling.native_source', return_value=(Path('unused'), 'sha', 'commit')),
            mock.patch('logshield.sampling.load_native_sampler', return_value=sampler),
            mock.patch('logshield.data.sqlite3.connect', side_effect=tracked_connect),
            mock.patch.dict(os.environ, {'PYTHONHASHSEED': '42'}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def text(line_id):
        return f'Log λ Alice account {line_id}'

    @staticmethod
    def spans(line_id):
        return [] if line_id % 2 else [
            {'start': 6, 'end': 11, 'category': 'PERSONAL_NAME'},
            {'start': 6, 'end': 9, 'category': 'ACCOUNT_RECORD_LICENSE_ID'}]

    def sample(self, role, directory='samples'):
        return data.records(self.root / directory, self.source, role)

    def test_one_shared_entropy_selection_before_gold_and_disjoint_roles(self):
        # Request calibration first to ensure all roles use the same procedure.
        selected = {role: self.sample(role) for role in ('calibration', 'fit', 'validation')}
        self.assertEqual({role: len(rows) for role, rows in selected.items()}, data.SAMPLE_SIZES)
        self.assertEqual(self.events.count('selection'), 1)
        self.assertEqual(self.events[0], 'selection')
        self.assertEqual(self.scanned_intervals, [(0, self.total // 5)])
        ids = {role: {row['line_id'] for row in rows} for role, rows in selected.items()}
        self.assertEqual(len(set.union(*ids.values())), 2048)
        self.assertFalse(ids['fit'] & ids['validation'])
        self.assertFalse(ids['fit'] & ids['calibration'])
        self.assertFalse(ids['validation'] & ids['calibration'])
        self.assertEqual(set(self.selected_ids), set.union(*ids.values()))
        self.assertEqual(len(self.selected_ids), 2048)
        # Fit examples can come from the last historical 2%, unlike the old code.
        self.assertTrue(any(row['rank'] >= 2700 for row in selected['fit']))
        for records in selected.values():
            for row in records:
                self.assertLess(row['rank'], self.total // 5)
                self.assertEqual(row['line_id'], int(self.order[row['rank']]) + 1)
                self.assertEqual(row['text'], self.text(row['line_id']))
                self.assertEqual(row['spans'], self.spans(row['line_id']))
        shared = json.loads((self.root / 'samples/Fixture.selection.json').read_text())
        self.assertEqual(shared['labels_inspected'], 0)
        self.assertTrue(all('spans' not in row for row in shared['records']))

    def test_repeatable_assignment_and_reuse_without_resampling_or_label_reads(self):
        first = self.sample('fit')
        self.events.clear()
        self.assertEqual(self.sample('fit'), first)
        self.assertEqual(self.events, [])
        self.assertEqual(self.sample('fit', 'independent'), first)

    def test_legacy_protocol_rejected_without_new_selection(self):
        directory = self.root / 'samples'
        directory.mkdir()
        (directory / 'Fixture.fit.json').write_text('[]')
        (directory / 'Fixture.fit.sampling.json').write_text(json.dumps({'version': 3}))
        with self.assertRaisesRegex(ValueError, 'different protocol'):
            self.sample('calibration')
        self.assertEqual(self.events, [])

    def test_insufficient_eligible_history_fails_before_gold_without_test_fallback(self):
        self.eligible = False
        with self.assertRaisesRegex(ValueError, 'requires 2048 eligible unique historical texts'):
            self.sample('validation')
        self.assertEqual(self.events, [])
        self.assertEqual(self.scanned_intervals, [(0, self.total // 5)])

    def test_changed_role_records_are_rejected(self):
        self.sample('fit')
        path = self.root / 'samples/Fixture.fit.json'
        changed = json.loads(path.read_text())
        changed[0]['spans'] = []
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'differ from their shared selection or metadata'):
            self.sample('fit')

    def test_shared_selection_missing_is_not_silently_rebuilt(self):
        self.sample('fit')
        (self.root / 'samples/Fixture.selection.json').unlink()
        self.events.clear()
        with self.assertRaisesRegex(ValueError, 'Shared historical selection is missing'):
            self.sample('validation')
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
