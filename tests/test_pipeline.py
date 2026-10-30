"""Offline integration checks for cache routing, transactional replay and scoring."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from logshield.cache import ParameterCache
from logshield.run import calibrate, evaluate
from logshield.stream import detect


CATEGORY = 'ACCOUNT_RECORD_LICENSE_ID'


def name_span(text):
    start = text.index(' ') + 1
    return [{'start': start, 'end': len(text), 'category': CATEGORY}]


class TextRows:
    """Detector inputs have text and row identity only, never gold labels."""
    source = 'Fixture'

    def __init__(self, root, texts, line_ids=None):
        self.path = root / 'Fixture.log.gz'
        self.texts = texts
        self.line_ids = line_ids or list(range(1, len(texts) + 1))
        self.visited = []

    def __len__(self):
        return len(self.texts)

    def rows(self, start, stop):
        for rank in range(start, stop):
            self.visited.append(rank)
            yield {'rank': rank, 'line_id': self.line_ids[rank], 'source': self.source,
                   'text': self.texts[rank]}


class SelectiveCache:
    references = {}

    def predict(self, text, threshold=None):
        if text.startswith('cache '):
            return {'spans': name_span(text), 'score': 1.0}
        return {'spans': None, 'score': 0.0}

    def examples(self, text):
        return []


class SelectiveSLM:
    def __init__(self):
        self.calls = []

    def predict_batch(self, texts, sources):
        if len(texts) != 1 or sources != ['Fixture']:
            raise AssertionError('Online SLM must receive exactly the current source message')
        text = texts[0]
        self.calls.append(text)
        if text.startswith('small '):
            return [{'spans': name_span(text), 'score': 0.95}]
        return [{'spans': [], 'score': 0.95 if text.startswith('empty ') else 0.1}]


class RepairableLLM:
    repair_calls = 0
    cost = 0.0

    def __init__(self):
        self.calls = 0
        self.requests = []

    def predict(self, text, source, examples, draft):
        self.calls += 1
        self.requests.append((text, source, draft))
        if self.calls == 1:
            return {'spans': None, 'reason': 'simulated_api_failure'}
        return {'spans': name_span(text), 'reason': None}


class InterruptibleSLM:
    def __init__(self):
        self.interrupted = False
        self.calls = []

    def predict_batch(self, texts, sources):
        text = texts[0]
        self.calls.append(text)
        if text == 'small Donna' and not self.interrupted:
            self.interrupted = True
            raise RuntimeError('simulated_interruption')
        return [{'spans': name_span(text), 'score': 1.0}]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='logsid-pipeline-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'predictions.sqlite3'

    def predictions(self):
        with sqlite3.connect(self.output) as connection:
            return list(connection.execute('SELECT rank,line_id,spans,route FROM predictions ORDER BY rank'))

    def test_real_cache_localizes_current_values_and_empty_outputs_after_calibration(self):
        fitting = []
        for name in ('alice', 'beth', 'carol', 'donna'):
            text = f'open /home/{name}/report.log'
            fitting.append({'text': text, 'spans': [
                {'start': 11, 'end': 11 + len(name), 'category': CATEGORY}]})
        for name in ('common', 'shared', 'service', 'public'):
            fitting.append({'text': f'open /srv/{name}/report.log', 'spans': []})
        cache = ParameterCache([CATEGORY]).fit(fitting)
        calibration = [
            {'text': 'open /home/emily/report.log',
             'spans': [{'start': 11, 'end': 16, 'category': CATEGORY}]},
            {'text': 'open /srv/global/report.log', 'spans': []},
        ]
        threshold = calibrate(calibration, [cache.predict(row['text']) for row in calibration])
        self.assertIsNotNone(threshold)
        texts = ['open /home/danielle/report.log', 'open /srv/system/report.log',
                 'open /home/爱丽丝/report.log']
        rows = TextRows(self.root, texts)
        completed = detect(rows, cache, threshold, self.output)
        self.assertEqual(completed['status'], 'complete')
        self.assertEqual(completed['routes'], {'cache_template': 3})
        self.assertEqual([json.loads(row[2]) for row in self.predictions()], [
            [{'start': 11, 'end': 19, 'category': CATEGORY}], [],
            [{'start': 11, 'end': 14, 'category': CATEGORY}],
        ])
        self.assertEqual(completed['queries']['slm'], 0)
        self.assertEqual(completed['queries']['llm'], 0)

    def test_rejected_empty_draft_failure_resume_and_every_test_occurrence_scored(self):
        texts = ['cache Alice', 'small Bob', 'small Bob', 'review Carol', 'review Carol',
                 'empty negative', 'review Dave', 'cache Eve', 'small Bob', 'empty negative']
        line_ids = [10, 1, 4, 2, 9, 5, 3, 6, 8, 7]
        rows = TextRows(self.root, texts, line_ids)
        slm, llm = SelectiveSLM(), RepairableLLM()
        arguments = dict(slm=slm, llm=llm, slm_threshold={'empty': 0.8, 'nonempty': 0.8},
                         checkpoint=2, timing_boundary=2, model_id={'fixture': 1})
        partial = detect(rows, SelectiveCache(), 0.8, self.output, **arguments)
        self.assertEqual(partial['status'], 'stopped_for_inspection')
        self.assertEqual(partial['reason'], 'simulated_api_failure')
        self.assertEqual(partial['rows'], 3)
        self.assertEqual(len(self.predictions()), 3)
        self.assertEqual(llm.requests, [('review Carol', 'Fixture', [])])
        with self.assertRaisesRegex(ValueError, 'Evaluation requires one prediction'):
            evaluate('Fixture', self.output, len(rows), [CATEGORY])

        rows.visited.clear()
        completed = detect(rows, SelectiveCache(), 0.8, self.output, **arguments)
        self.assertEqual(completed['status'], 'complete')
        self.assertEqual(completed['new_rows'], 7)
        self.assertEqual(rows.visited, list(range(3, 10)))
        self.assertEqual(completed['queries']['llm'], 3)
        self.assertEqual(completed['queries']['slm'], 5)
        self.assertEqual(completed['routes'], {'cache_template': 2, 'slm': 2,
                                              'cache_duplicate': 4, 'llm': 2})
        predicted = self.predictions()
        self.assertEqual([row[0] for row in predicted], list(range(10)))
        self.assertEqual([row[1] for row in predicted], line_ids)
        self.assertTrue(all(json.loads(row[2]) is not None for row in predicted))
        self.assertEqual(completed['timing']['prefix_rows'], 2)
        self.assertEqual(completed['timing']['test80_rows'], 8)

        gold_path = self.root / 'log-data/Fixture/records.sqlite3'
        gold_path.parent.mkdir(parents=True)
        with sqlite3.connect(gold_path) as connection:
            connection.execute('CREATE TABLE records(line_id INTEGER PRIMARY KEY,payload TEXT)')
            connection.executemany('INSERT INTO records VALUES (?,?)', (
                (line_id, json.dumps({'Content': text,
                                     'Labels': [] if text.startswith('empty ') else name_span(text)}))
                for line_id, text in zip(line_ids, texts)))
        with mock.patch('logshield.run.CACHE', self.root):
            metrics = evaluate('Fixture', self.output, len(rows), [CATEGORY])
        self.assertEqual(metrics['all']['rows'], 10)
        self.assertEqual(metrics['train20_in_sample']['rows'], 2)
        self.assertEqual(metrics['test80']['rows'], 8)
        self.assertEqual(metrics['test80']['span_micro']['gold'], 6)
        self.assertEqual(metrics['test80']['span_micro']['tp'], 6)
        self.assertEqual(metrics['test80']['unresolved'], 0)
        self.assertEqual(metrics['test80']['span_micro']['f1'], 1.0)

    def test_example_contrast_uses_category_sets_not_span_multiplicity(self):
        fitting = [
            {'text': 'send alice to bob',
             'spans': [{'start': 5, 'end': 10, 'category': CATEGORY}]},
            {'text': 'send carol to donna',
             'spans': [{'start': 5, 'end': 10, 'category': CATEGORY},
                       {'start': 14, 'end': 19, 'category': CATEGORY}]},
        ]
        cache = ParameterCache([CATEGORY]).fit(fitting)
        self.assertEqual(len(cache.references), 1)
        self.assertEqual(len(cache.examples('send alice to bob')), 1)
        cache = ParameterCache([CATEGORY]).fit(fitting + [
            {'text': 'send public to shared', 'spans': []}])
        examples = cache.examples('send alice to bob')
        self.assertEqual(len(examples), 2)
        self.assertTrue(examples[0]['spans'])
        self.assertEqual(examples[1]['spans'], [])

    def test_uncommitted_prediction_rolls_back_and_is_replayed_after_interruption(self):
        rows = TextRows(self.root, ['small Alice', 'small Bob', 'small Carol', 'small Donna'])
        slm = InterruptibleSLM()
        arguments = dict(slm=slm, slm_threshold=0.8, checkpoint=2,
                         timing_boundary=0, model_id={'fixture': 2})
        with self.assertRaisesRegex(RuntimeError, 'simulated_interruption'):
            detect(rows, SelectiveCache(), 0.8, self.output, **arguments)
        self.assertEqual([row[0] for row in self.predictions()], [0, 1])
        with sqlite3.connect(self.output) as connection:
            state = json.loads(connection.execute("SELECT value FROM state WHERE key='progress'").fetchone()[0])
        self.assertEqual(state['cursor'], 2)
        self.assertEqual(state['timing']['interrupted_attempts'], 1)

        rows.visited.clear()
        completed = detect(rows, SelectiveCache(), 0.8, self.output, **arguments)
        self.assertEqual(completed['status'], 'complete')
        self.assertEqual(completed['new_rows'], 2)
        self.assertEqual(rows.visited, [2, 3])
        self.assertEqual([row[0] for row in self.predictions()], [0, 1, 2, 3])
        self.assertEqual(slm.calls.count('small Alice'), 1)
        self.assertEqual(slm.calls.count('small Carol'), 2)
        self.assertEqual(completed['queries']['slm'], 5)
        self.assertEqual(completed['routes'], {'slm': 4})


if __name__ == '__main__':
    unittest.main()
