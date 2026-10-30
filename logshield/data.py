"""Read prepared log files in stable time order; labels stay out of inference."""
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from .io import DEFAULT_CACHE, atomic_json, file_sha256, load_order

ROOT = Path(__file__).resolve().parents[1]
CACHE = DEFAULT_CACHE
SOURCES = ('Apache', 'BGL', 'OpenSSH', 'HealthApp', 'AIT-mail', 'Zanbil', 'Mac', 'Thunderbird')
SAMPLING_PROTOCOL = 'history20-entropy2048-partition1536-256-256-v4'
SAMPLE_SIZES = {'fit': 1536, 'validation': 256, 'calibration': 256}


def categories():
    with (ROOT / 'data/taxonomy.csv').open(encoding='utf-8-sig', newline='') as f:
        return [r['category_id'] for r in csv.DictReader(f)]


def records(directory, source, role):
    """Entropy-select 2,048 historical texts, then partition without labels.

    The paper specifies one historical sample and three disjoint roles. Its
    partition seed is not specified; this implementation uses seed 42 on the
    chronologically sorted selection, independently of the native sampler's
    return order. Only the requested role's labels are read afterwards.
    """
    import contextlib
    import io
    import os
    import time

    if role not in SAMPLE_SIZES:
        raise ValueError('Unknown sample role')
    directory = Path(directory)
    limit = SAMPLE_SIZES[role]
    total = sum(SAMPLE_SIZES.values())
    path = directory / f'{source}.{role}.json'
    metadata = path.with_suffix('.sampling.json')
    shared_path = directory / f'{source}.selection.json'
    began = time.perf_counter()
    rows = LogFile(source)
    try:
        train = len(rows) // 5
        identity = {'protocol': SAMPLING_PROTOCOL, 'version': 4, 'source': source,
                    'method': 'unleash_content_only', 'rank_start': 0,
                    'rank_stop': train, 'pool_rows': train, 'requested_rows': total,
                    'partition_seed': 42, 'partition_sizes': SAMPLE_SIZES}
        # Do not silently mix the former 16%/2%/2% selection with this protocol.
        for name in SAMPLE_SIZES:
            old_path = directory / f'{source}.{name}.json'
            old_metadata = old_path.with_suffix('.sampling.json')
            if old_path.exists() != old_metadata.exists():
                raise ValueError('Orphan sample file or metadata; use a new data directory')
            if old_metadata.exists():
                saved = json.loads(old_metadata.read_text())
                if saved.get('protocol') != SAMPLING_PROTOCOL or saved.get('version') != 4:
                    raise ValueError('Existing samples use a different protocol; use a new data directory')
                if not shared_path.exists():
                    raise ValueError('Shared historical selection is missing; use a new data directory')
        if shared_path.exists():
            shared = json.loads(shared_path.read_text())
            if any(shared.get(key) != value for key, value in identity.items()):
                raise ValueError('Existing selection uses a different protocol; use a new data directory')
            selected_all = shared['records']
        else:
            if os.environ.get('PYTHONHASHSEED') != '42':
                raise ValueError('Start Python with PYTHONHASHSEED=42 for native sampling')
            from .sampling import native_source, load_native_sampler
            from functools import lru_cache
            unique = {}
            for row in rows.rows(0, train):
                unique.setdefault(row['text'], row)
            native_path, native_sha, commit = native_source()
            sampler = load_native_sampler(native_path, native_sha)
            sampler['entropy'] = lru_cache(maxsize=None)(sampler['entropy'])
            texts = list(unique)
            eligible = sum(bool(sampler['clean'](text)[0].split()) for text in texts)
            if eligible < total:
                raise ValueError(f'{source}: requires {total} eligible unique historical texts; '
                                 f'found {eligible}; no supplementation from test data')
            with contextlib.redirect_stdout(io.StringIO()):
                chosen = sampler['sampling'](texts, texts, shots=[total])[total]
            if (len(chosen) != total or len({text for text, _ in chosen}) != total
                    or any(text != placeholder or text not in unique for text, placeholder in chosen)):
                raise ValueError('Native sampler returned invalid Content values')
            selected_all = sorted((dict(unique[text]) for text, _ in chosen), key=lambda row: row['rank'])
            shared = {**identity, 'records': selected_all,
                      'candidate_unique_contents': len(unique), 'eligible_unique_contents': eligible,
                      'pythonhashseed': '42', 'native_commit': commit,
                      'native_sampling_sha256': native_sha,
                      'labels_inspected': 0, 'selection_seconds': time.perf_counter() - began}
            atomic_json(shared_path, shared)
        if (len(selected_all) != total
                or len({row['text'] for row in selected_all}) != total
                or len({row['rank'] for row in selected_all}) != total
                or [row['rank'] for row in selected_all] != sorted(row['rank'] for row in selected_all)
                or any(set(row) != {'source', 'rank', 'line_id', 'text'}
                       or row['source'] != source or type(row['rank']) is not int
                       or not 0 <= row['rank'] < train
                       or row['line_id'] != int(rows.order[row['rank']]) + 1
                       for row in selected_all)):
            raise ValueError('Invalid shared content-only historical selection')
        permutation = np.random.default_rng(42).permutation(total)
        roles = list(SAMPLE_SIZES)
        first = sum(SAMPLE_SIZES[name] for name in roles[:roles.index(role)])
        selected = sorted((dict(selected_all[int(i)]) for i in permutation[first:first + limit]),
                          key=lambda row: row['rank'])
        shared_sha = file_sha256(shared_path)
        if path.exists():
            saved = json.loads(metadata.read_text())
            if (saved.get('shared_selection_sha256') != shared_sha
                    or saved.get('source') != source or saved.get('role') != role
                    or saved.get('requested_rows') != limit or saved.get('selected_rows') != limit
                    or saved.get('records_sha256') != file_sha256(path)):
                raise ValueError('Existing samples differ from their shared selection or metadata')
            existing = json.loads(path.read_text())
            if [{key: row[key] for key in ('source', 'rank', 'line_id', 'text')}
                    for row in existing] != selected:
                raise ValueError('Existing samples differ from the deterministic role assignment')
            return existing
    finally:
        rows.close()
    selection_seconds = time.perf_counter() - began
    ids = [row['line_id'] for row in selected]
    gold = sqlite3.connect((CACHE / 'log-data' / source / 'records.sqlite3').as_uri() + '?mode=ro', uri=True)
    try:
        found = {}
        for offset in range(0, len(ids), 512):
            batch = ids[offset:offset + 512]
            marks = ','.join('?' for _ in batch)
            found.update((line_id, (content, json.loads(spans))) for line_id, content, spans in gold.execute(
                f"SELECT line_id,json_extract(payload,'$.Content'),json_extract(payload,'$.Labels') "
                f"FROM records WHERE line_id IN ({marks})", batch))
        for row in selected:
            content, spans = found[row['line_id']]
            if row['text'] != content:
                raise ValueError('Selected Content differs from the prepared gold index')
            row['spans'] = spans
    finally:
        gold.close()
    sampling = {**{key: value for key, value in shared.items() if key != 'records'}, 'role': role,
                'requested_rows': limit, 'selected_rows': len(selected),
                'labels_inspected': len(selected), 'candidate_pool_labels_inspected': len(selected),
                'selection_seconds': selection_seconds, 'sampling_seconds': time.perf_counter() - began,
                'development_and_calibration_shared': False,
                'shared_selection_file': shared_path.name, 'shared_selection_sha256': shared_sha}
    atomic_json(path, selected)
    sampling['records_sha256'] = file_sha256(path)
    atomic_json(metadata, sampling)
    return selected


class LogFile:
    """Reuse the prepared Content index and chronological permutation.

    Buffered rows contain raw text only; the detector receives one at a time.
    No EventId or Labels column is selected by this class.
    """
    def __init__(self, source, input_file=None):
        self.source = source
        self.path = Path(input_file or ROOT / 'data' / f'{source}.log.gz').resolve()
        if self.path != (ROOT / 'data' / f'{source}.log.gz').resolve():
            raise ValueError('Input file must be the published source log')
        info = json.loads((CACHE / 'log-data' / source / 'index.json').read_text())
        if info.get('status') != 'complete' or info.get('source') != source:
            raise ValueError('The log Content index is incomplete or belongs to another source')
        self.size = info['rows']
        self.order, order_meta = load_order(CACHE / 'order', source, expected_rows=self.size)
        if (order_meta['data_fingerprint'].get(source + '_labels.jsonl.gz') != info['source_file_sha256']
                or order_meta['data_fingerprint'].get('taxonomy.csv') != info['taxonomy_sha256']):
            raise ValueError('Content index and chronological order refer to different data')
        database = CACHE / 'log-data' / source / 'records.sqlite3'
        self.db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        self.column = "json_extract(payload,'$.Content')"

    def __len__(self):
        return self.size

    def rows(self, start=0, stop=None):
        stop = self.size if stop is None else min(stop, self.size)
        for first in range(start, stop, 512):
            ids = [int(i) + 1 for i in self.order[first:min(first + 512, stop)]]
            marks = ','.join('?' for _ in ids)
            found = dict(self.db.execute(f'SELECT line_id,{self.column} FROM records WHERE line_id IN ({marks})', ids))
            for rank, line_id in enumerate(ids, first):
                yield {'rank': rank, 'line_id': line_id, 'source': self.source, 'text': found[line_id]}

    def close(self):
        self.db.close()
