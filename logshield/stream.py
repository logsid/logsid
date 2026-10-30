"""One-log-at-a-time detection with a transactional cursor and bounded cache."""
import json
import pickle
import sqlite3
import time
from collections import Counter, OrderedDict
from pathlib import Path


def allowed(result, threshold):
    spans = result.get('spans')
    if isinstance(threshold, dict):
        threshold = threshold['empty' if not spans else 'nonempty']
    return spans is not None and threshold is not None and result['score'] >= threshold


def span_set(spans):
    return {(s['start'], s['end'], s['category']) for s in spans or []}


def detect(rows, cache, threshold, path, *, llm=None, slm=None,
           slm_threshold=None, stop=None, checkpoint=8192, model_id=None, update=None,
           max_seconds=None, timing_boundary=None):
    if type(checkpoint) is not int or checkpoint < 1:
        raise ValueError('checkpoint must be a positive integer')
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be positive or None')
    if timing_boundary is not None and (type(timing_boundary) is not int
                                       or not 0 <= timing_boundary <= len(rows)):
        raise ValueError('timing_boundary must be a rank between zero and len(rows)')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, isolation_level=None)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(rank INTEGER PRIMARY KEY,line_id INTEGER,spans TEXT,route TEXT,reason TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS experts(text TEXT PRIMARY KEY,spans TEXT,position INTEGER,route TEXT)')
    config = {'input': str(rows.path), 'source': rows.source, 'rows': len(rows), 'model_id': model_id}
    if timing_boundary is not None:
        config['timing_boundary'] = timing_boundary
    prior = db.execute("SELECT value FROM state WHERE key='input'").fetchone()
    if prior and json.loads(prior[0]) != config:
        db.close()
        raise ValueError('Use a new result directory for another input')
    db.execute("INSERT OR IGNORE INTO state VALUES ('input',?)", (json.dumps(config),))
    saved_cache = db.execute("SELECT value FROM state WHERE key='cache'").fetchone()
    if saved_cache:
        cache, threshold, *saved_slm = pickle.loads(saved_cache[0])
        if saved_slm:
            slm_threshold = saved_slm[0]
    cache_dirty = False
    saved = db.execute("SELECT value FROM state WHERE key='progress'").fetchone()
    state = json.loads(saved[0]) if saved else {'cursor': 0, 'routes': {}, 'seconds': 0.0}
    initial, routes, started = state['cursor'], Counter(state['routes']), time.perf_counter()
    previous_seconds = state['seconds']
    timing = dict(state.get('timing', {})) if timing_boundary is not None else None
    if timing is not None:
        if initial and not timing:
            db.close()
            raise ValueError('Cannot infer a phase boundary from an untimed existing cursor')
        timing.setdefault('prefix_seconds', 0.0)
        timing.setdefault('test80_seconds', 0.0)
        timing.setdefault('interrupted_attempts', 0)
        timing.setdefault('uncommitted_attempt_seconds', 0.0)
        timing.setdefault('boundary_native_elapsed_seconds', 0.0 if timing_boundary == 0 else None)
        phase = 'prefix' if initial < timing_boundary else 'test80'
        phase_started = started

    def timed_progress(target, now):
        target['seconds'] = previous_seconds + now - started
        if timing is not None:
            snapshot = dict(timing)
            snapshot[phase + '_seconds'] += now - phase_started
            snapshot.update(boundary_rank=timing_boundary, full_seconds=target['seconds'],
                            prefix_rows=min(target['cursor'], timing_boundary),
                            test80_rows=max(0, target['cursor'] - timing_boundary),
                            boundary_committed=target['cursor'] >= timing_boundary,
                            scope='Native active detect wall time, including setup/update and checkpoint I/O; excludes downtime. Failed attempted work is included. The boundary row is prefix; the subsequent checkpoint I/O is test80.')
            if target['cursor'] < timing_boundary:
                snapshot['boundary_native_elapsed_seconds'] = None
            target['timing'] = snapshot
    queries = Counter(state.get('queries', {}))
    stage_seconds = Counter(state.get('stage_seconds', {}))
    for name in ('slm', 'slm_calibration', 'llm'):
        queries[name] += 0
    queries['llm'] = llm.calls if llm else 0
    queries['llm_repair'] = llm.repair_calls if llm else 0
    db.execute('DELETE FROM experts WHERE text NOT IN (SELECT text FROM experts ORDER BY position DESC LIMIT 4096)')
    experts = OrderedDict((text, json.loads(spans)) for text, spans in db.execute(
        'SELECT text,spans FROM experts ORDER BY position'))
    hot = OrderedDict((row['text'], row['spans']) for group in cache.references.values()
                      for row in group if row['reference'])
    while len(hot) > 4096:
        hot.popitem(last=False)
    hot_sources = {text: 'gold' for text in hot}
    saved_hot = db.execute("SELECT value FROM state WHERE key='hot'").fetchone()
    if saved_hot:
        restored = pickle.loads(saved_hot[0])
        if isinstance(restored, tuple) and len(restored) == 2:
            hot, hot_sources = restored
        # Legacy hot entries have no provenance; retain only trusted fit/expert results.
    examples = []
    for text, spans in db.execute('SELECT text,spans FROM experts ORDER BY position DESC LIMIT 2'):
        examples.append({'text': text, 'spans': json.loads(spans)})
    target = len(rows) if stop is None else min(len(rows), stop)
    status, reason, in_batch, batch_started = 'partial', None, 0, time.perf_counter()
    state.update(status=status, reason=None, blocked_line_id=None)

    def decision():
        return (getattr(cache, 'decision_version', 0),
                threshold.copy() if isinstance(threshold, dict) else threshold)

    def invalidate(previous):
        if decision() != previous:
            for text in [text for text, source in hot_sources.items() if source == 'template']:
                hot.pop(text, None)
                hot_sources.pop(text)

    def commit():
        nonlocal in_batch, batch_started, cache_dirty
        if cache_dirty:
            db.execute("INSERT OR REPLACE INTO state VALUES ('cache',?)", (pickle.dumps((cache, threshold, slm_threshold)),))
        db.execute("INSERT OR REPLACE INTO state VALUES ('hot',?)", (pickle.dumps((hot, hot_sources)),))
        state.update(routes=dict(routes), queries=dict(queries), stage_seconds=dict(stage_seconds))
        timed_progress(state, time.perf_counter())
        db.execute("INSERT OR REPLACE INTO state VALUES ('progress',?)", (json.dumps(state),))
        db.commit()
        in_batch, batch_started, cache_dirty = 0, time.perf_counter(), False

    try:
        db.execute('BEGIN')
        if update is not None:
            previous_slm_rows = slm.prediction_rows if slm else 0
            previous_decision = decision()
            cache, threshold, slm_threshold = update(cache)
            invalidate(previous_decision)
            queries['slm_calibration'] += (slm.prediction_rows if slm else 0) - previous_slm_rows
            cache_dirty = True
        for row in rows.rows(state['cursor'], target):
            if row['rank'] != state['cursor']:
                raise ValueError('Rows must arrive in contiguous chronological rank order')
            if max_seconds is not None and previous_seconds + time.perf_counter() - started >= max_seconds:
                status, reason = 'stopped_for_inspection', 'time_limit'
                state.update(status=status, reason=reason, blocked_line_id=row['line_id'])
                commit()
                break
            text = row['text']
            spans, route, reason, draft = None, 'unresolved', None, None
            hot_source = None
            if text in experts or text in hot:
                spans, route = experts[text] if text in experts else hot[text], 'cache_duplicate'
                hot_source = 'expert' if text in experts else hot_sources[text]
            else:
                cached = cache.predict(text, threshold)
                if allowed(cached, threshold):
                    spans, route = cached['spans'], 'cache_template'
                    hot_source = 'template'
                elif slm is not None and slm_threshold is not None:
                    began = time.perf_counter()
                    candidate = slm.predict_batch([text], sources=[rows.source])[0]
                    stage_seconds['slm'] += time.perf_counter() - began
                    queries['slm'] += 1
                    draft = candidate['spans']
                    if allowed(candidate, slm_threshold):
                        spans, route = candidate['spans'], 'slm'
                        hot_source = 'expert'
                if spans is None and llm is not None:
                    began = time.perf_counter()
                    answer = llm.predict(text, rows.source, cache.examples(text) or examples, draft=draft)
                    stage_seconds['llm'] += time.perf_counter() - began
                    spans, route, reason = answer['spans'], 'llm', answer['reason']
                    hot_source = 'expert'
                    queries['llm'] = llm.calls
                    queries['llm_repair'] = llm.repair_calls
            if spans is None:
                reason = reason or 'no_accepted_prediction'
                status = 'stopped_for_inspection'
                state.update(status=status, reason=reason, blocked_line_id=row['line_id'])
                commit()
                break
            db.execute('INSERT INTO predictions VALUES (?,?,?,?,?)',
                       (row['rank'], row['line_id'], json.dumps(spans, ensure_ascii=False), route, reason))
            routes[route] += 1
            state['cursor'] = row['rank'] + 1
            if route in ('llm', 'slm') and update is not None:
                began = time.perf_counter()
                previous_slm_rows = slm.prediction_rows if slm else 0
                previous_decision = decision()
                cache, threshold, slm_threshold = update(cache, text, spans, allow_refit=route == 'llm')
                invalidate(previous_decision)
                queries['slm_calibration'] += (slm.prediction_rows if slm else 0) - previous_slm_rows
                stage_seconds['cache_update'] += time.perf_counter() - began
                cache_dirty = True
            if spans is not None:
                hot[text] = spans
                hot.move_to_end(text)
                hot_sources[text] = hot_source
                if len(hot) > 4096:
                    evicted, _ = hot.popitem(last=False)
                    hot_sources.pop(evicted)
                if route in ('llm', 'slm'):
                    db.execute('INSERT OR REPLACE INTO experts VALUES (?,?,?,?)',
                               (text, json.dumps(spans), row['rank'], route))
                    experts[text] = spans
                    experts.move_to_end(text)
                    if len(experts) > 4096:
                        evicted, _ = experts.popitem(last=False)
                        db.execute('DELETE FROM experts WHERE text=?', (evicted,))
                    examples = [{'text': text, 'spans': spans}] + examples[:1]
            in_batch += 1
            boundary_now = timing is not None and state['cursor'] == timing_boundary and phase == 'prefix'
            if boundary_now:
                now = time.perf_counter()
                timing['prefix_seconds'] += now - phase_started
                phase, phase_started = 'test80', now
                if timing['boundary_native_elapsed_seconds'] is None:
                    timing['boundary_native_elapsed_seconds'] = previous_seconds + now - started
            if boundary_now or route == 'llm' or in_batch >= checkpoint or time.perf_counter() - batch_started >= 60:
                commit()
                db.execute('BEGIN')
        if db.in_transaction:
            commit()
        if state['cursor'] == len(rows) and status != 'stopped_for_inspection':
            status, reason = 'complete', None
        state.update(status=status)
        timed_progress(state, time.perf_counter())
        db.execute("INSERT OR REPLACE INTO state VALUES ('progress',?)", (json.dumps(state),))
    except BaseException as error:
        if db.in_transaction:
            db.rollback()
        saved = db.execute("SELECT value FROM state WHERE key='progress'").fetchone()
        committed = json.loads(saved[0]) if saved else {'cursor': initial, 'routes': dict(state['routes'])}
        if llm:
            queries['llm'], queries['llm_repair'] = llm.calls, llm.repair_calls
        committed.update(status='stopped_for_inspection', reason=type(error).__name__,
                         seconds=previous_seconds + time.perf_counter() - started,
                         queries=dict(queries), stage_seconds=dict(stage_seconds))
        if timing is not None:
            old = committed.get('timing', {})
            now = time.perf_counter()
            timing['interrupted_attempts'] += 1
            timing['uncommitted_attempt_seconds'] += max(0.0, previous_seconds + now - started
                                                         - old.get('full_seconds', previous_seconds))
            timed_progress(committed, now)
        db.execute("INSERT OR REPLACE INTO state VALUES ('progress',?)", (json.dumps(committed),))
        raise
    finally:
        db.close()
    result = {'status': status, 'reason': reason, 'rows': state['cursor'], 'total_rows': len(rows),
            'new_rows': state['cursor'] - initial, 'routes': dict(routes), 'queries': dict(queries),
            'seconds': state['seconds'], 'stage_seconds': dict(stage_seconds),
            'cache_updates': dict(getattr(cache, 'updates', {})),
            'this_run_seconds': time.perf_counter() - started, 'threshold': threshold,
            'slm_threshold': slm_threshold,
            'llm_http_calls': llm.calls if llm else 0, 'llm_known_cost': llm.cost if llm else 0.0}
    if timing is not None:
        result['timing'] = state['timing']
    return result
