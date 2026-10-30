"""Fit, calibrate, replay and score through one small pipeline."""
import argparse
import getpass
import json
import os
import pickle
import sqlite3
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import groupby
from multiprocessing import get_context
from pathlib import Path

import joblib

from .metrics import SpanMetrics
from .cache import ParameterCache, _case
from .data import CACHE, SOURCES, LogFile, categories, records
from .stream import allowed, detect, span_set
from .train import source_metadata, train_source


ROUTING_VERSION = 24
CALIBRATION_PROTOCOL = 'entropy2048-initial-cache-residual-fixed-slm-v24'


def metrics(rows, predictions, names):
    meter, unresolved = SpanMetrics(names), 0
    for row, candidate in zip(rows, predictions):
        spans = candidate['spans']
        meter.add(row['text'], row['spans'], spans or [])
        if spans is None:
            unresolved += 1
            if not row['spans']:
                meter.logs['exact_span_set'] -= 1
                meter.logs['binary_tn'] -= 1
    return {**meter.to_dict(), 'unresolved': unresolved}


def calibrate(rows, predictions, *, confirmed_cache=False, by_output=False):
    """Accept the most rows with calibration micro recall >= .90 and F1 >= .85.

    Scores rank complete outputs. Quality is measured on accepted calibration
    rows, including empty outputs; it does not guarantee future correctness.
    """
    candidates = sorted((p['score'], i) for i, p in enumerate(predictions) if p['spans'] is not None)
    if by_output:
        curves = {}
        for empty in (False, True):
            curve, tp, fp, fn, count = [(0, 0, 0, 0, None)], 0, 0, 0, 0
            ordered = [item for item in reversed(candidates)
                       if (not predictions[item[1]]['spans']) == empty]
            for score, group in groupby(ordered, key=lambda item: item[0]):
                for _, i in group:
                    expected, predicted = span_set(rows[i]['spans']), span_set(predictions[i]['spans'])
                    tp += len(expected & predicted)
                    fp += len(predicted - expected)
                    fn += len(expected - predicted)
                    count += 1
                curve.append((count, tp, fp, fn, float(score)))
            curves[empty] = curve
        negative_fn = [point[3] for point in curves[True]]
        best = (0, None, None)
        for count, tp, fp, fn, score in curves[False]:
            # Recall >= 9/10 and F1 >= 17/20, measured jointly on accepted rows.
            budget = min(tp // 9 - fn, (6 * tp) // 17 - fp - fn)
            if budget < 0:
                continue
            negative = curves[True][bisect_right(negative_fn, budget) - 1]
            if count + negative[0] > best[0]:
                best = count + negative[0], score, negative[4]
        return {'nonempty': best[1], 'empty': best[2]} if best[0] else None
    # With no covered holdout structure, reuse only previously confirmed cases.
    # Tree leaf scores are strictly below 1; this does not accept new mappings.
    threshold, tp, fp, fn = (1.0 if confirmed_cache and not candidates else None), 0, 0, 0
    for score, group in groupby(reversed(candidates), key=lambda item: item[0]):
        for _, i in group:
            expected, predicted = span_set(rows[i]['spans']), span_set(predictions[i]['spans'])
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        if recall >= .90 and f1 >= .85:
            threshold = float(score)
    return threshold


def calibrate_models(cache, rows, slm=None, source=None):
    """Fit fixed SLM thresholds on the initial cache's rejected calibration rows."""
    predictions = [cache.predict(r['text']) for r in rows]
    threshold = calibrate(rows, predictions, confirmed_cache=True)
    residual = [row for row, prediction in zip(rows, predictions) if not allowed(prediction, threshold)]
    slm_threshold = None
    if slm is not None and residual:
        slm_threshold = calibrate(residual, slm.predict_batch(
            [row['text'] for row in residual], sources=[source] * len(residual)), by_output=True)
    return threshold, slm_threshold


def evaluate(source, database, size, names):
    """Score complete streams offline; unresolved outputs count as omissions."""
    predicted = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    count, first, last, distinct = predicted.execute(
        'SELECT COUNT(*),MIN(rank),MAX(rank),COUNT(DISTINCT line_id) FROM predictions').fetchone()
    if (count != size or distinct != size
            or (size and (first != 0 or last != size - 1))):
        predicted.close()
        raise ValueError('Evaluation requires one prediction for every original message; '
                         'resume the incomplete stream before reporting full-source metrics')
    gold = sqlite3.connect((CACHE / 'log-data' / source / 'records.sqlite3').as_uri() + '?mode=ro', uri=True)
    meters = {key: SpanMetrics(names) for key in ('all', 'train20_in_sample', 'test80')}
    unresolved = Counter()
    cursor = predicted.execute('SELECT rank,line_id,spans FROM predictions ORDER BY rank')
    while batch := cursor.fetchmany(512):
        ids = [row[1] for row in batch]
        marks = ','.join('?' for _ in ids)
        references = dict(gold.execute(f'SELECT line_id,payload FROM records WHERE line_id IN ({marks})', ids))
        for rank, line_id, payload in batch:
            reference, spans = json.loads(references[line_id]), json.loads(payload)
            key = 'train20_in_sample' if rank < size // 5 else 'test80'
            meters[key].add(reference['Content'], reference['Labels'], spans or [])
            if spans is None:
                unresolved[key] += 1
                if not reference['Labels']:
                    meters[key].logs['exact_span_set'] -= 1
                    meters[key].logs['binary_tn'] -= 1
    for key in ('train20_in_sample', 'test80'):
        meters['all'].merge(meters[key])
        unresolved['all'] += unresolved[key]
    gold.close()
    predicted.close()
    return {key: {**meter.to_dict(), 'unresolved': unresolved[key]} for key, meter in meters.items()}


def cache_update(package):
    calibration_predictions = None
    calibration_cases = None
    thresholds = None

    def update(cache, text=None, spans=None, *, allow_refit=True):
        nonlocal calibration_predictions, calibration_cases, thresholds
        if calibration_predictions is None:
            calibration_predictions = [cache.predict(row['text']) for row in package['calibration']]
            thresholds = (calibrate(package['calibration'], calibration_predictions, confirmed_cache=True),
                          package['slm_threshold'])
        if text is None:
            return cache, *thresholds
        if calibration_cases is None:
            calibration_cases = {}
            for i, (row, prediction) in enumerate(zip(package['calibration'], calibration_predictions)):
                key = prediction['template'], _case(row['text'], prediction['spans'])
                calibration_cases.setdefault(key, []).append(i)
        prior = cache.predict(text)
        matching = prior['spans'] is not None and span_set(prior['spans']) == span_set(spans)
        batch = None
        if not allow_refit and not matching and prior['template'] is not None:
            if not hasattr(cache, 'pending'):
                cache.pending = {}
            waiting = cache.pending.setdefault(prior['template'], [])
            spans = cache._observed(text, spans)
            if any(row['text'] == text and span_set(row['spans']) == span_set(spans) for row in waiting):
                return cache, *thresholds
            waiting.append({'text': text, 'spans': spans})
            counts = getattr(cache, 'updates', Counter())
            counts['slm_pending'] = sum(map(len, cache.pending.values()))
            cache.updates = counts
            if len(waiting) < 32:
                return cache, *thresholds
            batch = cache.pending.pop(prior['template'])
            counts['slm_pending'] -= len(batch)
            counts['slm_batches'] += 1
        affected = calibration_cases.get((prior['template'], _case(text, prior['spans'])), [])
        backup = pickle.dumps(cache) if batch is not None or affected or not matching else None
        refitted = cache.observe_many(batch) if batch is not None else cache.observe(text, spans)
        if refitted is None:
            return cache, *thresholds
        if refitted:
            affected = ([i for i, p in enumerate(calibration_predictions)
                         if p['template'] is None or p['template'] in refitted]
                        if isinstance(refitted, frozenset) else range(len(calibration_predictions)))
        elif not affected:
            counts = getattr(cache, 'updates', Counter())
            counts['accepted'] += 1
            cache.updates = counts
            return cache, *thresholds
        predictions = list(calibration_predictions)
        for i in affected:
            predictions[i] = cache.predict(package['calibration'][i]['text'])
        proposed_threshold = calibrate(package['calibration'], predictions, confirmed_cache=True)
        before = sum(allowed(p, thresholds[0]) for p in calibration_predictions)
        after = sum(allowed(p, proposed_threshold) for p in predictions)
        rollback = (proposed_threshold is None if batch is not None else
                    after < before and not (allow_refit and refitted and proposed_threshold is not None))
        if rollback:
            cache = pickle.loads(backup)
        counts = getattr(cache, 'updates', Counter())
        counts['rejected' if rollback else 'accepted'] += 1
        if batch is not None and rollback:
            counts['slm_batches_rejected'] += 1
        cache.updates = counts
        if rollback:
            return cache, *thresholds
        calibration_predictions, thresholds = predictions, (proposed_threshold, package['slm_threshold'])
        if refitted:
            calibration_cases = None
        return cache, *thresholds
    return update


def one_source(args, source):
    began_pipeline = time.perf_counter()
    names, work = categories(), args.work_dir / source
    work.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.slm_dir is None:
        training = train_source(source, args.data_dir, work / 'slm', args.device)
        print(json.dumps({'source': source, 'stage': 'training',
                          'seconds': training['seconds'], 'status': training['status']}), flush=True)
    model_path = work / 'cache.joblib'
    package = joblib.load(model_path) if model_path.exists() and not args.fit else None
    # Reject historical ablation packages while retaining full-method checkpoints.
    if package is not None and package.get('variant', 'full') != 'full':
        raise ValueError('This historical cache is not the complete LogSID method; use a new work directory')
    if package is not None and (package.get('calibration_protocol') != CALIBRATION_PROTOCOL
            or package.get('fit_report', {}).get('calibration_protocol') != CALIBRATION_PROTOCOL):
        raise ValueError('Calibration protocol changed; use --fit and a new output directory')
    slm_dir = (args.slm_dir or work / 'slm').expanduser().resolve()
    saved = source_metadata(slm_dir, source)
    slm_training = {'sources': saved['config']['source_names'],
                    **{key: saved['training'].get(key) for key in
                       ('seconds', 'train_supplied', 'validation_supplied', 'best_epoch')}}
    import torch
    from .slm import ScopeSLM
    torch.set_num_threads(2)
    slm = ScopeSLM.load(slm_dir, device=args.device)
    report = {'source': source, 'protocol': 'train20-entropy-2048-online-full-file',
              'calibration_protocol': CALIBRATION_PROTOCOL,
              'quality_target': {'recall': .90, 'f1': .85}}
    if package is None:
        fit, dev, cal = (records(args.data_dir, source, role)
                         for role in ('fit', 'validation', 'calibration'))
        began = time.perf_counter()
        cache = ParameterCache(names).fit(fit)
        report.update(fit_seconds=time.perf_counter() - began, training=cache.training)
        began = time.perf_counter()
        previous_slm_rows = slm.prediction_rows
        threshold, slm_threshold = calibrate_models(cache, cal, slm, source)
        report['calibration_seconds'] = time.perf_counter() - began
        report['calibration_slm_queries'] = slm.prediction_rows - previous_slm_rows
        report['calibration_cache_rejected'] = sum(
            not allowed(cache.predict(row['text']), threshold) for row in cal)
        began = time.perf_counter()
        dev_predictions = [cache.predict(r['text']) for r in dev]
        accepted = [p if allowed(p, threshold) else {'spans': None} for p in dev_predictions]
        report['development_raw'] = metrics(dev, dev_predictions, names)
        report['development_cache_accepted'] = metrics(dev, accepted, names)
        routes = Counter(cache=sum(p['spans'] is not None for p in accepted), slm=0, llm_required=0)
        report['development_slm_queries'] = 0
        if slm_threshold is not None:
            residual = [i for i, p in enumerate(accepted) if p['spans'] is None]
            predicted = slm.predict_batch([dev[i]['text'] for i in residual], sources=[source] * len(residual))
            for i, candidate in zip(residual, predicted):
                if allowed(candidate, slm_threshold):
                    accepted[i] = candidate
                    routes['slm'] += 1
            report['development_slm_queries'] = len(residual)
        routes['llm_required'] = sum(p['spans'] is None for p in accepted)
        report.update(development_seconds=time.perf_counter() - began,
                      development_accepted=metrics(dev, accepted, names), development_routes=dict(routes))
        report['sampling'] = {role: json.loads((args.data_dir / f'{source}.{role}.sampling.json').read_text())
                              for role in ('fit', 'validation', 'calibration')}
        package = {'cache': cache, 'threshold': threshold, 'slm_threshold': slm_threshold,
                   'slm_dir': str(slm_dir), 'fit_report': report,
                   'calibration': cal, 'calibration_protocol': CALIBRATION_PROTOCOL}
        joblib.dump(package, model_path)
    else:
        report.update(package['fit_report'])
        if report.get('protocol') != 'train20-entropy-2048-online-full-file':
            raise ValueError('Use a new work directory for the training20 entropy 2048-label protocol')
        if package['slm_dir'] != str(slm_dir):
            raise ValueError('Refit calibration when changing the enabled SLM')
    report.update(threshold=package['threshold'], slm_threshold=package['slm_threshold'],
                  slm_training=slm_training)
    if slm.device.type == 'cuda':
        torch.cuda.empty_cache()
    report['llm_model'] = args.llm_model
    if not args.probe_only:
        from .llm import LLM
        llm = LLM(work / 'detection_calls.sqlite3', model=report['llm_model'], max_calls=args.max_llm_queries,
                  retry_failed=args.retry_failed)
        rows = LogFile(source)
        path = args.output_dir / f'{source}.sqlite3'

        try:
            model_id = {'cache': str(model_path.resolve()), 'modified': model_path.stat().st_mtime_ns,
                        'slm': str(slm_dir), 'llm': llm.model,
                        'routing': ROUTING_VERSION, 'llm_prompt': llm.prompt_version,
                        'llm_request': llm.request_version}
            report['detection'] = detect(rows, package['cache'], package['threshold'], path,
                llm=llm, slm=slm, slm_threshold=package['slm_threshold'],
                update=cache_update(package),
                model_id=model_id, stop=args.max_rows or None, max_seconds=args.max_seconds or None,
                timing_boundary=len(rows) // 5 if args.record_timing else None)
            began = time.perf_counter()
            report['evaluation_status'] = report['detection']['status']
            report['evaluation'] = (evaluate(source, path, len(rows), names)
                                    if report['detection']['status'] == 'complete' else {})
            report['evaluation_seconds'] = time.perf_counter() - began
        finally:
            rows.close()
            llm.db.close()
    report['this_run_pipeline_seconds'] = time.perf_counter() - began_pipeline
    path = args.output_dir / f'{source}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    raw = report['development_accepted']
    print(json.dumps({'source': source, 'training': report['training'], 'slm_training': slm_training,
        'development_micro_f1': raw['span_micro']['f1'], 'development_macro_f1': raw['span_macro_f1_gold_present'],
        'development_unresolved': raw['unresolved'], 'development_seconds': report['development_seconds'],
        'threshold': report['threshold'], 'detection': report.get('detection'),
        'evaluation_status': report.get('evaluation_status', 'not_run'),
        'evaluation': {key: {'rows': value['rows'],
                            **{name: value['span_micro'][name] for name in ('recall', 'f1', 'fn')}}
                       for key, value in report.get('evaluation', {}).items()},
        'this_run_pipeline_seconds': report['this_run_pipeline_seconds']}, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    p.add_argument('--data-dir', type=Path, default=CACHE / 'logsid-paper/samples')
    p.add_argument('--work-dir', type=Path, default=CACHE / 'logsid-paper/work')
    p.add_argument('--output-dir', type=Path, default=CACHE / 'logsid-paper/output')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--record-timing', action='store_true', help='Record historical-prefix and test80 time; use a new result directory')
    p.add_argument('--probe-only', action='store_true')
    p.add_argument('--max-rows', type=int, default=0, help='0 selects the full file')
    p.add_argument('--llm-model', default='qwen/qwen3.8-27b')
    p.add_argument('--max-llm-queries', type=int, default=1024)
    p.add_argument('--max-seconds', type=float, default=900,
                   help='Cumulative detection time per source before checkpoint and inspection; 0 disables')
    p.add_argument('--retry-failed', action='store_true', help='After inspecting the API, explicitly retry failed or unknown calls once; preserve all prior attempts')
    p.add_argument('--slm-dir', type=Path, help='Reuse one completed target-source SLM instead of training it')
    p.add_argument('--device', default='cpu')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--read-key', action='store_true')
    args = p.parse_args()
    if args.read_key and not args.probe_only and not os.environ.get('OPENROUTER_API_KEY'):
        os.environ['OPENROUTER_API_KEY'] = getpass.getpass('OpenRouter key (not saved): ')
    if (args.max_rows < 0 or args.max_seconds < 0 or args.max_llm_queries < 1
            or args.workers < 1 or len(set(args.sources)) != len(args.sources)):
        p.error('Distinct sources and positive limits required')
    if args.slm_dir and len(args.sources) != 1:
        p.error('--slm-dir selects one completed target-source SLM; choose exactly one source')
    failures = []
    sources = list(args.sources)
    if len(sources) == 1:
        try:
            report = one_source(args, sources[0])
            detection = report.get('detection', {})
            if detection and detection['status'] != 'complete':
                raise RuntimeError(detection.get('reason') or detection['status'])
        except Exception as error:
            failures.append({'source': sources[0], 'stage': 'detection', 'error': f'{type(error).__name__}: {error}'})
            print(json.dumps(failures[-1]), flush=True)
    elif sources:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            queued = iter(sources)
            pending = {}
            halted = False
            for _ in range(min(args.workers, len(sources))):
                source = next(queued)
                pending[pool.submit(one_source, args, source)] = source
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    source = pending.pop(future)
                    try:
                        report = future.result()
                        detection = report.get('detection', {})
                        if detection and detection['status'] != 'complete':
                            raise RuntimeError(detection.get('reason') or detection['status'])
                    except (Exception, KeyboardInterrupt) as error:
                        halted = True
                        failures.append({'source': source, 'stage': 'detection',
                                         'error': f'{type(error).__name__}: {error}'})
                        print(json.dumps(failures[-1]), flush=True)
                while not halted and len(pending) < args.workers:
                    source = next(queued, None)
                    if source is None:
                        break
                    pending[pool.submit(one_source, args, source)] = source
            if halted:
                print(json.dumps({'not_started': list(queued),
                                  'reason': 'Inspect the stopped source before continuing.'}), flush=True)
    if failures:
        p.exit(1, 'Failed sources retain checkpoints; inspect errors before resuming.\n')


if __name__ == '__main__':
    main()
