"""Prepare original data, the pinned sampler, and the RoBERTa backbone."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from .io import (DEFAULT_CACHE, ROOT, ORDER_PROTOCOL, atomic_json, build_order,
                 file_sha256, fingerprints, load_order, source_inventory)
from .sampling import NATIVE_FILE
from .index import prepare_source
from .checkpoint import COMMIT, MODEL_ID, MODEL_REVISION
from .data import SOURCES


def announce(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def prepare_order(data_dir, source, order_dir):
    """Add one verified chronological permutation without replacing other sources."""
    order_dir = Path(order_dir)
    manifest = order_dir / 'metadata.json'
    saved = json.loads(manifest.read_text()) if manifest.exists() else None
    current = fingerprints(data_dir, [source])
    if saved is not None:
        if saved.get('status') != 'complete' or saved.get('protocol') != ORDER_PROTOCOL:
            raise ValueError('Existing ordering is incomplete or incompatible; use a new LOGSID_CACHE')
        for name, digest in current.items():
            if name in saved['data_fingerprint'] and saved['data_fingerprint'][name] != digest:
                raise ValueError('Dataset or taxonomy changed; use a new LOGSID_CACHE')
        if source in saved['sources']:
            if any(saved['data_fingerprint'].get(name) != digest for name, digest in current.items()):
                raise ValueError('Existing ordering is missing data fingerprints')
            _, verified = load_order(order_dir, source,
                                     expected_rows=int(source_inventory(data_dir)[source]['Logs']))
            return verified['sources'][source]

    order_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=order_dir) as temporary:
        staging = Path(temporary)
        prepared = build_order(data_dir, source, staging, progress=lambda event: announce(**event))
        combined = saved or {'protocol': ORDER_PROTOCOL, 'status': 'complete',
                             'sources': {}, 'data_fingerprint': {}}
        combined['sources'][source] = prepared
        combined['data_fingerprint'].update(current)
        # The manifest is the commit point. A failed preparation can be retried.
        name = prepared['order_file']
        os.replace(staging / name, order_dir / name)
        atomic_json(manifest, combined)
    _, verified = load_order(order_dir, source)
    return verified['sources'][source]


def prepare_data(data_dir, sources, cache_dir=DEFAULT_CACHE):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    inventory = source_inventory(data_dir)
    for source in sources:
        if source not in inventory:
            raise ValueError(f'Source absent from data/statistics.csv: {source}')
        for name in (source + '.log.gz', source + '_labels.jsonl.gz'):
            if not (data_dir / name).is_file():
                raise FileNotFoundError(f'Missing dataset file: {data_dir / name}')
    for source in sources:
        manifest = cache_dir / 'log-data' / source / 'index.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if (saved.get('source_file_sha256') != file_sha256(data_dir / (source + '_labels.jsonl.gz'))
                    or saved.get('taxonomy_sha256') != file_sha256(data_dir / 'taxonomy.csv')):
                raise ValueError('Existing index belongs to different data; use a new LOGSID_CACHE')
        announce(operation='index', source=source, status='preparing')
        index = prepare_source(data_dir, source, cache_dir / 'log-data')
        order = prepare_order(data_dir, source, cache_dir / 'order')
        announce(operation='data', source=source, status='complete', rows=index['rows'],
                 ordering=order['ordering'])


def prepare_sampler(cache_dir=DEFAULT_CACHE):
    manifest = json.loads((ROOT / 'logshield/sampling_source.json').read_text())
    if manifest['commit'] != COMMIT:
        raise ValueError('Sampler manifest and implementation commits disagree')
    entry = manifest['source_files'][NATIVE_FILE]
    destination = Path(cache_dir) / 'upstream' / ('UNLEASH-' + COMMIT) / NATIVE_FILE
    if destination.exists():
        if file_sha256(destination) != entry['sha256']:
            raise ValueError('Existing sampler checksum differs from the pinned source')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix='.sampler-', dir=destination.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, 'wb') as output, urlopen(entry['url'], timeout=60) as response:
            shutil.copyfileobj(response, output)
            output.flush()
            os.fsync(output.fileno())
        if file_sha256(temporary) != entry['sha256']:
            raise ValueError('Downloaded sampler checksum differs from the pinned source')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_model(cache_dir=DEFAULT_CACHE):
    from huggingface_hub import snapshot_download

    options = dict(repo_id=MODEL_ID, revision=MODEL_REVISION,
                   cache_dir=str(Path(cache_dir) / 'huggingface'))
    # Prefer safetensors; older pinned snapshots may only provide PyTorch bins.
    directory = Path(snapshot_download(**options, allow_patterns=[
        'config.json', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'vocab.json', 'merges.txt', 'model.safetensors']))
    if not (directory / 'model.safetensors').is_file():
        directory = Path(snapshot_download(**options, allow_patterns=['pytorch_model.bin']))
    required = ('config.json', 'vocab.json', 'merges.txt')
    if (any(not (directory / name).is_file() for name in required)
            or not any((directory / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))):
        raise ValueError('Pinned RoBERTa snapshot is incomplete')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Use LOGSID_CACHE to choose the cache directory before starting Python. '
        'Run one preparation process per cache. No training or LLM calls are made.'))
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--skip-model', action='store_true', help='Skip the pretrained backbone download')
    parser.add_argument('--skip-sampler', action='store_true', help='Skip the pinned sampling-code download')
    args = parser.parse_args()
    prepare_data(ROOT / 'data', list(dict.fromkeys(args.sources)))
    if not args.skip_sampler:
        announce(operation='sampler', status='complete', path=str(prepare_sampler()))
    if not args.skip_model:
        announce(operation='backbone', status='complete', path=str(prepare_model()))


if __name__ == '__main__':
    main()
