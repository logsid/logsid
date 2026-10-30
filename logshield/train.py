"""Train one SLM per source from its historical first-20% examples."""
import argparse
import json
from pathlib import Path

from .data import SOURCES, categories, records


def source_metadata(model_dir, source):
    saved = json.loads((Path(model_dir) / 'scope_slm.json').read_text())
    if (saved['config']['source_names'] != [source]
            or saved.get('metadata', {}).get('initial_model') is not None):
        raise ValueError(f'{source}: SLM must be trained from the pretrained backbone on this source alone')
    if saved['training'].get('status') != 'complete' or not (Path(model_dir) / 'model.pt').is_file():
        raise ValueError(f'{source}: SLM export is incomplete')
    return saved


def train_source(source, data_dir, model_dir, device, epochs=4, batch_size=4):
    model_dir = Path(model_dir).expanduser().resolve()
    saved = None
    if (model_dir / 'scope_slm.json').exists():
        saved = source_metadata(model_dir, source)
        for name, requested in (('epochs', epochs), ('batch_size', batch_size)):
            if saved['config'].get(name) != requested:
                raise ValueError(f'{source}: completed SLM {name}={saved["config"].get(name)!r} '
                                 f'differs from requested {requested!r}; use a new model directory')
    fit = records(data_dir, source, 'fit')
    dev = records(data_dir, source, 'validation')
    sampling = {'method': 'unleash_entropy_train20',
                'fit_line_ids': [row['line_id'] for row in fit],
                'holdout_line_ids': [row['line_id'] for row in dev]}
    if saved is not None:
        if saved['metadata'].get('sampling') != sampling:
            raise ValueError('Selected logs changed; train in a new model directory')
        for name in ('latest.pt', 'best-state.pt'):
            (model_dir / name).unlink(missing_ok=True)
        return saved['training']
    import torch
    from .slm import ScopeSLM
    torch.set_num_threads(2)
    model = ScopeSLM(categories(), {
        'device': device, 'source_names': [source], 'max_chars': 1024, 'max_tokens': 512,
        'epochs': epochs, 'batch_size': batch_size, 'encoder_layers_to_train': 4,
        'mixed_precision': 'bf16' if device.startswith('cuda') else 'no',
        'gradient_checkpointing': device.startswith('cuda')})
    model.metadata['sampling'] = sampling
    training = model.fit(fit, dev, model_dir)
    if training['status'] != 'complete':
        raise RuntimeError(f'{source}: SLM training did not complete; resume before detection')
    source_metadata(model_dir, source)
    for name in ('latest.pt', 'best-state.pt'):
        (model_dir / name).unlink(missing_ok=True)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Work root; each source is saved under SOURCE/slm')
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.sources)) != len(args.sources):
        parser.error('Distinct sources and positive training limits required')
    for source in args.sources:
        training = train_source(source, args.data_dir, args.output_dir / source / 'slm',
                                args.device, args.epochs, args.batch_size)
        print(json.dumps({'source': source, 'training_sources': [source],
                          'training_seconds': training['seconds'], 'status': training['status']}), flush=True)


if __name__ == '__main__':
    main()
