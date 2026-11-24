"""CPU-only element audit of the first committed policy/value update; never alters training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT
from storymaker.rlaif.code.rlaif_storage import read_parameter, save_json
from storymaker.rlaif.code.rlaif_train import file_digest, verify_sources


def flat_slice(value, shape, start, stop):
    stride = math.prod(shape[1:])
    first, last = start // stride, (stop + stride - 1) // stride
    return value[first:last].reshape(-1)[start % stride:start % stride + stop - start]


def compare_parameter(master, source, shape, dtype):
    assert list(master.shape) == list(shape) and master.dtype == torch.float32
    stats = dict(elements=master.numel(), fp32_changed_elements=0, stored_changed_elements=0,
                 before_squared_sum=0., fp32_delta_squared_sum=0., stored_delta_squared_sum=0.,
                 fp32_max_abs_delta=0., stored_max_abs_delta=0.)
    digest, source_digest = hashlib.sha256(), hashlib.sha256()
    for start in range(0, master.numel(), 1_000_000):
        stop = min(start + 1_000_000, master.numel())
        before = flat_slice(source, shape, start, stop).to(dtype).float()
        after = master.flatten()[start:stop]
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        stored = after.to(dtype)
        assert torch.isfinite(stored).all(), 'Stored inference weights overflowed'
        source_digest.update(before.to(dtype).contiguous().view(torch.uint8).numpy().tobytes())
        digest.update(stored.contiguous().view(torch.uint8).numpy().tobytes())
        fp32_delta = after.double() - before.double()
        stored_delta = stored.double() - before.double()
        stats['fp32_changed_elements'] += int(torch.count_nonzero(fp32_delta))
        stats['stored_changed_elements'] += int(torch.count_nonzero(stored_delta))
        stats['before_squared_sum'] += float(before.double().square().sum())
        stats['fp32_delta_squared_sum'] += float(fp32_delta.square().sum())
        stats['stored_delta_squared_sum'] += float(stored_delta.square().sum())
        stats['fp32_max_abs_delta'] = max(stats['fp32_max_abs_delta'], float(fp32_delta.abs().max()))
        stats['stored_max_abs_delta'] = max(stats['stored_max_abs_delta'], float(stored_delta.abs().max()))
    stats['stored_sha256'] = digest.hexdigest()
    stats['source_sha256'] = source_digest.hexdigest()
    stats['shape'] = list(shape)
    for prefix in ('fp32', 'stored'):
        stats[prefix + '_changed_fraction'] = stats[prefix + '_changed_elements'] / stats['elements']
        stats[prefix + '_delta_rms'] = math.sqrt(stats[prefix + '_delta_squared_sum'] / stats['elements'])
    stats['before_rms'] = math.sqrt(stats['before_squared_sum'] / stats['elements'])
    stats['stored_dtype'] = str(dtype)
    return stats


def audit_checkpoint(checkpoint, model_root, value_head=None):
    assert not torch.cuda.is_initialized(), 'Audit must not load a CUDA model'
    checkpoint, model_root = Path(checkpoint), Path(model_root)
    manifest = p.read(checkpoint / 'manifest.json')
    assert manifest['previous'] is None, 'This audit compares update0001 against its original initialization'
    assert file_digest(checkpoint / 'rng.pt') == manifest['rng_sha256']
    weight_map = p.read(model_root / 'model.safetensors.index.json')['weight_map']
    prefix = 'model.language_model.'
    names = {name.removeprefix(prefix) for name in weight_map if name.startswith(prefix)}
    heads = torch.load(value_head, map_location='cpu', weights_only=True) if value_head else {}
    expected = names | {'value_head.' + name for name in heads}
    assert len(manifest['parameters']) == len(expected)
    assert {row['name'] for row in manifest['parameters']} == expected, 'Missing or unexpected trainable parameters'
    rows = []
    for record in manifest['parameters']:
        name = record['name']
        assert (checkpoint / record['file']).resolve().is_relative_to(checkpoint.resolve())
        metadata = torch.load(checkpoint / record['file'], map_location='cpu', weights_only=True)
        assert all((checkpoint / row['file']).resolve().is_relative_to(checkpoint.resolve()) for row in metadata['chunks'])
        master, state = read_parameter(checkpoint, record)
        assert int(state['state'][0]['step']) == 1 and len(state['param_groups']) == 1
        group = state['param_groups'][0]
        assert group['weight_decay'] == 0. and group['foreach'] is False
        if name.startswith('value_head.'):
            initial = heads[name.removeprefix('value_head.')]
            assert initial.dtype == torch.float32
            result = compare_parameter(master, initial, initial.shape, torch.float32)
        else:
            key = prefix + name
            source_file = (model_root / weight_map[key]).resolve()
            assert source_file.is_relative_to(model_root.resolve())
            with safe_open(source_file, framework='pt', device='cpu') as reader:
                initial = reader.get_slice(key)
                # The runner explicitly loads all language parameters as BF16.
                result = compare_parameter(master, initial, initial.get_shape(), torch.bfloat16)
        assert result['stored_sha256'] == record['bf16_sha256'], 'Audit must match the committed inference weights'
        rho = min(group['lr'], 1.)
        rows.append(dict(name=name, **result, optimizer_step=1, configured_lr=group['lr'],
            adafactor_rho=rho, estimated_alpha_before_adaptive_clipping=max(group['eps'][1], result['before_rms']) * rho))
        del master, state, metadata, initial
    assert sum(row['stored_changed_elements'] > 0 for row in rows) == manifest['changed_tensors']
    totals = {key: sum(row[key] for row in rows) for key in ['elements', 'fp32_changed_elements',
              'stored_changed_elements', 'before_squared_sum', 'fp32_delta_squared_sum', 'stored_delta_squared_sum']}
    for prefix in ('fp32', 'stored'):
        totals[prefix + '_changed_fraction'] = totals[prefix + '_changed_elements'] / totals['elements']
        totals[prefix + '_delta_rms'] = math.sqrt(totals[prefix + '_delta_squared_sum'] / totals['elements'])
    source_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item['name']):
        source_digest.update(row['name'].encode())
        source_digest.update(str((tuple(row['shape']), row['stored_dtype'])).encode())
        source_digest.update(row['source_sha256'].encode())
    return {'parameters': rows, 'totals': totals, 'initial_model_sha256': source_digest.hexdigest(),
        'checkpoint_manifest_sha256': file_digest(checkpoint / 'manifest.json'),
        'CUDA_initialized': torch.cuda.is_initialized(), 'new_optimizer_steps': 0, 'paid_API_calls': 0,
        'scope': 'All parameter entries versus original loaded BF16 backbone and initial FP32 value head; not a quality or convergence verdict',
        'optimizer_note': 'Torch Adafactor uses parameter-RMS scaling and adaptive clipping. Configured1e-5 is not a constant AdamW1e-5 element step; alpha estimate uses float64 source RMS.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['policy', 'value'], default='policy')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    receipt_path = OUT / f'batch_0000/update_{args.role}_verified.json'
    destination = OUT / f'batch_0000/update_{args.role}_element_audit.json'
    if not args.execute:
        print(json.dumps({'plan_only': True, 'committed_update_exists': receipt_path.exists(),
            'output': str(destination), 'GPU_models_loaded': 0, 'API_calls': 0}))
        return
    torch.set_num_threads(2)
    assert not destination.exists(), 'Never overwrite an earlier audit'
    config = p.read(OUT / 'config.json')
    receipt = p.read(receipt_path)
    assert receipt['config_sha256'] == file_digest(OUT / 'config.json')
    assert receipt['optimizer_steps'] == 1 and receipt['saved_and_readback_verified']
    checkpoint = OUT / 'checkpoints/update_0001' / args.role
    assert checkpoint.resolve() == Path(receipt['checkpoint']).resolve()
    verify_sources(config, 'first_update_element_audit_' + args.role)
    result = audit_checkpoint(checkpoint, config['model'],
        OUT / 'batch_0000/prepared/value/initial_head.pt' if args.role == 'value' else None)
    prepared_path = OUT / f'batch_0000/prepared/{args.role}/verified.json'
    prepared = p.read(prepared_path)
    assert prepared['config_sha256'] == receipt['config_sha256']
    assert result['initial_model_sha256'] == prepared['model_state_sha256'], 'Initial backbone or value head differs from the actual prepared model'
    result.update(role=args.role, config_sha256=receipt['config_sha256'],
        update_receipt_sha256=file_digest(receipt_path), prepared_receipt_sha256=file_digest(prepared_path),
        auditor_sha256=file_digest(__file__))
    save_json(destination, result)
    print(json.dumps({'audit': str(destination), 'totals': result['totals'], 'GPU_models_loaded': 0}))


if __name__ == '__main__':
    main()
