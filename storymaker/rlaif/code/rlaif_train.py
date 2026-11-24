"""Sequential full-parameter direct-RLAIF preparation and updates for one fixed batch."""
import argparse
import gc
import hashlib
import json
import msvcrt
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, gpu, validate_settings, preserve_batch_config
from storymaker.rlaif.code.rlaif_loss import regularized_return, policy_loss, value_loss, reward_baseline
from storymaker.rlaif.code.rlaif_storage import GradientBuffer, digest_tensor, optimizer_step, restore_parameters, save_json
from storymaker.rlaif.code.rlaif_precision import enable, projection


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(part)
    return result.hexdigest()


def model_digest(parameters):
    result = hashlib.sha256()
    for name, tensor in sorted(parameters):
        result.update(name.encode())
        result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        result.update(digest_tensor(tensor).encode())
    return result.hexdigest()


def verify_sources(config, phase):
    original = Path(config['source_sft_run']) / 'evaluation/generation_manifest.json'
    assert file_digest(original) == p.read(OUT / 'manifest.json')['source_sft_completed_manifest_sha256']
    expected = p.read(original)['identity']['source_sha256']
    model_root = Path(config['model']).resolve()
    checked = {}
    for filename, digest in expected.items():
        path = Path(filename)
        if path.resolve().is_relative_to(model_root) or path.resolve() == (p.PROCESSED / 'dataset.json').resolve():
            assert file_digest(path) == digest, filename
            checked[filename] = digest
    assert sum(Path(x).suffix == '.safetensors' for x in checked) == 2
    assert file_digest(p.PROCESSED / 'dataset.json') == p.read(OUT / 'manifest.json')['dataset_sha256']
    protected = p.read(Path(config['source_sft_run']) / 'watchdog/protected_sha256.json')
    for filename, digest in protected.items():
        assert file_digest(filename) == digest, filename
    save_json(OUT / 'environment' / ('sources_' + phase + '_verified.json'), {
        'source_files': checked, 'protected_files': len(protected), 'passed': True,
        'config_sha256': file_digest(OUT / 'config.json')})


def records_for(batch, config):
    paths = sorted((batch / 'rollouts').glob('*.json'))
    expected_ids = config.get('batch_ids', config.get('initial_batch_ids'))
    assert isinstance(expected_ids, list) and len(set(expected_ids)) == len(expected_ids)
    assert 0 < len(paths) == len(expected_ids) <= config['logical_batch_size']
    assert set(expected_ids) <= set(config['train_ids'])
    assert not set(expected_ids) & (set(config['validation_ids']) | set(config['excluded_test_ids']))
    rows = []
    for index, path in enumerate(paths):
        story = p.read(path)
        assert story['identity']['source_id'] == expected_ids[index]
        assert story['identity']['policy_version'] == config.get('policy_version', 'initial_original_0000')
        assert story['identity']['config_sha256'] == file_digest(OUT / 'config.json')
        assert story['identity']['source_id'] not in config['excluded_test_ids']
        assert len(story['generated_token_ids']) <= config['max_new_tokens']
        reward_path = batch / 'rewards' / path.stem / 'reward.json'
        reward = p.read(reward_path)
        if config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
            from storymaker.rlaif.code import rlaif_reward_pipeline as pipeline
            context=p.read(batch/'temporal_contexts'/path.name)
            assert pipeline.verified_score(path,reward_path.parent,context)==reward
        assert reward['source_sha256'] == file_digest(path)
        assert reward['certified_decimal_places'] == 6 and -1 <= reward['normalized_reward'] <= 1
        rows.append((path.stem, story, reward))
    return rows


def load_model(config, value=False, previous=None):
    model = AutoModelForImageTextToText.from_pretrained(config['model'], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa')
    model.requires_grad_(False)
    model.model.language_model.to('cuda')
    assert next(model.model.visual.parameters()).device.type == 'cpu'
    parameters = list(model.model.language_model.named_parameters())
    assert any(parameter is model.lm_head.weight for _, parameter in parameters), 'Expected tied language output weights'
    head = None
    if value:
        torch.manual_seed(config['seed'])
        head = torch.nn.Linear(model.config.text_config.hidden_size, 1, dtype=torch.float32, device='cuda')
        parameters += [('value_head.' + name, param) for name, param in head.named_parameters()]
    if previous is not None:
        restore_parameters(parameters, previous)
        manifest = p.read(Path(previous) / 'manifest.json')
        assert file_digest(Path(previous) / 'rng.pt') == manifest['rng_sha256']
        state = torch.load(Path(previous) / 'rng.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(state['cpu'])
        if state['cuda']:
            torch.cuda.set_rng_state_all(state['cuda'])
    enable(model, config.get('operation_precision'))
    model.eval()
    return model, head, parameters


def hidden_for(model, story):
    prompt, completion = story['prompt_ids'], story['generated_token_ids']
    ids = torch.tensor([prompt + completion], device='cuda')
    hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         use_cache=False, return_dict=True).last_hidden_state[0, len(prompt)-1:-1]
    assert hidden.shape[0] == len(completion)
    return hidden, ids[0, len(prompt):]


def chosen_logps(hidden, weight, targets, temperature):
    result = []
    for start in range(0, len(targets), 32):
        logits = projection(hidden[start:start+32], weight) / temperature
        result.append(-F.cross_entropy(logits, targets[start:start+32], reduction='none').cpu())
    return torch.cat(result)


def prepare(phase, batch, config, rows, previous, status):
    folder = batch / 'prepared' / phase
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'verified.json').exists(), 'Committed preparation is not rerun automatically'
    if phase == 'reference' and p.read(batch / 'prepared/policy/verified.json')['checkpoint'] is None:
        # The initial policy is literally the frozen original checkpoint, not an independently trained copy.
        source = batch / 'prepared/policy'
        policy = p.read(source / 'verified.json')
        assert policy['checkpoint'] is None and policy['source_verified']
        for key, _, _ in rows:
            shutil.copy2(source / (key + '.pt'), folder / (key + '.pt'))
        save_json(folder / 'verified.json', {'phase': phase, 'count': len(rows),
            'checkpoint': None, 'model_state_sha256': policy['model_state_sha256'],
            'same_initial_policy_and_reference': True, 'source_verified': True,
            'config_sha256': file_digest(OUT / 'config.json')})
        return
    status('loading_' + phase)
    model, head, parameters = load_model(config, value=phase == 'value', previous=previous)
    digest = model_digest(parameters)
    for index, (key, story, _) in enumerate(rows):
        assert p.read(OUT / 'control.json')['enabled']
        target = folder / (key + '.pt')
        assert not target.exists(), 'Inspect partial preparation before resuming'
        with torch.no_grad():
            hidden, tokens = hidden_for(model, story)
            value = head(hidden.float()).squeeze(-1).cpu() if head is not None else chosen_logps(
                    hidden, model.lm_head.weight, tokens, config['temperature'])
        assert torch.isfinite(value).all()
        torch.save({'values' if head is not None else 'logps': value,
                    'source_sha256': file_digest(batch / 'rollouts' / (key + '.json')),
                    'model_state_sha256': digest}, target)
        del hidden, tokens, value
        torch.cuda.empty_cache()
        status('preparing_' + phase, completed=index+1, total=len(rows))
    if head is not None:
        torch.save(head.state_dict(), folder / 'initial_head.pt')
    save_json(folder / 'verified.json', {'phase': phase, 'count': len(rows),
        'checkpoint': str(previous) if previous else None, 'model_state_sha256': digest,
        'source_verified': True, 'config_sha256': file_digest(OUT / 'config.json')})


def prepared_tensor(batch, phase, key, source_sha256, expected_model_sha256):
    item = torch.load(batch / 'prepared' / phase / (key + '.pt'), map_location='cpu', weights_only=True)
    assert item['source_sha256'] == source_sha256, 'Prepared values belong to another trajectory'
    assert item['model_state_sha256'] == expected_model_sha256, 'Prepared model identity changed'
    return item['values' if phase == 'value' else 'logps']


def update(phase, batch, config, rows, previous, status):
    policy = phase == 'policy'
    prepared_receipts = {}
    config_sha256 = file_digest(OUT / 'config.json')
    for name in ('policy', 'reference', 'value'):
        prepared_receipts[name] = p.read(batch / 'prepared' / name / 'verified.json')
        assert prepared_receipts[name]['count'] == len(rows) and prepared_receipts[name]['source_verified']
        assert prepared_receipts[name]['config_sha256'] == config_sha256
    equivalence = p.read(OUT / 'environment/probability_equivalence/verified.json')
    assert equivalence['passed'] and equivalence['config_sha256'] == config_sha256
    destination = OUT / 'checkpoints' / f"update_{config.get('update_index', 1):04d}" / phase
    assert not destination.exists(), 'A partial or committed update must be inspected, never overwritten'
    # Gradients+both roles' master checkpoints remain on disk; leave additional space for recovery.
    assert shutil.disk_usage(OUT).free > (85 if policy else 55)*2**30, 'Insufficient space; never delete old checkpoints automatically'
    model, head, parameters = load_model(config, value=not policy, previous=previous)
    prepared = batch / 'prepared' / phase
    identity = p.read(prepared / 'verified.json')['model_state_sha256']
    assert model_digest(parameters) == identity, 'Parameters changed after baseline/logprob preparation'
    for _, parameter in parameters:
        parameter.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    assert all(module.p == 0 for module in model.model.language_model.modules() if isinstance(module, torch.nn.Dropout))
    gradients = GradientBuffer(OUT / 'working' / (f"batch{config.get('batch_index', 0):04d}_" + phase), parameters)
    gradients.install()
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    for index, (key, story, reward) in enumerate(rows):
        assert p.read(OUT / 'control.json')['enabled']
        source_sha256 = file_digest(batch / 'rollouts' / (key + '.json'))
        old_values = prepared_tensor(batch, 'value', key, source_sha256, prepared_receipts['value']['model_state_sha256'])
        old_logps = prepared_tensor(batch, 'policy', key, source_sha256, prepared_receipts['policy']['model_state_sha256'])
        ref_logps = prepared_tensor(batch, 'reference', key, source_sha256, prepared_receipts['reference']['model_state_sha256'])
        assert old_values.shape == old_logps.shape == ref_logps.shape == (len(story['generated_token_ids']),)
        raw_reward = reward['normalized_reward']
        terminal = float(regularized_return(raw_reward, old_logps, ref_logps, config['kl_beta']))
        status('backward_' + phase, completed=index, total=len(rows), source_id=story['identity']['source_id'])
        activation_context = torch.autograd.graph.save_on_cpu(pin_memory=False) if config.get('cpu_saved_activations') else nullcontext()
        with activation_context:
            hidden, targets = hidden_for(model, story)
            if policy:
                loss = hidden.new_zeros((), dtype=torch.float32)
                discrepancy = 0.
                for start in range(0, len(targets), 16):
                    def part(h, t, baseline):
                        lp = -F.cross_entropy(projection(h, model.lm_head.weight)/config['temperature'], t, reduction='none')
                        return policy_loss(lp, terminal, reward_baseline(baseline, config['kl_beta'], config['value_baseline_bounds']), len(rows)), lp.detach()
                    piece, observed = checkpoint(part, hidden[start:start+16], targets[start:start+16],
                                   old_values[start:start+16].to('cuda'), use_reentrant=False)
                    loss = loss + piece
                    discrepancy = max(discrepancy, float((observed.cpu()-old_logps[start:start+16]).abs().max()))
                assert discrepancy <= .03, 'Training and prepared policy logprobs differ'
            else:
                observed = head(hidden.float()).squeeze(-1)
                discrepancy = float((observed.detach().cpu()-old_values).abs().max())
                assert discrepancy <= .03, 'The value baseline changed before its update'
                # The value baseline predicts the raw terminal API reward. Policy utility applies (1-beta) separately.
                loss = value_loss(observed, raw_reward, len(rows))
            assert torch.isfinite(loss)
        number = float(loss.detach())
        torch.cuda.empty_cache()
        loss.backward()
        metrics.append({'source_id': story['identity']['source_id'], 'tokens': len(targets),
            'raw_reward': raw_reward, 'regularized_return': terminal,
            'sample_sequence_logratio': float((old_logps-ref_logps).sum()),
            'mean_old_value': float(old_values.mean()), 'loss': number, 'logprob_or_value_max_difference': discrepancy})
        del hidden, targets, loss, observed, old_values, old_logps, ref_logps
        gc.collect()
        torch.cuda.empty_cache()
        save_json(batch / ('update_' + phase + '_progress.json'), {'completed': index+1, 'total': len(rows),
                  'optimizer_steps_committed': 0, 'last': metrics[-1]})
    norm = gradients.finish()
    assert norm > 0
    status('optimizer_' + phase, gradient_norm=norm)
    result = optimizer_step(parameters, gradients, destination, config['learning_rate'], previous=previous, status=status)
    receipt = {'phase': phase, 'trajectories': len(rows), 'optimizer_steps': 1,
        'gradient_norm': norm, 'metrics': metrics, 'checkpoint': str(destination),
        'changed_tensors': result['changed_tensors'], 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
        'objective': 'A.3/E: (1-beta)*API reward minus beta*sampled trajectory logratio; learned raw-reward value baseline',
        'value_baseline_bounds': config['value_baseline_bounds'],
        'value_head': 'independent same-start backbone plus linear scalar head' if not policy else None,
        'parameterization': 'full language backbone', 'saved_and_readback_verified': True,
        'config_sha256': config_sha256}
    save_json(batch / ('update_' + phase + '_verified.json'), receipt)
    gradients.close()
    status('complete_' + phase, optimizer_steps_committed=1, checkpoint=str(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare-policy', 'prepare-reference', 'prepare-value', 'update-policy', 'update-value'], required=True)
    parser.add_argument('--policy-checkpoint', type=Path)
    parser.add_argument('--value-checkpoint', type=Path)
    args = parser.parse_args()
    config = p.read(OUT / 'config.json')
    validate_settings(config)
    batch = OUT / f"batch_{config.get('batch_index', 0):04d}"
    started = time.monotonic()

    def status(stage, **extra):
        save_json(batch / 'training_status.json', {'stage': stage, 'phase': args.phase, 'pid': os.getpid(),
            'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'gpu': gpu(), **extra})
        print(stage, extra, flush=True)

    try:
        with GPU_LOCK.open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            assert gpu()['free_mib'] > 13500 and gpu()['temperature_c'] < 80
            preserve_batch_config(OUT,config)
            rows = records_for(batch, config)
            verify_sources(config, args.phase)
            if args.phase.startswith('prepare-'):
                role = args.phase.removeprefix('prepare-')
                checkpoint_path = (args.value_checkpoint or config.get('value_checkpoint')) if role == 'value' else (args.policy_checkpoint or config.get('policy_checkpoint'))
                if role == 'reference':
                    checkpoint_path = None
                prepare(role, batch, config, rows, checkpoint_path, status)
                status('complete_' + args.phase)
            else:
                role = args.phase.removeprefix('update-')
                previous = (args.policy_checkpoint or config.get('policy_checkpoint')) if role == 'policy' else (args.value_checkpoint or config.get('value_checkpoint'))
                update(role, batch, config, rows, previous, status)
    except BaseException as error:
        status('needs_attention', error=type(error).__name__, message=str(error), automatic_retry=False)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
