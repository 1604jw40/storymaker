"""Read-only value calibration on a committed batch's unchanged training states."""
import argparse
import gc
import json
import math
import msvcrt
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, gpu
from storymaker.rlaif.code.rlaif_sequence import gpu_ready, lock_free, host_processes
from storymaker.rlaif.code.rlaif_storage import save_json
from storymaker.rlaif.code.rlaif_train import (file_digest, hidden_for, load_model, model_digest,
                         prepared_tensor)
from storymaker.rlaif.code.rlaif_training_report import committed_batch


def trajectory_metrics(old, new, reward, source_id, eos):
    assert old.ndim == new.ndim == 1 and old.shape == new.shape and old.numel() > 0
    assert torch.isfinite(old).all() and torch.isfinite(new).all()
    assert math.isfinite(reward) and -1 <= reward <= 1 and isinstance(eos, bool)
    result = {'source_id': source_id, 'tokens': old.numel(), 'reward': reward, 'eos': eos,
              'zero_squared_error': reward**2*old.numel()}
    for name, values in [('before', old.double()), ('after', new.double())]:
        result[name + '_squared_error'] = float((values - reward).square().sum())
        result[name + '_projected_squared_error'] = float((values.clamp(-1,1)-reward).square().sum())
        result[name + '_sum'] = float(values.sum())
        result[name + '_outside_reward_range'] = int((values.abs() > 1).sum())
    return result


def summarize(rows, expected_ids):
    assert [r['source_id'] for r in rows] == expected_ids and len(set(expected_ids)) == len(rows) > 0
    assert all(r['tokens'] > 0 for r in rows)
    tokens = sum(r['tokens'] for r in rows)
    result = {'trajectories': len(rows), 'tokens': tokens, 'non_EOS_trajectories': sum(not r['eos'] for r in rows)}
    result['zero_baseline_token_MSE'] = sum(r['zero_squared_error'] for r in rows)/tokens
    for phase in ['before', 'after']:
        result[phase] = {'token_MSE': sum(r[phase + '_squared_error'] for r in rows) / tokens,
                         'projected_token_MSE': sum(r[phase + '_projected_squared_error'] for r in rows)/tokens,
                         'token_mean_value': sum(r[phase + '_sum'] for r in rows) / tokens,
                         'outside_reward_range_fraction': sum(r[phase + '_outside_reward_range'] for r in rows) / tokens}
    result['MSE_after_minus_before'] = result['after']['token_MSE'] - result['before']['token_MSE']
    result['projected_after_minus_zero_MSE'] = result['after']['projected_token_MSE']-result['zero_baseline_token_MSE']
    return result


def calibration_source(root, checkpoint):
    assert checkpoint.name=='value'
    batch=committed_batch(root,checkpoint.parent.name)
    joint=batch/'joint_update_verified.json'
    if not joint.exists():joint=batch/'first_update_verified.json'
    committed=p.read(joint);assert Path(committed['value']).resolve()==checkpoint.resolve()
    source_config=batch/'config.json'
    if not source_config.exists():source_config=batch.parent/'config.json'
    config=p.read(source_config);ids=config.get('batch_ids',config.get('initial_batch_ids'))
    assert isinstance(ids,list) and 0<len(ids)==len(set(ids))<=config['logical_batch_size']
    assert set(ids)<=set(config['train_ids']) and not set(ids)&set(config['validation_ids']+config['excluded_test_ids'])
    for role in ('policy','value'):
        receipt=p.read(batch/f'update_{role}_verified.json')
        assert receipt['phase']==role and receipt['config_sha256']==file_digest(source_config)
        assert receipt['optimizer_steps']==1 and receipt['trajectories']==len(ids) and receipt['saved_and_readback_verified']
        assert Path(receipt['checkpoint']).resolve()==Path(committed[role]).resolve()
    return batch,source_config,ids,joint


def ready_for_calibration(stage, workers, free_lock, ready_gpu, monitor):
    observed = datetime.fromisoformat(monitor['at_utc'])
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    fresh = 0 <= age < 120
    return (stage == 'committed_update_verified' and not workers and free_lock and ready_gpu
            and fresh and monitor.get('inspection_error') is None and not monitor.get('alerts')
            and monitor.get('ram', {}).get('free_kib', 0) > 8 * 1024**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--value-checkpoint',type=Path)
    args = parser.parse_args()
    folder = OUT / 'value_calibration'
    if args.value_checkpoint is not None:folder=folder/args.value_checkpoint.parent.name
    if not args.execute:
        print(json.dumps({'plan_only': True, 'output': str(folder), 'GPU_jobs_started': 0, 'API_calls': 0,
                          'value_checkpoint':str(args.value_checkpoint) if args.value_checkpoint else None,
                          'waits_for': 'Verified committed joint update and free GPU/native lock'}))
        return
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder / 'manifest.json').exists(), 'Inspect prior or partial calibration; never overwrite'
    started = time.monotonic()
    own_lock_obtained = False

    def status(stage, **extra):
        save_json(folder / 'status.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
                  'pid': os.getpid(), 'stage': stage, 'elapsed_seconds_including_wait': time.monotonic() - started,
                  'new_optimizer_steps': 0, 'paid_API_calls': 0, 'automatic_retry': False, **extra})

    try:
        with (folder / 'calibration.lock').open('a+b') as own_lock:
            own_lock.seek(0)
            msvcrt.locking(own_lock.fileno(), msvcrt.LK_NBLCK, 1)
            own_lock_obtained = True
            assert not (folder / 'manifest.json').exists(), 'Inspect prior or partial calibration; never overwrite'
            config = p.read(OUT / 'config.json')
            checkpoint = args.value_checkpoint or Path(config['value_checkpoint'])
            batch,source_config_path,expected_ids,joint=calibration_source(OUT,checkpoint)
            source_config=p.read(source_config_path);source_config_sha=file_digest(source_config_path)
            assert set(expected_ids)<=set(config['train_ids']) and not set(expected_ids)&set(config['validation_ids']+config['excluded_test_ids'])
            files = [p.code_path(n) for n in ('rlaif_train.py','rlaif_storage.py','rlaif_precision.py',
                     'rlaif_value_calibration.py','rlaif_environment.py','rlaif_sequence.py','rlaif_training_report.py','project.py')]
            files += [OUT / 'config.json', source_config_path, checkpoint / 'manifest.json', joint,
                      batch/'update_policy_verified.json',batch/'update_value_verified.json',batch / 'prepared/value/verified.json']
            signatures = {str(path): file_digest(path) for path in files}
            save_json(folder / 'manifest.json', {'source_sha256': signatures,
                      'checkpoint':str(checkpoint),'source_batch':str(batch),'actual_trajectories':len(expected_ids),
                      'scope': 'Read-only same-batch training-state value calibration; frozen actual rewards, no new policy update or generalization estimate'})

            def unchanged():
                assert all(file_digest(path) == digest for path, digest in signatures.items())

            consecutive = 0
            while p.read(OUT / 'control.json')['enabled']:
                unchanged()
                state = {'stage': 'committed_update_verified'}
                try:
                    ready, snapshot = gpu_ready()
                    workers = [r for r in host_processes() if 'rlaif_reward_api' not in r['CommandLine'] or '--prepare-rag' in r['CommandLine']]
                    monitor = p.read(OUT / 'monitor/state.json')
                    eligible = (p.read(OUT / 'monitor/control.json')['enabled'] and
                                ready_for_calibration(state['stage'], workers,
                                    lock_free(GPU_LOCK), ready, monitor))
                    consecutive = consecutive + 1 if eligible else 0
                    status('waiting_verified_source_and_gpu', ready_observations=consecutive,
                           validation_stage=state['stage'], gpu=snapshot)
                    if consecutive >= 2:
                        break
                except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as error:
                    consecutive = 0
                    status('inspection_error_no_restart', error=type(error).__name__, message=str(error))
                time.sleep(30)
            else:
                status('stopped_by_control')
                return
            with GPU_LOCK.open('a+b') as gpu_lock:
                gpu_lock.seek(0)
                msvcrt.locking(gpu_lock.fileno(), msvcrt.LK_NBLCK, 1)
                unchanged()
                assert not [r for r in host_processes() if 'rlaif_reward_api' not in r['CommandLine'] or '--prepare-rag' in r['CommandLine']] and gpu_ready()[0]
                paths = sorted((batch / 'rollouts').glob('*.json'))
                assert len(paths) == len(expected_ids)
                rows = []
                for i, path in enumerate(paths):
                    story = p.read(path)
                    reward = p.read(batch / 'rewards' / path.stem / 'reward.json')
                    assert story['identity']['config_sha256'] == source_config_sha
                    assert story['identity']['source_id'] == expected_ids[i] and expected_ids[i] in config['train_ids']
                    assert reward['source_sha256'] == file_digest(path)
                    rows.append((path.stem, story, reward))
                status('loading_committed_value')
                model, head, parameters = load_model(config, value=True, previous=checkpoint)
                head.requires_grad_(False)
                assert all(not parameter.requires_grad for _, parameter in parameters)
                loaded_digest = model_digest(parameters)
                torch.cuda.reset_peak_memory_stats()
                old_identity = p.read(batch / 'prepared/value/verified.json')['model_state_sha256']
                metrics, sources = [], {}
                for index, (key, story, reward) in enumerate(rows):
                    assert p.read(OUT / 'control.json')['enabled']
                    source = batch / 'rollouts' / (key + '.json')
                    source_sha = file_digest(source)
                    old = prepared_tensor(batch, 'value', key, source_sha, old_identity)
                    with torch.no_grad():
                        hidden, targets = hidden_for(model, story)
                        new = head(hidden.float()).squeeze(-1).cpu()
                    metrics.append(trajectory_metrics(old, new, reward['normalized_reward'],
                                   story['identity']['source_id'], story['ended_with_eos']))
                    target = folder / (key + '.pt')
                    assert not target.exists()
                    torch.save({'values': new, 'source_sha256': source_sha, 'model_state_sha256': loaded_digest}, target)
                    sources[key] = {'rollout_sha256': source_sha, 'after_tensor_sha256': file_digest(target),
                                    'before_tensor_sha256': file_digest(batch / 'prepared/value' / (key + '.pt'))}
                    del hidden, targets, old, new
                    torch.cuda.empty_cache()
                    status('predicting_committed_value', completed=index + 1, total=len(rows), gpu=gpu())
                result = summarize(metrics, expected_ids)
                previous = p.read(batch / 'training_report/summary.json')['value_token_MSE_diagnostic']
                assert math.isclose(result['before']['token_MSE'], previous, rel_tol=1e-6, abs_tol=1e-6)
                assert model_digest(parameters) == loaded_digest
                unchanged()
                result.update(model_state_sha256=loaded_digest, source_sha256=signatures, predictions=sources,
                              rows=metrics, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                              new_optimizer_steps=0, paid_API_calls=0,
                              scope='Post-update value calibration on the SAME pre-update training states; not generalization or new-policy quality',
                              limitations=['MSE/range changes do not show better storytelling or guarantee negative advantage on future unfinished outputs.',
                                           'Same training states and frozen batch rewards: in-sample prediction diagnostic, not held-out storytelling. Every unfinished case retained.'])
                save_json(folder / 'verified.json', result)
                del model, head, parameters
                gc.collect()
                torch.cuda.empty_cache()
                status('complete', completed=len(rows), total=len(rows), before_MSE=result['before']['token_MSE'],
                       after_MSE=result['after']['token_MSE'])
    except BaseException as error:
        if own_lock_obtained:
            status('needs_attention', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
