"""Finish matched validation of the saved policy and original; no training, API or retries."""
import argparse
import json
import msvcrt
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_budget import lock_path as budget_lock
from storymaker.rlaif.code.rlaif_environment import GPU_LOCK
from storymaker.rlaif.code.rlaif_evaluate import policy_context, sha, output_origin
from storymaker.rlaif.code.rlaif_sequence import OUT, read_shared, save, lock_free, gpu_ready


def active_workers():
    script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $rows=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(rlaif_sequence|rlaif_generate|rlaif_reward_api|rlaif_probability_probe|rlaif_train|rlaif_evaluate|rlaif_value_calibration)(\\.py|\\s|$)' } | Select-Object ProcessId,ParentProcessId,CommandLine); ConvertTo-Json -InputObject $rows -Compress"
    return json.loads(subprocess.check_output(
        ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
        text=True, encoding='utf-8', timeout=20))


def verify_generation(receipt, variant):
    assert receipt['completed'] == 43 and receipt['tag'] == variant and receipt['split'] == 'validation'
    assert receipt['actual_training_updates_in_this_job'] == 0 and receipt['quality_claim'] is None


def verify_outputs(folder, variant, config_sha, plan_sha):
    receipt = read_shared(folder/'verified.json')
    verify_generation(receipt, variant)
    manifest = read_shared(folder/'manifest.json')
    assert receipt['manifest_sha256'] == sha(folder/'manifest.json')
    assert manifest['tag'] == variant and manifest['split'] == 'validation'
    assert manifest['config_sha256'] == config_sha and manifest['evaluation_plan_sha256'] == plan_sha
    ids = manifest['prompt_ids']
    assert len(ids) == len(set(ids)) == 43
    for number in ids:
        record = read_shared(folder/(number+'.json'))
        identity = record['identity']
        assert identity['tag'] == variant and identity['id'] == number
        assert identity['seed'] == manifest['settings']['seed']
        assert identity['policy_sha256'] == receipt['policy_sha256']
        assert record['quality_resampled'] is False and record['teacher_forced'] is not None
        assert record['generated_tokens'] == len(record['generated_token_ids']) <= manifest['settings']['max_new_tokens']
        if identity['manifest_sha256'] != receipt['manifest_sha256']:
            output_origin(folder/(number+'.json'))


def may_launch(folder, processes):
    assert not (folder/'failure.json').exists(), 'Inspect the preserved failure; never resample'
    status_path = folder/'status.json'
    if status_path.exists():
        state = read_shared(status_path)
        assert state['stage'] not in ('needs_attention', 'stopped_by_control'), 'Inspect the stopped validation'
    if processes:
        return False
    # An exited partial job requires inspection, never an implicit restart.
    if status_path.exists() or any(folder.glob('[0-9]*.json')):
        raise RuntimeError('Validation exited without a verified result; inspect before continuation')
    return True


def main(execute=False):
    folder = OUT/'evaluation/queue'
    folder.mkdir(parents=True, exist_ok=True)
    config = read_shared(OUT/'config.json')
    tag, checkpoint = policy_context(Path(config['policy_checkpoint']))
    variants = [(tag, checkpoint), ('original', None)]
    code_names = ['rlaif_evaluation_queue.py', 'rlaif_evaluate.py', 'rlaif_sequence.py', 'rlaif_train.py',
                  'rlaif_storage.py', 'rlaif_environment.py', 'generate_story.py', 'sft_v5_train.py',
                  'rlaif_precision.py', 'project.py']
    signatures = {name: sha(p.code_path(name)) for name in code_names}
    plan_path = OUT/'evaluation/evaluation_plan.json'
    config_sha, plan_sha = sha(OUT/'config.json'), sha(plan_path)
    pointer = p.ROOT/'rlaif/configs/active_run.json'
    identity = {'code_sha256': signatures, 'config_sha256': config_sha,
        'active_run_sha256': sha(pointer), 'evaluation_plan_sha256': plan_sha,
        'variants': [v for v, _ in variants],
        'scope': 'Saved policy/original matched validation43 generation+CE; no API or training',
        'automatic_restart': False}
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        assert read_shared(manifest_path) == identity, 'Changed queue context; inspect before proceeding'
    else:
        save(manifest_path, identity)
    if not execute:
        print(json.dumps({'plan_only': True, 'manifest': str(manifest_path), 'variants': identity['variants']}))
        return

    def enabled():
        return read_shared(OUT/'control.json')['enabled'] and read_shared(folder/'control.json')['enabled']

    def unchanged():
        assert sha(OUT/'config.json') == config_sha and sha(plan_path) == plan_sha
        assert sha(pointer) == identity['active_run_sha256']
        assert all(sha(p.code_path(name)) == digest for name, digest in signatures.items())

    def status(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at_utc': datetime.now(timezone.utc).isoformat(),
            'actual_training_updates_in_this_job': 0, 'paid_api_calls': 0, 'automatic_restart': False, **extra}
        save(folder/'state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    with (folder/'queue.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            for variant, checkpoint in variants:
                destination = OUT/'evaluation/validation'/variant
                verified = destination/'verified.json'
                consecutive = 0
                while enabled():
                    unchanged()
                    try:
                        processes = active_workers()
                        ready, snapshot = gpu_ready()
                        free = lock_free(GPU_LOCK) and lock_free(budget_lock(OUT))
                        complete = verified.exists()
                        if complete:
                            verify_outputs(destination, variant, config_sha, plan_sha)
                            launch = False
                        else:
                            launch = may_launch(destination, processes)
                        consecutive = consecutive+1 if (complete or launch) and not processes and free and ready else 0
                        status('waiting_matched_validation', variant=variant, processes=processes,
                            verified=complete, gpu=snapshot, released_locks=free, ready_observations=consecutive)
                        if consecutive >= 2:
                            break
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, OSError) as error:
                        consecutive = 0
                        status('inspection_error_no_restart', error=type(error).__name__, message=str(error))
                    time.sleep(30)
                else:
                    status('stopped_by_control', next_variant=variant)
                    return
                if not verified.exists():
                    unchanged()
                    assert enabled() and not active_workers() and lock_free(GPU_LOCK)
                    command = p.worker_command('rlaif_evaluate.py', '--split', 'validation', '--execute',
                        *(['--checkpoint', str(checkpoint)] if checkpoint else []), python=config['python'])
                    with (folder/(variant+'.stdout.log')).open('ab') as stdout, (folder/(variant+'.stderr.log')).open('ab') as stderr:
                        child = subprocess.Popen(command, cwd=p.ROOT.parent, stdout=stdout, stderr=stderr,
                                                 creationflags=subprocess.CREATE_NO_WINDOW)
                    while child.poll() is None:
                        status('running_matched_validation', variant=variant, worker_pid=child.pid)
                        time.sleep(30)
                    assert child.returncode == 0 and verified.exists(), 'Validation worker failed; inspect, never resample'
                verify_outputs(destination, variant, config_sha, plan_sha)
                status('validation_variant_complete', variant=variant)
            status('matched_generation_complete', outputs=86, quality_API='pending',
                   next='Paired quality diagnosis and cycle decision')
        except BaseException as error:
            status('needs_attention', error=type(error).__name__, message=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
