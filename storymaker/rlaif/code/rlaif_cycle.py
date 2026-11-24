"""One authorized collect/update/paired-validation cycle; no automatic restart."""
import argparse
import json
import msvcrt
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import ACTIVE_RUN, OUT, GPU_LOCK, validate_settings
from storymaker.rlaif.code.rlaif_sequence import lock_free, gpu_ready
from storymaker.rlaif.code.rlaif_reward_api import save
from storymaker.rlaif.code.rlaif_budget import lock_path
from storymaker.rlaif.code.reward_rag import sha


def commands(config):
    current = f"update_{config['update_index']:04d}"
    previous = Path(config['policy_checkpoint']).parent.name
    assert current != previous
    checkpoint = OUT/'checkpoints'/current
    return [
        ('collection', 'rlaif_generate.py', []),
        ('train_RAG', 'rlaif_reward_api.py', ['--prepare-rag']),
        ('train_reward', 'rlaif_reward_api.py', ['--execute']),
        ('joint_update', 'rlaif_sequence.py', ['--execute']),
        ('training_report', 'rlaif_training_report.py', []),
        ('value_diagnostic', 'rlaif_value_calibration.py', ['--value-checkpoint', str(checkpoint/'value'), '--execute']),
        ('validation_original', 'rlaif_evaluate.py', ['--split', 'validation', '--execute']),
        ('validation_previous', 'rlaif_evaluate.py', ['--split', 'validation', '--checkpoint', config['policy_checkpoint'], '--execute']),
        ('validation_current', 'rlaif_evaluate.py', ['--split', 'validation', '--checkpoint', str(checkpoint/'policy'), '--execute']),
        ('previous_RAG', 'rlaif_reward_evaluate.py', ['--policy-tag', previous, '--prepare-rag']),
        ('previous_reward', 'rlaif_reward_evaluate.py', ['--policy-tag', previous, '--execute']),
        ('current_RAG', 'rlaif_reward_evaluate.py', ['--policy-tag', current, '--prepare-rag']),
        ('current_reward', 'rlaif_reward_evaluate.py', ['--policy-tag', current, '--execute']),
    ]


def compare(config):
    from storymaker.qwen.code.evaluation_report import interval
    previous = Path(config['policy_checkpoint']).parent.name
    current = f"update_{config['update_index']:04d}"
    folder = OUT/'evaluation/reward_metrics'
    before = p.read(folder/f'validation_{previous}/report.json')['rows']
    after = p.read(folder/f'validation_{current}/report.json')['rows']
    assert [r['id'] for r in before] == [r['id'] for r in after] == list(map(str, config['validation_ids']))
    differences = []
    for a, b in zip(before, after):
        for key in ('normalized_reward', 'judgment'):
            assert a['models']['original']['result'][key] == b['models']['original']['result'][key]
        differences.append({'source_id': a['id'], 'expected_reward_difference': b['models'][current]['result']['normalized_reward']-a['models'][previous]['result']['normalized_reward']})
    values = [r['expected_reward_difference'] for r in differences]
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'N': len(values), 'previous': previous, 'current': current,
              'mean_expected_reward_difference': statistics.mean(values), 'source_bootstrap95': interval(values),
              'win_tie_loss': [sum(v > 0 for v in values), sum(v == 0 for v in values), sum(v < 0 for v in values)],
              'rows': differences, 'summaries': {tag: p.read(folder/f'validation_{tag}/summary.json')['models'] for tag in (previous, current)},
              'scope': 'Same43 morals, new compact prompt, fixed new reward and identical conditions for original/previous/current. Old frozen scores remain separate. One seed and same API family, not expert quality or measured editing time.'}
    save(OUT/'evaluation/consecutive_comparison.json', result)
    return result


def resume_inputs(folder, config_sha, runtime, batch, count):
    authorization = p.read(folder/'resume_authorization.json')
    assert authorization['passed'] and authorization['resume_from'] == 'train_reward'
    assert authorization['config_sha256'] == config_sha and authorization['runtime_code_sha256'] == runtime
    prior = Path(authorization['previous_state_file'])
    assert sha(prior) == authorization['previous_state_sha256']
    assert p.read(prior)['message'] == 'train_reward failed; preserve logs, no automatic restart'
    assert p.read(folder/'cycle_state.json')['stage'] == 'needs_attention'
    for name in ('collection_verified.json', 'train_RAG_verified.json'):
        path = folder.parent/'environment'/name
        assert sha(path) == authorization['evidence_sha256'][str(path)]
        proof = p.read(path)
        if name == 'collection_verified.json':
            assert proof['config_sha256'] == config_sha and len(proof['rows']) == count
        assert proof['passed'] and proof.get('N', proof.get('contexts')) == count
        sources = proof.get('source_sha256', {})
        if isinstance(sources, dict):
            assert all(sha(Path(path)) == digest for path, digest in sources.items())
        if name == 'collection_verified.json':
            assert all(sha(Path(row['source'])) == row['source_sha256'] for row in proof['rows'])
    assert not (batch/'update_policy_verified.json').exists()
    return authorization


def main(execute, resume_from=None):
    config = p.read(OUT/'config.json')
    count = len(validate_settings(config))
    batch = OUT/f"batch_{config.get('batch_index', 0):04d}"
    folder = OUT/'pipeline'
    folder.mkdir(exist_ok=True)
    manifest = p.read(OUT/'manifest.json')
    config_sha = sha(OUT/'config.json')
    assert config_sha == manifest['config_sha256']
    plan = {'config_sha256': config_sha, 'commands': commands(config), 'code_sha256': manifest['runtime_code_sha256'], 'automatic_restart': False}
    if (folder/'cycle_plan.json').exists():
        assert p.read(folder/'cycle_plan.json') == json.loads(json.dumps(plan))
    else:
        save(folder/'cycle_plan.json', plan)
    if not execute:
        print(json.dumps({'plan_only': True, 'batch_size': count, 'stages': len(plan['commands']), 'new_API_calls': 0}))
        return

    def enabled():
        return p.read(OUT/'control.json')['enabled'] and p.read(folder/'control.json')['enabled']

    def guard():
        assert sha(OUT/'config.json') == config_sha
        assert (p.ROOT/Path(p.read(ACTIVE_RUN)['run_dir'])).resolve() == OUT
        assert all(sha(p.code_path(name)) == digest for name, digest in plan['code_sha256'].items())
        readiness = p.read(OUT/'environment/training_readiness.json')
        assert readiness['passed'] and readiness['config_sha256'] == config_sha
        from storymaker.rlaif.code.rlaif_reward_pipeline import accepted_protocol
        accepted_protocol(config)

    def status(stage, **extra):
        save(folder/'cycle_state.json', dict(stage=stage, at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), automatic_restart=False, **extra))

    def wait_idle():
        consecutive = 0
        while enabled():
            guard()
            try:
                script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'storymaker.*rlaif' -and $_.CommandLine -notmatch 'storymaker.rlaif.code.(rlaif_cycle|rl_pilot_monitor)' } | Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"
                raw = subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command', script], text=True, encoding='utf-8', timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
                workers = json.loads(raw) if raw.strip() else []
                ready, gpu = gpu_ready()
                consecutive = consecutive+1 if not workers and ready and lock_free(GPU_LOCK) and lock_free(lock_path(OUT)) else 0
                if consecutive >= 2:
                    return
                status('waiting_idle', native_workers=workers, gpu=gpu, ready_observations=consecutive)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
                consecutive = 0
                status('inspection_error_no_restart', error=str(error))
            time.sleep(1 if consecutive else 30)
        raise InterruptedError('Control disabled; no child terminated')

    if resume_from is not None:
        assert execute and resume_from == 'train_reward'
    with (folder/'controller.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if resume_from is None:
            assert not (folder/'cycle_state.json').exists(), 'Inspect any earlier run; never restart automatically'
            tasks = plan['commands']; logs = folder
        else:
            resume_inputs(folder, config_sha, plan['code_sha256'], batch, count)
            tasks = plan['commands'][2:]
            assert tasks[0][0] == resume_from
            logs = folder/'resumed_train_reward'/sha(folder/'resume_authorization.json')[:16]
            logs.mkdir(parents=True, exist_ok=True)
        try:
            for label, worker, args in tasks:
                wait_idle()
                guard()
                if label.endswith('reward'):
                    assert p.read(OUT/'api_pricing_check.json')['checked_date_kst'] == datetime.now(timezone(timedelta(hours=9))).date().isoformat(), 'Verify official pricing for this date before another paid phase'
                command = p.worker_command(worker, *args, python=config['python'])
                with (logs/(label+'.stdout.log')).open('xb') as out, (logs/(label+'.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=p.ROOT.parent, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    status('running_child', phase=label, wrapper_pid=child.pid, control_enabled=enabled(), logs_dir=str(logs))
                    time.sleep(15)
                assert child.returncode == 0, label+' failed; preserve logs, no automatic restart'
                guard()
                status('phase_complete', phase=label)
            report = compare(config)
            status('round_complete_for_agent_review', comparison='evaluation/consecutive_comparison.json', N=report['N'], next='Review product quality, failures, cost and measured time before any further cycle or final model selection.')
        except BaseException as error:
            status('needs_attention' if enabled() else 'stopped_by_control', error=type(error).__name__, message=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-from', choices=['train_reward'])
    args = parser.parse_args()
    main(args.execute, args.resume_from)
