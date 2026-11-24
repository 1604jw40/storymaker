"""Run one configured direct-RLAIF batch after collection/scoring exit; no automatic restart."""
import argparse
import hashlib
import json
import msvcrt
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, validate_settings, preserve_batch_config
from storymaker.rlaif.code.rlaif_budget import lock_path


def read_shared(path):
    # Close the Windows handle before parsing or waiting; never keep a job's status open.
    raw = Path(path).read_bytes()
    return json.loads(raw)


def validate_receipt(value, phase, config_sha256, count):
    assert value['config_sha256'] == config_sha256, 'Receipt belongs to another configuration'
    if phase == 'probability_equivalence':
        assert value['passed'] and len(value['results']) == 2
    elif phase.startswith('prepare-'):
        assert value['phase'] == phase.split('-')[1] and value['count'] == count and value['source_verified']
    else:
        assert value['phase'] == phase.split('-')[1] and value['trajectories'] == count
        assert value['optimizer_steps'] == 1 and value['saved_and_readback_verified']


def save(path, value):
    for attempt in range(6):
        try:
            return p.save(path, value)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def host_processes():
    script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $rows=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(rlaif_generate|rlaif_reward_api|rlaif_probability_probe|rlaif_train)(\\.py|\\s|$)' } | Select-Object ProcessId,ParentProcessId,CommandLine); ConvertTo-Json -InputObject $rows -Compress"
    return json.loads(subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                      text=True, encoding='utf-8', timeout=20))


def lock_free(path):
    with Path(path).open('a+b') as stream:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return True


def gpu_ready():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,temperature.gpu',
                                  '--format=csv,noheader,nounits'], text=True, timeout=15)
    free, temperature = map(float, raw.strip().split(','))
    return free > 13500 and temperature < 80, {'free_mib': free, 'temperature_c': temperature}


def main(execute=False):
    config = read_shared(OUT / 'config.json')
    count = len(validate_settings(config))
    batch = OUT / f"batch_{config.get('batch_index', 0):04d}"
    destination = OUT / 'pipeline'
    destination.mkdir(exist_ok=True)
    code_files = ['rlaif_probability_probe.py', 'rlaif_train.py', 'rlaif_loss.py', 'rlaif_storage.py',
                  'rlaif_generate.py', 'rlaif_reward_api.py', 'rlaif_sequence.py', 'rlaif_environment.py',
                  'generate_story.py', 'project.py', 'rlaif_precision.py', 'rlaif_budget.py', 'reward_rag.py', 'rlaif_temporal_review.py', 'rlaif_training_report.py']
    if config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
        code_files.append('rlaif_reward_pipeline.py')
    code_hashes = {name: hashlib.sha256(p.code_path(name).read_bytes()).hexdigest() for name in code_files}
    config_sha256 = hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()
    plan_path = destination / 'manifest.json'
    if plan_path.exists():
        assert read_shared(plan_path)['code_sha256'] == code_hashes, 'Code changed; inspect before a controlled continuation'
        assert read_shared(plan_path)['config_sha256'] == config_sha256
    else:
        save(plan_path, {'code_sha256': code_hashes, 'config_sha256': config_sha256,
            'scope': f'Configured batch of{count}; one committed policy/value update then matched validation43',
            'automatic_restart': False})
    if not execute:
        print(json.dumps({'plan':str(plan_path),'batch_size':count,'update_index':config['update_index'],
            'policy_checkpoint':config['policy_checkpoint'],'value_checkpoint':config['value_checkpoint'],
            'execute':False}))
        return
    assert p.read(OUT/'environment/training_readiness.json')['passed']

    def status(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at_utc': datetime.now(timezone.utc).isoformat(),
                 'automatic_restart': False, **extra}
        save(destination / 'state.json', value)
        with (destination / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')
        print(stage, extra, flush=True)

    def enabled():
        return read_shared(OUT/'control.json')['enabled'] and read_shared(destination/'control.json')['enabled']

    def assert_code_unchanged():
        assert hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest() == config_sha256
        assert all(hashlib.sha256(p.code_path(name).read_bytes()).hexdigest() == expected for name,expected in code_hashes.items())

    try:
        with (destination / 'sequence.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            consecutive = 0
            while enabled():
                try:
                    processes = host_processes()
                    generation = read_shared(batch/'status.json')
                    scoring = read_shared(batch/'api_status.json') if (batch/'api_status.json').exists() else {'completed':0}
                    generation_live = any('rlaif_generate' in row['CommandLine'] for row in processes)
                    scoring_live = any('rlaif_reward_api' in row['CommandLine'] for row in processes)
                    complete = generation['stage']=='batch_generation_complete' and scoring.get('stage')=='complete'
                    if not generation_live and generation.get('completed',0)<count:
                        raise RuntimeError('Generation exited before batch completion; inspect, do not restart')
                    if not scoring_live and scoring.get('completed',0)<count:
                        raise RuntimeError('API worker exited before scoring completion; inspect preserved responses')
                    ready, snapshot = gpu_ready()
                    free = lock_free(GPU_LOCK) and lock_free(lock_path(OUT))
                    consecutive = consecutive+1 if complete and not processes and free and ready else 0
                    status('waiting_fixed_batch', generation=generation.get('completed',0), scoring=scoring.get('completed',0),
                           processes=processes, gpu=snapshot, released_locks=free, ready_observations=consecutive)
                    if consecutive>=2:
                        break
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, OSError) as error:
                    # Observation failure is not proof a process stopped. Keep observing the same jobs.
                    consecutive=0
                    status('inspection_error_no_restart', error=type(error).__name__, message=str(error))
                time.sleep(30)
            else:
                status('stopped_by_control')
                return
            assert_code_unchanged()
            preserve_batch_config(OUT,config)
            if config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
                from storymaker.rlaif.code.rlaif_reward_pipeline import accepted_protocol
                accepted_protocol(config)
            frozen = read_shared(OUT/'reward_protocol_frozen.json')
            assert frozen['api_code_sha256'] == code_hashes['rlaif_reward_api.py']
            if frozen.get('temporal_review_required'):
                assert frozen['temporal_review_module_sha256']==code_hashes['rlaif_temporal_review.py']
            steps = [('probability_equivalence', ['rlaif_probability_probe.py'], OUT/'environment/probability_equivalence/verified.json')]
            for phase in ('prepare-policy','prepare-reference','prepare-value','update-policy','update-value'):
                role=phase.split('-')[1]
                receipt=batch/('prepared/'+role+'/verified.json') if phase.startswith('prepare') else batch/('update_'+role+'_verified.json')
                steps.append((phase,['rlaif_train.py','--phase',phase],receipt))
            for name, arguments, receipt in steps:
                if not enabled():
                    status('stopped_by_control', next_phase=name)
                    return
                assert_code_unchanged()
                if name == 'update-policy' and config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
                    from storymaker.rlaif.code.rlaif_learning_signal import inspect
                    assert inspect(batch,config)['passed'], 'Fresh critic is worse than zero; inspect saved learning signal before any update'
                ready,snapshot=gpu_ready()
                assert ready and lock_free(GPU_LOCK)
                assert not host_processes(), 'Existing worker: never start a duplicate'
                if receipt.exists():
                    # A previous controller may have exited after the worker committed its receipt.
                    checked=read_shared(receipt)
                    validate_receipt(checked, name, config_sha256, count)
                    status('verified_phase_reused',phase=name,receipt=str(receipt))
                    continue
                command=p.worker_command(arguments[0],*arguments[1:],python=config['python'])
                with (destination/(name+'.stdout.log')).open('ab') as stdout,(destination/(name+'.stderr.log')).open('ab') as stderr:
                    child=subprocess.Popen(command,cwd=p.ROOT.parent,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                status('running_phase',phase=name,worker_pid=child.pid,gpu=snapshot)
                while child.poll() is None:
                    status('running_phase',phase=name,worker_pid=child.pid)
                    time.sleep(30)
                assert child.returncode==0 and receipt.exists(), 'Worker failed; inspect logs, no automatic restart'
                result=read_shared(receipt)
                validate_receipt(result, name, config_sha256, count)
                status('phase_completed',phase=name,receipt=str(receipt))
            policy=read_shared(batch/'update_policy_verified.json');value=read_shared(batch/'update_value_verified.json')
            assert policy['optimizer_steps']==value['optimizer_steps']==1
            assert policy['trajectories']==value['trajectories']==count
            save(batch/'joint_update_verified.json',{'policy':policy['checkpoint'],'value':value['checkpoint'],
                 'committed_policy_updates':1,'committed_value_updates':1,'trajectories':count,
                 'update_index':config['update_index'],
                 'next':'Matched development evaluation, not final quality certification',
                 'pipeline_manifest_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest()})
            status('joint_update_complete',committed_policy_updates=1,committed_value_updates=1)
    except BaseException as error:
        status('needs_attention',error=type(error).__name__,message=str(error),automatic_restart=False)
        traceback.print_exc()
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
