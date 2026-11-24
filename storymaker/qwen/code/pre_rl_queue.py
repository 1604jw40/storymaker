"""Continue a validated GPU benchmark after CE exits. No retries, training or process termination."""
import json
import msvcrt
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import watch_sft as w
def main():
    out = p.ROOT / 'outputs/pre_rl_20261114'
    folder = out / 'queue'
    folder.mkdir(parents=True, exist_ok=True)
    control = folder / 'control.json'
    if not control.exists():
        p.save(control, {'enabled': True})

    def status(state, **extra):
        p.save(folder / 'status.json', {'state': state, 'pid': os.getpid(),
               'at_utc': datetime.now(timezone.utc).isoformat(), **extra})

    with (folder / 'queue.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        status('waiting_for_ce')
        while p.read(control)['enabled']:
            if w.lock_held(p.ROOT / 'outputs/generation.lock'):
                time.sleep(15)
                continue
            state = p.read(out / 'ce/status.json')['state']
            if state == 'failed':
                status('needs_attention', reason='CE failed. No automatic retry.')
                return
            if state == 'complete' and not w.lock_held(p.ROOT / 'outputs/generation.lock'):
                break
            time.sleep(15)
        else:
            status('stopped_by_control')
            return
        from storymaker.qwen.code.sft_v5_train import gpu_ready
        status('waiting_for_gpu_ready')
        while p.read(control)['enabled']:
            ready = gpu_ready()
            temp = float(subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
            if ready['ready'] and temp < 80:
                break
            status('waiting_for_gpu_ready', gpu=ready, temperature_c=temp)
            time.sleep(15)
        else:
            status('stopped_by_control')
            return
        active = p.read(p.ROOT.parent / 'outputs/sft_v5/active_run.json')['run_dir']
        command = [sys.executable, '-u', str(p.ROOT / 'storymaker.qwen.code.pre_rl_benchmark.py'),
                   '--run-dir', active, '--output-dir', str(out / 'benchmarks')]
        status('running_benchmark', command=command)
        p.save(folder / 'transition.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
               'from': 'M1_ce', 'to': 'M2_benchmark', 'readme_figma_sync_pending': True})
        with (out / 'benchmarks/stdout.log').open('w', encoding='utf-8') as stdout, (out / 'benchmarks/stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            status('running_benchmark', child_pid=process.pid, command=command)
            result = process.wait()
        if result:
            status('needs_attention', reason='Benchmark failed. No automatic retry.', returncode=result)
        else:
            assert p.read(out / 'benchmarks/status.json')['state'] == 'complete'
            status('complete', next='Review results and run generation metrics/RAG stage. No RL.')


if __name__ == '__main__':
    main()
