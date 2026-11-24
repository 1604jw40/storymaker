"""30-second observational GPU monitor. Never kills or restarts a worker."""
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
OUT = p.ROOT / 'outputs/pre_rl_20261114'


def save(path, value):
    for attempt in range(6):
        try:
            p.save(path, value)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.15)


def main():
    folder = OUT / 'monitor'
    folder.mkdir(parents=True, exist_ok=True)
    control = folder / 'control.json'
    if not control.exists():
        save(control, {'enabled': True, 'interval_seconds': 30})
    script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $r=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'pre_rl_(ce|metrics|benchmark|queue|rollout_probe)\\.py|reward_rag\\.py' } | Select-Object ProcessId,ParentProcessId,CommandLine); ConvertTo-Json -InputObject $r -Compress"
    with (folder / 'monitor.lock').open('a+b') as lock:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        while p.read(control)['enabled']:
            sample = {'at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                      'alerts': [], 'automatic_termination': False, 'automatic_restart': False}
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.free,memory.total,power.draw', '--format=csv,noheader,nounits'], text=True, timeout=15)
                values = next(csv.reader(raw.splitlines()))
                keys = ['utilization_percent', 'temperature_c', 'used_mib', 'free_mib', 'total_mib', 'power_w']
                sample['gpu'] = {key: float(value.strip()) for key, value in zip(keys, values)}
                if sample['gpu']['temperature_c'] >= 85:
                    sample['alerts'].append('temperature_at_least_85C_operator_review_before_next_job')
                proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                                      capture_output=True, text=True, encoding='utf-8', timeout=20, check=True)
                sample['processes'] = json.loads(proc.stdout)
                sample['logs'] = []
                for log in OUT.glob('*/stdout.log'):
                    info = log.stat()
                    sample['logs'].append({'path': str(log), 'bytes': info.st_size,
                                           'age_seconds': time.time() - info.st_mtime})
                for log in OUT.glob('*/stderr.log'):
                    with log.open('rb') as handle:
                        handle.seek(max(0, log.stat().st_size - 8192))
                        tail = handle.read().decode('utf-8', errors='replace')
                    if re.search(r'CUDA error:|OutOfMemoryError|CUDA out of memory|device-side assert|illegal memory access|Traceback', tail):
                        sample['alerts'].append('worker_error:' + str(log))
                sample['inspection_error'] = None
                stages = {'storymaker.qwen.code.pre_rl_ce.py': 'ce', 'storymaker.qwen.code.pre_rl_benchmark.py': 'benchmarks',
                          'storymaker.qwen.code.pre_rl_metrics.py': 'metrics', 'storymaker.rlaif.code.reward_rag.py': 'rag',
                          'storymaker.qwen.code.pre_rl_rollout_probe.py': 'checkpoint_rollouts'}
                commands = ' '.join(r['CommandLine'] for r in sample['processes'])
                active_logs = [OUT / folder / 'stdout.log' for name, folder in stages.items() if name in commands]
                ages = [time.time() - path.stat().st_mtime for path in active_logs if path.exists()]
                if ages and min(ages) > 900 and sample['gpu']['utilization_percent'] > 30:
                    sample['alerts'].append('no_observed_progress_for_15_minutes_review_only')
            except Exception as error:
                sample['inspection_error'] = f'{type(error).__name__}: {error}'
                sample['alerts'].append('inspection_failed_no_restart')
            sample['needs_attention'] = bool(sample['alerts'])
            save(folder / 'state.json', sample)
            with (folder / 'samples.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            if sample['alerts']:
                save(folder / 'needs_attention.json', sample)
            time.sleep(30)
        save(folder / 'stopped.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()})


if __name__ == '__main__':
    main()
