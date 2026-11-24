"""30s observation only for the RL pilot. No automatic termination or restart."""
import csv
import argparse
import json
import msvcrt
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from storymaker import project as p
OUT = p.ROOT / 'outputs/rl_pilot_20261114'


def memory_alerts(value):
    alerts = []
    if (value.get('ram') or {}).get('free_kib', 2**31) < 2*1024**2:
        alerts.append('free_RAM<2GiB_review_only')
    shared = sum(int(row['SharedUsage']) for row in value.get('process_gpu_memory', []))
    if shared > 2*1024**3:
        alerts.append('GPU_shared_memory>2GiB_review_only')
    if value.get('disk_free_gib', 100) < 40:
        alerts.append('workspace_free_disk<40GiB_review_only')
    if value.get('memory_inspection_error'):
        alerts.append('memory_inspection_error_no_restart')
    return alerts


def save_observation(path, value):
    for attempt in range(6):
        try:
            return p.save(path, value)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05*2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=p.Path, default=OUT)
    args = parser.parse_args()
    folder = args.run_dir / 'monitor'
    folder.mkdir(exist_ok=True)
    with (folder / 'monitor.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        while p.read(folder / 'control.json')['enabled']:
            value = {'at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
                     'alerts':[], 'automatic_restart':False,'automatic_termination':False}
            try:
                raw = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.free,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15)
                value['gpu'] = dict(zip(['utilization','temperature_c','used_mib','free_mib','total_mib','power_w'],map(float,next(csv.reader(raw.splitlines())))))
                if value['gpu']['temperature_c'] >= 85:
                    value['alerts'].append('temperature>=85C_review_only')
                script = r"""[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$rows=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(rl_pilot|rlaif_|reward_rag)[^ ]*(\.py)?' } | Select-Object ProcessId,ParentProcessId,CommandLine)
$memory=@(); $memoryError=$null; $ram=$null
try {
  $osInfo=Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
  $ram=@{total_kib=[long]$osInfo.TotalVisibleMemorySize;free_kib=[long]$osInfo.FreePhysicalMemory}
  $jobIds=@($rows | ForEach-Object { [int]$_.ProcessId })
  $memory=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop | Where-Object { $_.Name -match 'pid_(\d+)_' -and $jobIds -contains [int]$Matches[1] } | Select-Object Name,DedicatedUsage,SharedUsage)
} catch { $memoryError=$_.Exception.Message }
@{processes=$rows;ram=$ram;process_gpu_memory=$memory;memory_inspection_error=$memoryError} | ConvertTo-Json -Depth 4 -Compress"""
                observed = json.loads(subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',script],text=True,encoding='utf-8',timeout=25))
                value.update(observed)
                value['disk_free_gib'] = shutil.disk_usage(args.run_dir).free/2**30
                value['alerts'].extend(memory_alerts(value))
                value['inspection_error'] = None
            except Exception as error:
                value['inspection_error'] = str(type(error).__name__)+': '+str(error)
                value['alerts'].append('inspection_error_no_restart')
            save_observation(folder / 'state.json',value)
            with (folder / 'samples.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(value,ensure_ascii=False)+'\n')
            time.sleep(30)
        save_observation(folder / 'stopped.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()})


if __name__ == '__main__':
    main()
