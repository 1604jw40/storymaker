"""Start gating only: observe busy/hot GPU, never terminate a process."""
import math
import subprocess
import time

from storymaker import project as p
CONTROL = p.ROOT / 'outputs/pre_rl_20261114/control.json'


def enabled():
    if CONTROL.exists() and not p.read(CONTROL)['enabled']:
        raise RuntimeError('New pre-RL work disabled by control; running GPU processes were not terminated')


def ready(snapshot, temperature):
    return bool(snapshot['ready'] and math.isfinite(temperature) and temperature < 80)


def wait_ready(notify):
    from storymaker.qwen.code.sft_v5_train import gpu_ready
    while True:
        enabled()
        snapshot = gpu_ready()
        temperature = float(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            text=True, timeout=15).strip().splitlines()[0])
        if ready(snapshot, temperature):
            return {**snapshot, 'temperature_c': temperature}
        notify('waiting_for_gpu_ready', gpu=snapshot, temperature_c=temperature)
        time.sleep(15)
