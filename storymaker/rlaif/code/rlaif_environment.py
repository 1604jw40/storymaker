"""Active RLAIF result directory, checked settings and device observations."""
import math
import json
import os
import subprocess
from pathlib import Path
from storymaker import project as p

ACTIVE_RUN = p.ROOT / 'rlaif/configs/active_run.json'
DEFAULT_RUN = p.read(ACTIVE_RUN)['run_dir'] if ACTIVE_RUN.is_file() else 'rlaif/results/local'
OUT = (p.ROOT / Path(os.environ.get('STORYMAKER_RLAIF_RUN_DIR') or DEFAULT_RUN)).resolve()
assert OUT.is_relative_to((p.ROOT / 'rlaif/results').resolve())
GPU_LOCK = p.ROOT / 'rlaif/results/generation.lock'


def preserve_batch_config(root, config):
    raw=(root/'config.json').read_bytes()
    assert json.loads(raw)==config, 'Configuration changed before its batch snapshot'
    index=config.get('batch_index',0)
    assert type(index) is int and index>=0
    target=root/f'batch_{index:04d}/config.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with target.open('xb') as stream:stream.write(raw)
    assert target.read_bytes()==raw, 'Never overwrite an earlier batch configuration'
    return target


def validate_settings(config):
    ids = config.get('batch_ids', config.get('initial_batch_ids'))
    assert isinstance(ids, list) and len(ids) == len(set(ids))
    assert type(config['logical_batch_size']) is int and 0 < len(ids) <= config['logical_batch_size']
    train, validation, test = (set(config[key]) for key in ('train_ids', 'validation_ids', 'excluded_test_ids'))
    assert not (train & validation or train & test or validation & test)
    assert set(ids) <= train
    assert type(config['max_new_tokens']) is int and config['max_new_tokens'] > 0
    for key in ('learning_rate', 'temperature'):
        assert type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] > 0
    assert math.isfinite(config['kl_beta']) and 0 <= config['kl_beta'] < 1
    assert config['top_p'] == 1 and config['top_k'] == 0 and config.get('repetition_penalty', 1) == 1
    assert config['rag_required'] is True
    assert config['value_baseline_bounds'] == [-1, 1]
    return ids


def gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    return dict(zip(('temperature_c', 'used_mib', 'free_mib', 'utilization'), map(float, raw)))
