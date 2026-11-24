"""Six development-only checkpoint rollouts to diagnose CE/generation divergence. No retries."""
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import evaluation_generate as g
from storymaker.qwen.code import watch_sft as w
from storymaker.qwen.code import pre_rl_gpu
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-rag-index', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    run = Path(p.read(p.ROOT.parent / 'outputs/sft_v5/active_run.json')['run_dir'])
    if args.job:
        from storymaker.qwen.code import generate_story
        job = p.read(args.job)
        p.RUN = run
        value = generate_story.generate(job['moral'], Path(job['adapter']))
        p.save(job['output'], {'job': job, 'generation': value})
        return
    out = p.ROOT / 'outputs/pre_rl_20261114/checkpoint_rollouts'
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(state, **extra):
        p.save(out / 'status.json', {'state': state, 'pid': os.getpid(),
               'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time() - started, **extra})

    try:
        if args.wait_for_rag_index:
            status('waiting_for_rag_index')
            rag = out.parent / 'rag'
            while True:
                pre_rl_gpu.enabled()
                if w.lock_held(p.ROOT / 'outputs/generation.lock'):
                    time.sleep(15)
                    continue
                if (rag / 'index_manifest.json').exists():
                    assert p.read(rag / 'status.json')['state'] == 'index_complete_pending_api_validation'
                    break
                if (rag / 'status.json').exists() and p.read(rag / 'status.json')['state'] == 'failed':
                    raise RuntimeError('RAG build failed; no automatic retry')
                time.sleep(15)
        g.completed_context(run)
        config = p.read(run / 'config.json')
        books = {r['source_id']: r for r in p.read(p.PROCESSED / 'dataset.json')}
        ids = [205, 143, 9]
        assert all(books[n]['source_split'] != 'test' for n in ids)
        adapters = {'epoch1': run / 'final_refit/checkpoint-61', 'epoch3': run / 'final_refit/checkpoint-183'}
        manifest = {'code_sha256': g.digest(__file__),
                    'generation_code_sha256': g.digest(p.ROOT / 'storymaker.qwen.code.generate_story.py'),
                    'generation_manifest_identity': '66628398d1911f7bcff60e58d4bf3550f6dee33973cea80af057700c215ddabe',
                    'source_ids': ids, 'adapters': {k: g.digest(v / 'adapter_model.safetensors') for k, v in adapters.items()},
                    'purpose': 'Post-hoc development3 diagnostic, not representative quality estimate or epoch reselection',
                    'comparison': 'Reuse existing frozen pilot base/final outputs; generate epoch1/3 once each',
                    'optimizer_updates': 0, 'api_calls': 0, 'test_generation': 0}
        if (out / 'manifest.json').exists():
            assert p.read(out / 'manifest.json') == manifest
        else:
            p.save(out / 'manifest.json', manifest)
        rows = []
        with (p.ROOT / 'outputs/generation.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            for variant, adapter in adapters.items():
                for number in ids:
                    target = out / f'{variant}_{number}.json'
                    job = {'variant': variant, 'source_id': number, 'moral': books[number]['input']['moral'],
                           'adapter': str(adapter), 'output': str(target), 'manifest': g.signature(manifest)}
                    if target.exists():
                        value = p.read(target)
                        assert value['job'] == job
                    else:
                        pre_rl_gpu.wait_ready(status)
                        job_path = out / f'{variant}_{number}.job.json'
                        p.save(job_path, job)
                        command = [config['python'], '-u', str(Path(__file__).resolve()), '--job', str(job_path)]
                        status('generating', variant=variant, source_id=number, completed=len(rows), total=6)
                        with (out / f'{variant}_{number}.stdout.log').open('w', encoding='utf-8') as stdout, (out / f'{variant}_{number}.stderr.log').open('w', encoding='utf-8') as stderr:
                            subprocess.run(command, stdout=stdout, stderr=stderr, check=True,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                        value = p.read(target)
                    generated = value['generation']
                    rows.append({'variant': variant, 'source_id': number, 'eos': generated['ended_with_eos'],
                                 'tokens': generated['generated_tokens'], 'format_problems': generated['format_problems'],
                                 'seconds': generated['generation_seconds'], 'output': str(target)})
                    p.save(out / 'summary_rows.json', rows)
                    print(json.dumps(rows[-1]), flush=True)
        assert len(rows) == 6
        status('complete', completed=6, total=6)
    except BaseException as error:
        status('failed', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
