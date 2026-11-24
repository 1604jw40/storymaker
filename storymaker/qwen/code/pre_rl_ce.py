"""Read-only base/checkpoint CE diagnostics. Never trains or changes the frozen test protocol."""
import argparse
import contextlib
import hashlib
import json
import math
import msvcrt
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import evaluation_generate as g
def aggregate(rows):
    result = {}
    for variant in sorted({r['variant'] for r in rows}):
        for split in ('development', 'test'):
            group = [r for r in rows if r['variant'] == variant and r['split'] == split]
            if not group:
                continue
            count = sum(r['supervised_tokens'] for r in group)
            ce = sum(r['ce'] * r['supervised_tokens'] for r in group) / count
            result[f'{variant}/{split}'] = {
                'stories': len(group), 'supervised_tokens': count, 'token_weighted_ce': ce,
                'conditional_perplexity': math.exp(ce),
                'story_mean_ce': sum(r['ce'] for r in group) / len(group),
                'gold_prefix_eos_mean_nll': sum(r['eos_nll'] for r in group) / len(group),
                'gold_prefix_eos_top1_count': sum(r['eos_rank'] == 1 for r in group),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(state, **extra):
        p.save(out / 'status.json', {'state': state, 'pid': os.getpid(),
               'updated_at_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.time() - started, **extra})

    try:
        status('preflight')
        run = args.run_dir.resolve()
        context = g.completed_context(run)
        config = p.read(run / 'config.json')
        protected = p.read(run / 'watchdog/protected_sha256.json')
        assert all(g.digest(path) == sha for path, sha in protected.items())
        with (p.ROOT / 'outputs/generation.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            from storymaker.qwen.code.sft_v5_train import chunked_ce, gpu_ready
            ready = gpu_ready()
            if not ready['ready']:
                raise RuntimeError(f'GPU is not ready: {ready}')
            import torch
            from torch.nn import functional as F
            from transformers import AutoModelForImageTextToText, AutoTokenizer
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
            encoded_dev = {r['source_id']: r for r in p.read(run / 'encoded_development.json')}
            records = []
            for split in ('train', 'validation', 'test'):
                for line in (p.PROCESSED / f'{split}.jsonl').read_text('utf-8-sig').splitlines():
                    row = json.loads(line)
                    messages = row['messages']
                    prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids']
                    full = tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=False, enable_thinking=False, return_dict=True)['input_ids']
                    assert full[:len(prefix)] == prefix
                    labels = [-100] * len(prefix) + full[len(prefix):]
                    assert labels.count(tokenizer.eos_token_id) == 1
                    if split != 'test':
                        assert full == encoded_dev[row['source_id']]['input_ids']
                        assert labels == encoded_dev[row['source_id']]['labels']
                    records.append({'source_id': row['source_id'], 'split': 'test' if split == 'test'
                                    else 'development', 'input_ids': full, 'labels': labels})
            records.sort(key=lambda r: (r['split'], r['source_id']))
            assert len(records) == len({r['source_id'] for r in records}) == 304
            adapters = {'epoch1': run / 'final_refit/checkpoint-61',
                        'epoch3': run / 'final_refit/checkpoint-183',
                        'epoch5': run / 'final_refit/adapter'}
            manifest = {'run_dir': str(run), 'code_sha256': g.digest(__file__),
                        'config_sha256': g.digest(run / 'config.json'),
                        'data_sha256': config['dataset_sha256'],
                        'adapters': {key: g.digest(path / 'adapter_model.safetensors') for key, path in adapters.items()},
                        'variants': ['base', *adapters], 'records_per_variant': 304,
                        'selection_changes': False, 'optimizer_updates': 0, 'api_calls': 0,
                        'scope': 'development is in-sample; test61 already used for diagnosis; checkpoint comparisons are post-hoc, not new selection',
                        'ce_definition': 'sum assistant JSON plus template ending token NLL / supervised tokens; no truncation',
                        'eos_definition': 'EOS NLL/rank conditioned on complete gold answer; not free-generation termination probability'}
            if (out / 'manifest.json').exists():
                assert p.read(out / 'manifest.json') == manifest, 'Diagnostic manifest changed'
            else:
                p.save(out / 'manifest.json', manifest)
            rows = p.read(out / 'rows.json') if (out / 'rows.json').exists() else []
            completed = {(r['variant'], r['source_id']) for r in rows}
            assert len(completed) == len(rows)
            status('loading_model', completed=len(rows), total=1216)
            base = AutoModelForImageTextToText.from_pretrained(config['model'], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').eval()
            base.model.language_model.to('cuda')
            base.lm_head.to('cuda')
            assert next(base.model.visual.parameters()).device.type == 'cpu'
            model = PeftModel.from_pretrained(base, adapters['epoch1'], adapter_name='epoch1')
            for key in ('epoch3', 'epoch5'):
                model.load_adapter(adapters[key], adapter_name=key)
            for variant in manifest['variants']:
                if variant != 'base':
                    model.set_adapter(variant)
                model.eval()
                guard = model.disable_adapter() if variant == 'base' else contextlib.nullcontext()
                with guard, torch.inference_mode():
                    for record in records:
                        if (variant, record['source_id']) in completed:
                            continue
                        before = time.perf_counter()
                        tokens = torch.tensor([record['input_ids']], device='cuda')
                        labels = torch.tensor([record['labels']], device='cuda')
                        result = base.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                        ce = chunked_ce(result.last_hidden_state[:, :-1], base.lm_head.weight, labels[:, 1:], 128)
                        position = record['labels'].index(tokenizer.eos_token_id)
                        logits = F.linear(result.last_hidden_state[:, position - 1], base.lm_head.weight).float()
                        eos_nll = -F.log_softmax(logits, -1)[0, tokenizer.eos_token_id].item()
                        row = {'variant': variant, 'split': record['split'], 'source_id': record['source_id'],
                               'ce': ce.item(), 'supervised_tokens': int((labels[:, 1:] != -100).sum()),
                               'eos_nll': eos_nll, 'eos_probability': math.exp(-eos_nll),
                               'eos_rank': int((logits[0] > logits[0, tokenizer.eos_token_id]).sum()) + 1,
                               'seconds': time.perf_counter() - before,
                               'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                               'reserved_mib': torch.cuda.memory_reserved() / 2**20}
                        assert math.isfinite(row['ce']) and math.isfinite(eos_nll)
                        rows.append(row)
                        p.save(out / 'rows.json', rows)
                        status('running', variant=variant, source_id=record['source_id'],
                               completed=len(rows), total=1216, latest=row)
                        print(json.dumps(row), flush=True)
                        del result, ce, logits, tokens, labels
                        torch.cuda.empty_cache()
            assert all(g.digest(path) == sha for path, sha in protected.items())
            assert len(rows) == 1216
            p.save(out / 'summary.json', {'manifest': manifest, 'groups': aggregate(rows),
                   'protected_files_unchanged': len(protected), 'elapsed_seconds': time.time() - started})
            status('complete', completed=len(rows), total=1216)
    except BaseException as error:
        status('failed', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
