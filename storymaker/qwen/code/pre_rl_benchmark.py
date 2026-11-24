"""Frozen zero-shot KoBEST + a preselected Korean CommonGen sample. No training/API."""
import argparse
import contextlib
import json
import math
import msvcrt
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import evaluation_generate as g
def read_lines(path):
    return [json.loads(line) for line in path.read_text('utf-8-sig').splitlines() if line.strip()]


def choices_for(task, row):
    if task == 'copa':
        connector = {'원인': ' 왜냐하면', '결과': ' 그래서'}[row['question'].strip()]
        return f"{row['premise']} {connector}", [row['alternative_1'], row['alternative_2']]
    return f"문장: {row['context']}", [row[f'ending_{i}'] for i in range(1, 5)]


def prepare(out):
    sources = out / 'sources'
    copa = read_lines(sources / 'kobest_copa_test.jsonl')
    hella = read_lines(sources / 'kobest_hellaswag_test.jsonl')
    common = read_lines(sources / 'korean_commongen_official_test.json')
    assert (len(copa), len(hella), len(common)) == (1000, 500, 2040)
    sample = sorted(random.Random(42).sample(range(len(common)), 100))
    config = {
        'sources': {path.name: g.digest(path) for path in sorted(sources.iterdir()) if path.is_file()},
        'copa_count': 1000, 'hellaswag_count': 500,
        'commongen_total': 2040, 'commongen_sample_indices': sample,
        'variants': ['base', 'sft'], 'zero_shot': True,
        'kobest_prompt': 'Plain continuation using official lm-evaluation-harness KoBEST utils; no chat template',
        'choice_scoring': 'sum continuation log likelihood; additional character-length normalization, no EOS scoring',
        'choice_delimiter': ' ',
        'commongen_system': '한국어 문장 생성 과제를 수행하세요. 설명이나 JSON 없이 문장만 출력하세요.',
        'commongen_prompt': '다음 개념어들을 자연스럽게 활용하여 상식에 맞는 한국어 문장 하나를 쓰세요.\n개념어: {concepts}',
        'generation': {'do_sample': False, 'max_new_tokens': 128, 'thinking': False, 'seed': 42},
        'license': {
            'kobest': 'CC-BY-SA-4.0, SK Telecom dataset card',
            'commongen': 'Author-public test file for paper evaluation; repository has no explicit data license. Local academic evaluation only, do not redistribute or train on it.'
        },
        'urls': {
            'kobest': 'https://huggingface.co/datasets/skt/kobest_v1',
            'harness': 'https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/kobest',
            'commongen': 'https://github.com/J-Seo/Korean-CommonGen'
        },
        'limitations': 'Custom local evaluator reproduces specified prompts/scoring; not an official leaderboard submission. CommonGen100 is a fixed diagnostic subset, not the full benchmark. Pretraining contamination is unknown.'
    }
    path = out / 'benchmark_config.json'
    if path.exists():
        assert p.read(path) == config, 'Benchmark configuration changed'
    else:
        p.save(path, config)
    return config, {'copa': copa, 'hellaswag': hella, 'commongen': common}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-generation-device-fix', type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config, data = prepare(out)
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'counts': [1000, 500, 100], 'variants': 2}))
        return
    started = time.time()

    def status(state, **extra):
        p.save(out / 'status.json', {'state': state, 'pid': os.getpid(),
               'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time() - started, **extra})

    try:
        status('preflight')
        run = args.run_dir.resolve()
        g.completed_context(run)
        train_config = p.read(run / 'config.json')
        protected = p.read(run / 'watchdog/protected_sha256.json')
        assert all(g.digest(path) == sha for path, sha in protected.items())
        manifest = {'benchmark_config_sha256': g.digest(out / 'benchmark_config.json'),
                    'code_sha256': g.digest(__file__), 'sft_config_sha256': g.digest(run / 'config.json'),
                    'adapter_sha256': g.digest(run / 'final_refit/adapter/adapter_model.safetensors'),
                    'model_revision': train_config['model_revision'], 'optimizer_updates': 0, 'api_calls': 0}
        if (out / 'manifest.json').exists():
            original_manifest = p.read(out / 'manifest.json')
            if original_manifest != manifest:
                assert args.resume_generation_device_fix, 'Code changed without explicit reviewed recovery'
                receipt = p.read(args.resume_generation_device_fix)
                assert receipt['original_code_sha256'] == original_manifest['code_sha256']
                assert receipt['commongen_outputs_before_fix'] == 0
                assert g.digest(out / 'choice_rows.json') == receipt['choice_rows_sha256']
                assert len(p.read(out / 'choice_rows.json')) == 3000
                assert {k: v for k, v in original_manifest.items() if k != 'code_sha256'} == {k: v for k, v in manifest.items() if k != 'code_sha256'}
                manifest['choice_source_manifest_sha256'] = g.digest(out / 'manifest.json')
                manifest['recovery_receipt_sha256'] = g.digest(args.resume_generation_device_fix)
                p.save(out / 'generation_manifest.json', manifest)
        else:
            p.save(out / 'manifest.json', manifest)
        with (p.ROOT / 'outputs/generation.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            from storymaker.qwen.code.sft_v5_train import chunked_ce, gpu_ready
            ready = gpu_ready()
            if not ready['ready']:
                raise RuntimeError(f'GPU not ready: {ready}')
            import torch
            from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(train_config['model'], local_files_only=True)
            status('loading_model')
            base = AutoModelForImageTextToText.from_pretrained(train_config['model'], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').eval()
            base.model.language_model.to('cuda')
            base.lm_head.to('cuda')
            model = PeftModel.from_pretrained(base, run / 'final_refit/adapter').eval()
            rows_path = out / 'choice_rows.json'
            rows = p.read(rows_path) if rows_path.exists() else []
            done = {(r['variant'], r['task'], r['index']) for r in rows}
            assert len(done) == len(rows)
            for variant in ('base', 'sft'):
                guard = model.disable_adapter() if variant == 'base' else contextlib.nullcontext()
                with guard, torch.inference_mode():
                    for task in ('copa', 'hellaswag'):
                        for index, record in enumerate(data[task]):
                            if (variant, task, index) in done:
                                continue
                            prefix, options = choices_for(task, record)
                            context_ids = tokenizer.encode(prefix, add_special_tokens=False)
                            scores = []
                            for option in options:
                                full = tokenizer.encode(prefix + ' ' + option, add_special_tokens=False)
                                assert full[:len(context_ids)] == context_ids, 'Tokenizer continuation boundary changed'
                                labels = [-100] * len(context_ids) + full[len(context_ids):]
                                tokens = torch.tensor([full], device='cuda')
                                targets = torch.tensor([labels], device='cuda')
                                hidden = base.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                                ce = chunked_ce(hidden.last_hidden_state[:, :-1], base.lm_head.weight, targets[:, 1:], 128)
                                scores.append(-ce.item() * (len(full) - len(context_ids)))
                                del hidden, ce, tokens, targets
                            pred = max(range(len(options)), key=lambda i: scores[i])
                            pred_norm = max(range(len(options)), key=lambda i: scores[i] / len(options[i]))
                            row = {'variant': variant, 'task': task, 'index': index, 'gold': int(record['label']),
                                   'log_likelihoods': scores, 'prediction': pred, 'prediction_char_norm': pred_norm,
                                   'correct': pred == int(record['label']), 'correct_char_norm': pred_norm == int(record['label'])}
                            rows.append(row)
                            p.save(rows_path, rows)
                            print(json.dumps(row), flush=True)
                            status('running_choice', variant=variant, task=task, completed=len(rows), total=3000)
                            torch.cuda.empty_cache()
            generations = out / 'commongen'
            generations.mkdir(exist_ok=True)
            # generate prepares cache/input placement from the complete model's device.
            model.to('cuda')
            assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
            for variant in ('base', 'sft'):
                guard = model.disable_adapter() if variant == 'base' else contextlib.nullcontext()
                with guard, torch.inference_mode():
                    for index in config['commongen_sample_indices']:
                        target = generations / f'{variant}_{index}.json'
                        if target.exists():
                            prior = p.read(target)
                            assert prior['manifest'] == g.signature(manifest)
                            continue
                        item = data['commongen'][index]
                        prompt = config['commongen_prompt'].format(concepts=item['concept-set'].replace('#', ', '))
                        messages = [{'role': 'system', 'content': config['commongen_system']}, {'role': 'user', 'content': prompt}]
                        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda')
                        set_seed(42)
                        before = time.perf_counter()
                        output = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, use_cache=True)
                        ids = output[0, inputs['input_ids'].shape[1]:].tolist()
                        row = {'index': index, 'variant': variant, 'manifest': g.signature(manifest),
                               'concepts': item['concept-set'], 'references': item['scene'],
                               'raw_text': tokenizer.decode(ids, skip_special_tokens=True), 'token_ids': ids,
                               'eos': tokenizer.eos_token_id in ids, 'seconds': time.perf_counter() - before}
                        p.save(target, row)
                        print(json.dumps({'variant': variant, 'task': 'commongen', 'index': index, 'eos': row['eos'], 'seconds': row['seconds']}), flush=True)
                        status('running_commongen', variant=variant, index=index,
                               completed=len(list(generations.glob('*.json'))), total=200)
                        del inputs, output
                        torch.cuda.empty_cache()
            assert all(g.digest(path) == sha for path, sha in protected.items())
            groups = {}
            for variant in ('base', 'sft'):
                for task in ('copa', 'hellaswag'):
                    group = [r for r in rows if r['variant'] == variant and r['task'] == task]
                    assert len(group) == len(data[task])
                    groups[f'{variant}/{task}'] = {'n': len(group),
                        'accuracy': sum(r['correct'] for r in group) / len(group),
                        'accuracy_char_norm': sum(r['correct_char_norm'] for r in group) / len(group)}
            p.save(out / 'summary.json', {'groups': groups, 'commongen_outputs': 200,
                   'protected_files_unchanged': len(protected), 'elapsed_seconds': time.time() - started})
            status('complete')
    except BaseException as error:
        status('failed', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
