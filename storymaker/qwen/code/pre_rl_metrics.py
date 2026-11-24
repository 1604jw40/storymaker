"""Reference metrics with explicit Korean tokenization, failure denominators and BERT length coverage."""
import argparse
import json
import math
import msvcrt
import os
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import evaluation_generate as g
from storymaker.qwen.code import watch_sft as w
from storymaker.qwen.code import pre_rl_gpu
def words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text), re.UNICODE)


def rouge_l(candidate, reference):
    a, b = words(candidate), words(reference)
    if not a or not b:
        return 0.
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def body(story):
    return '\n'.join([story.get('title', '')] + [s.get('text', '') for s in story.get('scenes', [])]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wait-for-benchmark', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(state, **extra):
        p.save(out / 'status.json', {'state': state, 'pid': os.getpid(),
               'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time() - started, **extra})

    try:
        benchmark = out.parent / 'benchmarks'
        if args.wait_for_benchmark:
            status('waiting_for_benchmark')
            while True:
                pre_rl_gpu.enabled()
                if w.lock_held(p.ROOT / 'outputs/generation.lock'):
                    time.sleep(15)
                    continue
                if (benchmark / 'summary.json').exists():
                    assert p.read(benchmark / 'status.json')['state'] == 'complete'
                    break
                if (benchmark / 'status.json').exists() and p.read(benchmark / 'status.json')['state'] == 'failed':
                    raise RuntimeError('Benchmark failed; metric stage will not retry it')
                time.sleep(15)
        else:
            assert p.read(benchmark / 'status.json')['state'] == 'complete'
        status('preflight')
        run = args.run_dir.resolve()
        g.completed_context(run)
        protected = p.read(run / 'watchdog/protected_sha256.json')
        assert all(g.digest(path) == sha for path, sha in protected.items())
        sys.path.insert(0, str(out.parent / 'dependencies'))
        import sacrebleu
        import bert_score
        from storymaker.qwen.code.evaluation_api import semantic_eligible
        from storymaker.qwen.code.evaluation_report import interval
        config = p.read(run / 'config.json')
        books = {r['source_id']: r for r in p.read(p.PROCESSED / 'dataset.json') if r['source_split'] == 'test'}
        model_path = (p.ROOT / 'qwen/models/xlm-roberta-base')
        assert model_path.is_dir()
        items, invalid = [], []
        for number, book in sorted(books.items()):
            for variant in ('base', 'sft'):
                path = run / f'evaluation/generations/test/{number}/{variant}.json'
                value = p.read(path)['generation']
                valid, parsed = semantic_eligible(value)
                if not valid:
                    invalid.append({'task': 'story61', 'variant': variant, 'source_id': number,
                                    'reason': 'no EOS or unreadable JSON narrative', 'metrics': None})
                    continue
                items.append({'task': 'story61', 'variant': variant, 'source_id': number,
                              'candidate': body(parsed), 'references': [body(book['output'])],
                              'source_sha256': g.digest(path)})
        assert len(items) == 97 and len(invalid) == 25
        for path in sorted((benchmark / 'commongen').glob('*.json')):
            row = p.read(path)
            items.append({'task': 'commongen100', 'variant': row['variant'], 'source_id': row['index'],
                          'candidate': row['raw_text'], 'references': row['references'],
                          'source_sha256': g.digest(path), 'eos': row['eos']})
        assert len(items) == 297
        manifest = {'code_sha256': g.digest(__file__), 'items': items, 'invalid': invalid,
                    'encoder': str(model_path), 'encoder_layer': 9, 'idf': False, 'rescale_with_baseline': False,
                    'encoder_weights_sha256': g.digest(model_path / 'model.safetensors'),
                    'bert_score_version': bert_score.__version__, 'sacrebleu_version': sacrebleu.__version__,
                    'bertscore_scope': 'Explicit prefix of first510 encoder tokens. Full-text score only when every compared text fits. Coverage reported; no claim to score omitted endings.',
                    'chrf': 'chrF++ char_order6 word_order2 beta2; full text',
                    'rouge_l': 'Unicode letter/number word tokens; LCS F1, full text; not English-only default tokenizer',
                    'limitations': 'Reference similarity is not moral correctness or creativity. Story61 valid semantic subsets differ; paired comparison uses same37 stories. Failure25 remains in total61 denominators.'}
        if (out / 'manifest.json').exists():
            assert p.read(out / 'manifest.json') == manifest
        else:
            p.save(out / 'manifest.json', manifest)
        with (p.ROOT / 'outputs/generation.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            pre_rl_gpu.wait_ready(status)
            from bert_score import BERTScorer
            status('loading_encoder')
            scorer = BERTScorer(model_type=str(model_path), num_layers=9, idf=False,
                                device='cuda', batch_size=8, rescale_with_baseline=False)
            tokenizer = scorer._tokenizer
            chrf = sacrebleu.CHRF(char_order=6, word_order=2, beta=2)
            rows = []
            for i, item in enumerate(items):
                texts = [item['candidate'], *item['references']]
                token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
                clipped = [tokenizer.decode(ids[:510], skip_special_tokens=True) for ids in token_ids]
                precision, recall, f1 = scorer.score([clipped[0]], [clipped[1:]], batch_size=1)
                full = all(len(ids) <= 510 for ids in token_ids)
                row = {k: item[k] for k in ('task', 'variant', 'source_id')}
                row.update({'bert_prefix_precision': precision.item(), 'bert_prefix_recall': recall.item(),
                            'bert_prefix_f1': f1.item(), 'bert_full_f1_if_fits': f1.item() if full else None,
                            'encoder_token_lengths': list(map(len, token_ids)),
                            'encoder_prefix_coverage': [min(510, len(ids)) / max(1, len(ids)) for ids in token_ids],
                            'chrf_pp_full': chrf.sentence_score(item['candidate'], item['references']).score,
                            'rouge_l_unicode_full': max(rouge_l(item['candidate'], ref) for ref in item['references'])})
                assert all(math.isfinite(row[k]) for k in ('bert_prefix_f1', 'chrf_pp_full', 'rouge_l_unicode_full'))
                rows.append(row)
                p.save(out / 'rows.json', rows)
                status('running', completed=len(rows), total=len(items), latest=row)
                print(json.dumps(row), flush=True)
            groups = {}
            for task in ('story61', 'commongen100'):
                for variant in ('base', 'sft'):
                    group = [r for r in rows if r['task'] == task and r['variant'] == variant]
                    groups[f'{task}/{variant}'] = {'valid_n': len(group), 'total_n': 61 if task == 'story61' else 100,
                        'bert_full_fits_n': sum(r['bert_full_f1_if_fits'] is not None for r in group),
                        **{key: sum(r[key] for r in group) / len(group) for key in
                           ('bert_prefix_f1', 'chrf_pp_full', 'rouge_l_unicode_full')}}
            paired = {}
            for task in ('story61', 'commongen100'):
                indexed = {(r['variant'], r['source_id']): r for r in rows if r['task'] == task}
                ids = sorted({n for variant, n in indexed if ('base', n) in indexed and ('sft', n) in indexed})
                paired[task] = {'n': len(ids), **{key: interval([indexed['sft', n][key] - indexed['base', n][key] for n in ids])
                    for key in ('bert_prefix_f1', 'chrf_pp_full', 'rouge_l_unicode_full')}}
            assert all(g.digest(path) == sha for path, sha in protected.items())
            p.save(out / 'summary.json', {'groups': groups, 'paired_differences': paired,
                   'invalid_story_outputs': invalid, 'protected_files_unchanged': len(protected),
                   'elapsed_seconds': time.time() - started, 'optimizer_updates': 0, 'api_calls': 0})
            status('complete', completed=len(rows), total=len(items))
    except BaseException as error:
        status('failed', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
