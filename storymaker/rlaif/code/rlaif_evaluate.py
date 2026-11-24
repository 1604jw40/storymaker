"""Matched original/RL generation and validation CE; never trains, repairs or resamples outputs."""
import argparse
import ast
import hashlib
import json
import math
import msvcrt
import os
import re
import shutil
import time
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code.generate_story import parse_story
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, gpu
from storymaker.rlaif.code.rlaif_storage import save_json
from storymaker.rlaif.code.rlaif_training_report import committed_batch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prompt_rows(config, plan, split, books):
    if split == 'validation':
        ids = plan['validation_ids']
        assert ids == config['validation_ids'] and len(ids) == len(set(ids)) == 43
        assert not set(ids) & (set(config['train_ids']) | set(config['excluded_test_ids']))
        return [{'id': str(i), 'source_id': i, 'moral': books[i]['input']['moral']} for i in ids]
    assert split == 'final'
    path = Path(plan['final_prompt_file'])
    assert sha(path) == plan['final_prompt_sha256']
    fixed = p.read(path)
    assert fixed['actual_rl_updates_at_freeze'] == 0 and fixed['final_only_after_policy_selection']
    assert len(fixed['prompts']) == 32 and len({r['id'] for r in fixed['prompts']}) == 32
    return fixed['prompts']


def local_metrics(raw, eos):
    parsed, problems = parse_story(raw)
    scenes = parsed.get('scenes') if isinstance(parsed, dict) else None
    readable = (isinstance(scenes, list) and bool(scenes)
        and all(isinstance(s, dict) and isinstance(s.get('text'), str) for s in scenes)
        and any(s['text'].strip() for s in scenes))
    text = [re.sub(r'\W', '', unicodedata.normalize('NFKC', s['text'])) for s in scenes] if readable else []
    long_text = [t for t in text if len(t) >= 20]
    repeated = sum(count-1 for count in Counter(long_text).values() if count > 1)
    return {'eos': bool(eos), 'strict_format': not problems, 'semantic_eligible': bool(eos and readable),
            'normalized_exact_repeated_long_scenes': repeated,
            'repetition_scope': 'exact normalized scene text >=20 characters; not semantic repetition or novelty certification',
            'format_problems': problems}


def policy_context(checkpoint):
    if checkpoint is None:
        return 'original', None
    checkpoint = checkpoint.resolve()
    assert checkpoint.name == 'policy'
    tag = checkpoint.parent.name
    assert re.fullmatch(r'update_\d{4}', tag)
    batch=committed_batch(OUT, tag)
    receipt_path=batch/'joint_update_verified.json'
    if not receipt_path.exists():receipt_path=batch/'first_update_verified.json'
    receipt=p.read(receipt_path)
    assert receipt['committed_policy_updates'] == receipt['committed_value_updates'] == 1
    assert checkpoint == Path(receipt['policy']).resolve(), 'Only an actually committed update is evaluated'
    assert (checkpoint/'manifest.json').exists()
    return tag, checkpoint


def output_origin(path):
    seen=set();path=Path(path).resolve()
    while True:
        assert path not in seen, 'Cycle in cached output provenance'
        seen.add(path)
        record=p.read(path);manifest_path=path.parent/'manifest.json';manifest=p.read(manifest_path)
        assert record['identity']['tag']==manifest['tag'] and record['quality_resampled'] is False
        if record['identity']['manifest_sha256']==sha(manifest_path):
            return path,manifest_path
        provenance=p.read(path.parent/'provenance'/path.name)
        source=Path(provenance['source']).resolve();previous=p.read(source.parent/'manifest.json')
        assert provenance['new_generation'] is False and provenance['identity_preserved'] is True
        assert provenance['sha256']==sha(path)==sha(source), 'Cached output bytes changed'
        field='reused_original_manifest_sha256' if manifest['tag']=='original' else 'reused_policy_manifest_sha256'
        assert manifest[field]==sha(source.parent/'manifest.json')
        assert previous['tag']==manifest['tag']
        if manifest['tag']!='original':
            assert previous['checkpoint_manifest_sha256']==manifest['checkpoint_manifest_sha256']
        assert previous['settings']==manifest['settings']
        assert previous['system_prompt_sha256']==manifest['system_prompt_sha256']
        path=source


def original_origin(path):
    assert p.read(path)['identity']['tag']=='original'
    return output_origin(path)


def verify_original_math(manifest_path):
    manifest=p.read(manifest_path)
    for name,function in [('rlaif_evaluate.py','teacher_forced'),('rlaif_train.py','load_model')]:
        candidates=[manifest_path.parent/'code_snapshot'/name,p.code_path(name),manifest_path.parents[3].parents[1]/name]
        source=next((path for path in candidates if path.exists() and sha(path)==manifest['code_sha256'][name]),None)
        assert source is not None, 'Exact cached original code must be preserved before reuse'
        trees=[ast.parse(path.read_text('utf-8')) for path in (source,p.code_path(name))]
        bodies=[next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function) for tree in trees]
        for body in bodies:
            for node in ast.walk(body):
                if isinstance(node,ast.ImportFrom) and node.module=='storymaker.rlaif.code.rlaif_precision':node.module='rlaif_precision'
        assert ast.dump(bodies[0])==ast.dump(bodies[1]), 'Changed numeric function; inspect cache compatibility'
    assert manifest['code_sha256']['rlaif_precision.py']==sha(p.code_path('rlaif_precision.py'))


def verify_reused_original(record, previous, manifest, prompt, prefix, policy_sha256, origin_manifest_sha256=None):
    assert previous['tag']==manifest['tag']=='original'
    return verify_reused_output(record,previous,manifest,prompt,prefix,policy_sha256,origin_manifest_sha256)


def verify_reused_output(record, previous, manifest, prompt, prefix, policy_sha256, origin_manifest_sha256=None):
    assert previous['tag']==manifest['tag']
    if manifest['tag']!='original':
        assert previous['checkpoint_manifest_sha256']==manifest['checkpoint_manifest_sha256']
        assert manifest['checkpoint_manifest_sha256'] is not None
    assert previous['settings']==manifest['settings']
    assert previous['system_prompt_sha256']==manifest['system_prompt_sha256']
    identity=record['identity']
    assert identity['tag']==manifest['tag'] and identity['id']==prompt['id']
    field='reused_original_manifest_sha256' if manifest['tag']=='original' else 'reused_policy_manifest_sha256'
    assert identity['manifest_sha256']==(origin_manifest_sha256 or manifest[field])
    assert identity['seed']==manifest['settings']['seed'] and identity['policy_sha256']==policy_sha256
    assert record['moral']==prompt['moral'] and record['prompt_ids']==prefix
    assert record['quality_resampled'] is False and record['teacher_forced'] is not None
    assert len(record['generated_token_ids'])<=manifest['settings']['max_new_tokens']


def teacher_forced(model, record, eos_token_id):
    import torch
    from torch.nn import functional as F
    from storymaker.rlaif.code.rlaif_precision import projection
    ids = torch.tensor([record['input_ids']], device='cuda')
    labels = torch.tensor([record['labels']], device='cuda')
    assert (labels == eos_token_id).sum() == 1
    hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         use_cache=False, return_dict=True).last_hidden_state
    h=hidden[:, :-1].reshape(-1,hidden.shape[-1]);targets=labels[:, 1:].reshape(-1)
    total=h.new_zeros((),dtype=torch.float32)
    for start in range(0,len(targets),32):
        total=total+F.cross_entropy(projection(h[start:start+32],model.lm_head.weight),
            targets[start:start+32],reduction='sum',ignore_index=-100)
    ce=total/(targets!=-100).sum()
    position = record['labels'].index(eos_token_id)
    logits = projection(hidden[:, position-1], model.lm_head.weight)[0]
    eos_nll = float(-F.log_softmax(logits, 0)[eos_token_id])
    result = {'ce': float(ce), 'conditional_perplexity': math.exp(float(ce)),
        'supervised_tokens': int((labels[:, 1:] != -100).sum()), 'gold_prefix_eos_nll': eos_nll,
        'gold_prefix_eos_rank': int((logits > logits[eos_token_id]).sum())+1,
        'scope': 'untruncated v5 assistant JSON plus template ending; EOS conditioned on gold answer'}
    assert math.isfinite(result['ce']) and math.isfinite(eos_nll)
    return result


def reference_record(record, prefix):
    """Keep the exact gold suffix while applying the run's declared system prompt."""
    old = record['prompt_tokens']
    assert len(record['input_ids']) == len(record['labels']) and old > 0
    suffix, labels = record['input_ids'][old:], record['labels'][old:]
    assert suffix and all(label == token or label == -100 for token, label in zip(suffix, labels))
    return dict(record, input_ids=list(prefix)+suffix, labels=[-100]*len(prefix)+labels, prompt_tokens=len(prefix))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['validation', 'final'], default='validation')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = p.read(OUT/'config.json')
    plan = p.read(OUT/'evaluation/evaluation_plan.json')
    protocol = Path(plan['protocol_file'])
    if not protocol.is_absolute():
        protocol = p.ROOT.parent/protocol
    assert sha(protocol) == plan['protocol_sha256']
    assert sha(OUT/'config.json') == p.read(OUT/'manifest.json')['config_sha256']
    tag, checkpoint = policy_context(args.checkpoint)
    books = {b['source_id']: b for b in p.read(p.PROCESSED/'dataset.json')}
    prompts = prompt_rows(config, plan, args.split, books)
    if args.split == 'final':
        selected = p.read(OUT/'evaluation/final_policy_selection.json')
        assert selected['final_prompt_sha256'] == plan['final_prompt_sha256']
        assert selected['chosen_tag'] == tag or tag == 'original'
        assert selected['selection_used_final_outputs'] is False
    folder = OUT/'evaluation'/args.split/tag
    folder.mkdir(parents=True, exist_ok=True)
    previous_folder=Path(config['source_previous_run'])/'evaluation/validation'/tag
    previous_manifest=previous_folder/'manifest.json'
    manifest = {'tag': tag, 'checkpoint': str(checkpoint) if checkpoint else None, 'split': args.split,
        'settings': plan['generation'], 'prompt_ids': [r['id'] for r in prompts],
        'evaluation_plan_sha256': sha(OUT/'evaluation/evaluation_plan.json'),
        'config_sha256': sha(OUT/'config.json'), 'system_prompt_sha256': sha(OUT/'system_prompt.txt'),
        'checkpoint_manifest_sha256': sha(checkpoint/'manifest.json') if checkpoint else None,
        'code_sha256': {name: sha(p.code_path(name)) for name in ['rlaif_evaluate.py', 'rlaif_train.py',
            'rlaif_storage.py', 'rlaif_environment.py', 'generate_story.py', 'sft_v5_train.py', 'rlaif_precision.py']},
        'reused_original_manifest_sha256':sha(previous_manifest) if config.get('reuse_previous_original',True) and tag=='original' and args.split=='validation' and previous_manifest.exists() else None,
        'reused_policy_manifest_sha256':sha(previous_manifest) if config.get('reuse_previous_policy_validation',False) and checkpoint==Path(config['policy_checkpoint']).resolve() and args.split=='validation' and previous_manifest.exists() else None,
        'training_updates': 0, 'paid_api_calls': 0, 'quality_resampling': False}
    path = folder/'manifest.json'
    if path.exists():
        assert p.read(path) == manifest, 'Changed evaluation context; never mix or overwrite prior outputs'
    else:
        save_json(path, manifest)
    if not args.execute:
        print(json.dumps({'plan_only': True, 'tag': tag, 'prompts': len(prompts), 'manifest': str(path)}))
        return
    started = time.monotonic()

    def status(stage, **extra):
        save_json(folder/'status.json', {'stage': stage, 'pid': os.getpid(), 'tag': tag,
            'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'gpu': gpu(), 'actual_training_updates_in_this_job': 0, **extra})

    try:
        assert not (folder/'failure.json').exists(), 'Inspect an earlier technical failure before continuing'
        with GPU_LOCK.open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            assert p.read(OUT/'control.json')['enabled']
            snapshot = gpu()
            assert snapshot['free_mib'] > 13500 and snapshot['temperature_c'] < 80
            from storymaker.rlaif.code.rlaif_train import load_model, verify_sources, model_digest
            from transformers import AutoTokenizer, set_seed
            import torch
            verify_sources(config, 'evaluation_'+args.split+'_'+tag)
            status('loading_evaluation_policy', completed=0, total=len(prompts))
            model, _, parameters = load_model(config, previous=checkpoint)
            policy_sha256 = model_digest(parameters)
            # The conditional generation wrapper requires consistent device routing, as in the fixed rollout worker.
            if not config.get('operation_precision'):
                model.to('cuda')
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
            encoded = {r['source_id']: r for r in p.read(Path(config['source_sft_run'])/'encoded_development.json')}
            settings = plan['generation']
            generated_count=reused_count=0
            reuse_manifest=manifest['reused_original_manifest_sha256'] or manifest['reused_policy_manifest_sha256']
            if reuse_manifest:
                # Reuse requires identical inference/CE math, weights, settings, seed and exact prefix.
                verify_original_math(previous_manifest)
                old=p.read(previous_manifest)
                assert tokenizer.pad_token_id != 0
            for index, prompt in enumerate(prompts):
                if not p.read(OUT/'control.json')['enabled']:
                    status('stopped_by_control',completed=index,total=len(prompts));return
                assert all(sha(p.code_path(name))==digest for name,digest in manifest['code_sha256'].items())
                target = folder/(prompt['id']+'.json')
                identity = {'id': prompt['id'], 'tag': tag, 'seed': settings['seed'],
                    'policy_sha256': policy_sha256, 'manifest_sha256': sha(path)}
                prefix = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': (OUT/'system_prompt.txt').read_text('utf-8')},
                    {'role': 'user', 'content': json.dumps({'moral': prompt['moral']}, ensure_ascii=False)}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids']
                if target.exists() and p.read(target)['identity']==identity:
                    continue
                preserved=previous_folder/(prompt['id']+'.json')
                if reuse_manifest and preserved.exists():
                    cached=p.read(preserved)
                    _,origin_manifest=output_origin(preserved)
                    verify_original_math(origin_manifest)
                    verify_reused_output(cached,old,manifest,prompt,prefix,policy_sha256,sha(origin_manifest))
                    if target.exists():assert sha(target)==sha(preserved)
                    else:shutil.copy2(preserved,target)
                    save_json(folder/'provenance'/target.name,{'source':str(preserved),'sha256':sha(preserved),
                        'source_manifest':str(previous_manifest),'new_generation':False,'identity_preserved':True})
                    reused_count+=1
                    status('verified_cached_original' if tag=='original' else 'verified_cached_policy',completed=index+1,total=len(prompts),reused_outputs=reused_count)
                    continue
                assert not target.exists(),'Changed output context; preserve and inspect'
                gold = None
                with torch.inference_mode():
                    if args.split == 'validation':
                        record = reference_record(encoded[prompt['source_id']],prefix)
                        assert prefix == record['input_ids'][:record['prompt_tokens']]
                        gold = teacher_forced(model, record, tokenizer.eos_token_id)
                    set_seed(settings['seed'])
                    torch.cuda.reset_peak_memory_stats()
                    inputs = torch.tensor([prefix], device='cuda')
                    status('generating', completed=index, total=len(prompts), prompt_id=prompt['id'])
                    tick = time.perf_counter()
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=settings['do_sample'], temperature=settings['temperature'],
                        top_p=settings['top_p'], top_k=settings['top_k'],
                        repetition_penalty=settings['repetition_penalty'], max_new_tokens=settings['max_new_tokens'],
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, use_cache=True)
                tokens = output[0, len(prefix):].tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                eos = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
                save_json(target, {'identity': identity, 'moral': prompt['moral'], 'prompt_ids': prefix,
                    'generated_token_ids': tokens, 'raw_text': raw, 'ended_with_eos': eos,
                    'generated_tokens': len(tokens), 'generation_seconds': time.perf_counter()-tick,
                    'local_metrics': local_metrics(raw, eos), 'teacher_forced': gold,
                    'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                    'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'quality_resampled': False})
                del output, inputs
                generated_count+=1
                status('saved', completed=index+1, total=len(prompts))
            save_json(folder/'verified.json', {'completed': len(prompts), 'tag': tag, 'split': args.split,
                'manifest_sha256': sha(path), 'policy_sha256': policy_sha256,
                'elapsed_seconds': time.monotonic()-started, 'actual_training_updates_in_this_job': 0,
                'new_outputs_in_this_invocation':generated_count,'reused_original_outputs':reused_count if tag=='original' else 0,
                'reused_policy_outputs':reused_count if tag!='original' else 0,
                'elapsed_seconds_scope':'current_invocation_only; reused output generation times are historical',
                'api_evaluation': 'pending', 'quality_claim': None})
            status('complete', completed=len(prompts), total=len(prompts))
    except BaseException as error:
        save_json(folder/'failure.json', {'error': type(error).__name__, 'message': str(error), 'automatic_retry': False})
        status('needs_attention', error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
