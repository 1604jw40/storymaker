"""Replay fixed tokens through the generation cache and compare teacher-forced probabilities."""
import hashlib
import json
import msvcrt
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, gpu, validate_settings
from storymaker.rlaif.code.rlaif_storage import save_json
from storymaker.rlaif.code.rlaif_training_report import committed_batches, sha


def replay_cases(root, config):
    batch=root/f"batch_{config.get('batch_index', 0):04d}"
    paths=sorted((batch/'rollouts').glob('*.json'))
    committed=committed_batches(root)
    source_config=root/'config.json'
    first_name='actual_first_rollout'
    if not paths:
        # Before collection, replay preserved train tokens under the configured policy to avoid a readiness cycle.
        tag=Path(config.get('policy_checkpoint','')).parent.name
        assert tag in committed, 'A committed configured policy is required before collection'
        previous=committed[tag]
        paths=sorted((previous/'rollouts').glob('*.json'))
        source_config=previous/'config.json'
        if not source_config.exists():source_config=previous.parent/'config.json'
        first_name='preserved_first_training_rollout'
    assert paths, 'No verified training trajectory exists for this policy'
    actual_path=paths[0]

    def checked(path, source_config):
        story=p.read(path)
        assert story['identity']['config_sha256']==sha(source_config)
        assert story['identity']['source_id'] in config['train_ids']
        assert story['identity']['source_id'] not in config['validation_ids']+config['excluded_test_ids']
        assert story['prompt_ids'] and 0<len(story['generated_token_ids'])<=config['max_new_tokens']
        assert story['generated_tokens']==len(story['generated_token_ids'])
        return story

    actual=checked(actual_path,source_config)
    for previous in committed.values():
        source_config=previous/'config.json'
        if not source_config.exists():source_config=previous.parent/'config.json'
        for path in sorted((previous/'rollouts').glob('*.json')):
            story=checked(path,source_config)
            if len(story['generated_token_ids'])==config['max_new_tokens']:
                return [(first_name,actual['prompt_ids'],actual['generated_token_ids'],actual_path),
                    ('preserved_full_length_training_output',story['prompt_ids'],story['generated_token_ids'],path)]
    raise AssertionError('A preserved full-length training rollout is required for cache/full-forward comparison')


def main():
    import torch
    from torch.nn import functional as F
    from storymaker.rlaif.code.rlaif_precision import projection
    from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor

    folder = OUT / 'environment/probability_equivalence'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'verified.json').exists()
    config = p.read(OUT / 'config.json')
    validate_settings(config)
    started = time.monotonic()

    def status(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at_utc': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': time.monotonic()-started, 'gpu': gpu(), 'new_generations': 0, **extra}
        save_json(folder / 'status.json', value)
        print(stage, extra, flush=True)

    try:
        with GPU_LOCK.open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            assert gpu()['free_mib'] > 13500 and gpu()['temperature_c'] < 80
            status('loading_probability_replay')
            if config.get('operation_precision'):
                from storymaker.rlaif.code.rlaif_train import load_model
                model,_,_=load_model(config, previous=Path(config['policy_checkpoint']) if config.get('policy_checkpoint') else None)
            else:
                model = AutoModelForImageTextToText.from_pretrained(config['model'], local_files_only=True,
                           dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
            tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
            cases = replay_cases(OUT, config)
            criteria = {'max_abs_logprob_difference': .05, 'mean_abs_logprob_difference': .005,
                        'min_argmax_agreement': .99, 'temperature': config['temperature'],
                        'scope': 'configured-operation finite-precision consistency on two fixed prefixes; not bitwise or all-prefix proof'}
            save_json(folder / 'plan.json', {'criteria': criteria,
                'cases': [{'name': name, 'tokens': len(tokens), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                          for name, _, tokens, path in cases]})
            results = []
            for name, prompt, completion, path in cases:
                trace, argmax = [], []

                class Replay(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        index = input_ids.shape[1] - len(prompt)
                        assert 0 <= index < len(completion)
                        trace.append(float(torch.log_softmax(scores[0].float()/config['temperature'], 0)[completion[index]]))
                        argmax.append(int(scores[0].argmax()))
                        forced = torch.full_like(scores, float('-inf'))
                        forced[0, completion[index]] = 0
                        return forced

                status('replaying_generation_cache', case=name, tokens=len(completion))
                inputs = torch.tensor([prompt], device='cuda')
                with torch.inference_mode():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=False, max_new_tokens=len(completion), logits_processor=[Replay()],
                        repetition_penalty=1., eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, use_cache=True)
                assert output[0, len(prompt):].tolist() == completion
                assert len(trace) == len(completion)
                status('comparing_full_forward', case=name)
                ids = torch.tensor([prompt+completion], device='cuda')
                full, full_argmax = [], []
                with torch.no_grad():
                    hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                              use_cache=False, return_dict=True).last_hidden_state[0, len(prompt)-1:-1]
                    for start in range(0, len(completion), 32):
                        logits = projection(hidden[start:start+32], model.lm_head.weight)/config['temperature']
                        targets = torch.tensor(completion[start:start+32], device='cuda')
                        full.extend((-F.cross_entropy(logits, targets, reduction='none')).cpu().tolist())
                        full_argmax.extend(logits.argmax(-1).cpu().tolist())
                differences = (torch.tensor(trace)-torch.tensor(full)).abs()
                result = {'name': name, 'tokens': len(completion),
                    'max_abs_logprob_difference': float(differences.max()),
                    'mean_abs_logprob_difference': float(differences.mean()),
                    'argmax_agreement': sum(a == b for a, b in zip(argmax, full_argmax))/len(completion),
                    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                result['passed'] = (result['max_abs_logprob_difference'] <= criteria['max_abs_logprob_difference']
                    and result['mean_abs_logprob_difference'] <= criteria['mean_abs_logprob_difference']
                    and result['argmax_agreement'] >= criteria['min_argmax_agreement'])
                torch.save({'cached_logps': torch.tensor(trace), 'full_logps': torch.tensor(full),
                            'cached_argmax': argmax, 'full_argmax': full_argmax}, folder / (name+'.pt'))
                save_json(folder / (name+'.json'), result)
                results.append(result)
                assert result['passed'], 'Cache/full-forward mismatch requires inspection before RL'
                del hidden, ids, output, logits, inputs
                torch.cuda.empty_cache()
            receipt = {'passed': all(x['passed'] for x in results), 'criteria': criteria, 'results': results,
                'policy_checkpoint': config.get('policy_checkpoint'),
                'elapsed_seconds': time.monotonic()-started, 'actual_rl_updates': 0,
                'new_generations': 0, 'config_sha256': hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()}
            save_json(folder / 'verified.json', receipt)
            status('complete', passed=True)
    except BaseException as error:
        status('needs_attention', error=type(error).__name__, message=str(error), automatic_retry=False)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
