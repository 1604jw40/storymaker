"""On-policy trajectories for direct-RLAIF; original policy stays fixed for a batch."""
import hashlib
import json
import msvcrt
import os
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

from storymaker import project as p
from storymaker.qwen.code.generate_story import parse_story
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK, gpu, validate_settings, preserve_batch_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_options(config, tokenizer):
    import math
    assert type(config['max_new_tokens']) is int and config['max_new_tokens'] > 0
    assert math.isfinite(config['temperature']) and config['temperature'] > 0
    # Chosen log probabilities currently represent the full, temperature-scaled policy.
    assert config['top_p'] == 1 and config['top_k'] == 0 and config.get('repetition_penalty', 1.) == 1
    return dict(max_new_tokens=config['max_new_tokens'], do_sample=True,
                temperature=config['temperature'], top_p=config['top_p'], top_k=config['top_k'],
                repetition_penalty=config.get('repetition_penalty', 1.), eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                use_cache=True)


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    config=p.read(OUT/'config.json');manifest=p.read(OUT/'manifest.json')
    assert sha(OUT/'config.json')==manifest['config_sha256']
    assert sha(Path(__file__))==manifest['generation_code_sha256']
    assert sha(OUT/'system_prompt.txt')==manifest['prompt_sha256']
    assert sha(p.PROCESSED/'dataset.json')==manifest['dataset_sha256']
    if config.get('operation_precision'):
        readiness=p.read(OUT/'environment/training_readiness.json')
        assert readiness['passed'] and readiness['config_sha256']==manifest['config_sha256']
        assert all(sha(p.Path(path))==digest for path,digest in readiness['evidence_sha256'].items())
        assert all(sha(p.code_path(name))==digest for name,digest in manifest['runtime_code_sha256'].items())
    else:
        assert p.read(OUT/'environment/full_policy_sharded/verified.json')['checkpoint_restore_all_parameters_verified']
        assert p.read(OUT/'environment/value_backward_verified.json')['all_backbone_gradients_finite']
    batch=OUT/f"batch_{config.get('batch_index', 0):04d}"
    ids=validate_settings(config)
    count=len(ids)
    folder=batch/'rollouts';folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()

    def status(stage,**extra):
        value=dict(stage=stage,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),
             elapsed_seconds=time.monotonic()-started,actual_rl_updates=0,gpu=gpu(),**extra)
        p.save(batch/'status.json',value);print(json.dumps(value),flush=True)

    try:
        with GPU_LOCK.open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            assert gpu()['free_mib']>13500 and gpu()['temperature_c']<80
            preserve_batch_config(OUT,config)
            books={x['source_id']:x for x in p.read(p.PROCESSED/'dataset.json')}
            assert not set(config['train_ids'])&set(config['validation_ids'])
            assert not (set(config['train_ids'])|set(config['validation_ids']))&set(config['excluded_test_ids'])
            assert all(books[i]['source_split']!='test' for i in ids)
            status('loading_policy',completed=0,total=count)
            if config.get('operation_precision'):
                from storymaker.rlaif.code.rlaif_train import load_model
                model,_,_=load_model(config, previous=Path(config['policy_checkpoint']) if config.get('policy_checkpoint') else None)
            else:
                model=AutoModelForImageTextToText.from_pretrained(config['model'],local_files_only=True,
                       dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
            tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
            for index,source_id in enumerate(ids):
                if not p.read(OUT/'control.json')['enabled']:
                    status('stopped_by_control',completed=index,total=count)
                    return
                assert all(sha(p.code_path(name))==digest for name,digest in manifest.get('runtime_code_sha256',{}).items())
                target=folder/(str(index).zfill(3)+'_'+str(source_id)+'.json')
                identity={'policy_version':config.get('policy_version', 'initial_original_0000'),'source_id':source_id,
                          'seed':config['seed']+index,'config_sha256':manifest['config_sha256']}
                if target.exists():
                    assert p.read(target)['identity']==identity
                    continue
                moral=books[source_id]['input']['moral']
                inputs=tokenizer.apply_chat_template([
                    {'role':'system','content':(OUT/'system_prompt.txt').read_text('utf-8')},
                    {'role':'user','content':json.dumps({'moral':moral},ensure_ascii=False)}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda')
                set_seed(identity['seed']);torch.cuda.reset_peak_memory_stats()
                status('generating',completed=index,total=count,source_id=source_id)
                tick=time.perf_counter()
                with torch.inference_mode():
                    tokens=model.generate(**inputs, **generation_options(config, tokenizer))
                generated=tokens[0,inputs['input_ids'].shape[1]:].tolist()
                raw=tokenizer.decode(generated,skip_special_tokens=True)
                _,problems=parse_story(raw)
                eos=bool(generated and generated[-1]==tokenizer.eos_token_id)
                if not eos:problems.append('no_eos_before_limit')
                value={'identity':identity,'moral':moral,'prompt_ids':inputs['input_ids'][0].tolist(),
                      'generated_token_ids':generated,'raw_text':raw,'ended_with_eos':eos,
                      'format_problems':problems,'generated_tokens':len(generated),
                      'generation_seconds':time.perf_counter()-tick,
                      'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                      'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                      'quality_resampled':False,'api_scored':False}
                p.save(target,value)
                del tokens,inputs
                torch.cuda.empty_cache()
                status('saved',completed=index+1,total=count,source_id=source_id,tokens=len(generated),eos=eos)
            status('batch_generation_complete',completed=count,total=count)
    except BaseException as error:
        status('needs_attention',error=type(error).__name__,message=str(error),automatic_retry=False)
        traceback.print_exc();raise


if __name__=='__main__':main()
