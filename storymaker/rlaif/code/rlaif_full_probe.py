"""Discarded full-policy backward and streamed FP32 Adafactor checkpoint probe."""
import gc
import hashlib
import json
import msvcrt
import os
import time
import traceback
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, gpu


def main():
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    folder=OUT/'environment/full_policy_sharded';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'verified.json').exists()
    started=time.monotonic(); gradients={}

    def status(stage,**extra):
        p.save(folder/'status.json',dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
               gpu=gpu(),actual_rl_updates=0,**extra))
        print(stage,extra,flush=True)

    try:
        with (p.ROOT/'outputs/generation.lock').open('a+b') as lock:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            assert gpu()['free_mib']>13500 and gpu()['temperature_c']<80
            status('loading_full_policy')
            model=AutoModelForImageTextToText.from_pretrained(p.MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
            model.requires_grad_(False)
            language=model.model.language_model;language.to('cuda');language.requires_grad_(True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.train()
            tokenizer=AutoTokenizer.from_pretrained(p.MODEL,local_files_only=True)
            data=p.read(p.ROOT/'outputs/pre_rl_20261114/checkpoint_rollouts/epoch3_205.json')['generation']
            prompt=tokenizer.apply_chat_template([
                {'role':'system','content':(p.ROOT/'outputs/rl_pilot_20261114/system_prompt.txt').read_text('utf-8')},
                {'role':'user','content':json.dumps({'moral':data['moral']},ensure_ascii=False)}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True)['input_ids']
            completion=data['generated_token_ids'];assert len(completion)==4096
            ids=torch.tensor([prompt+completion],device='cuda');targets=ids[0,len(prompt):]
            params=list(language.named_parameters())
            before={n:hashlib.sha256(x.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for n,x in params}

            def offload(name):
                def hook(param):
                    grad=param.grad.detach().to('cpu')
                    if name in gradients:gradients[name].add_(grad)
                    else:gradients[name]=grad
                    param.grad=None
                return hook

            handles=[x.register_post_accumulate_grad_hook(offload(n)) for n,x in params]
            torch.cuda.reset_peak_memory_stats()
            status('full_policy_forward',total_tokens=ids.shape[1])
            hidden=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state[0,len(prompt)-1:-1]
            weight=model.lm_head.weight
            loss=torch.zeros((),device='cuda')
            for start in range(0,len(completion),16):
                def part(h,t):
                    return F.cross_entropy(F.linear(h,weight).float(),t,reduction='sum')/len(completion)
                loss=loss+checkpoint(part,hidden[start:start+16],targets[start:start+16],use_reentrant=False)
            status('full_policy_backward',loss=float(loss.detach()))
            loss.backward()
            assert set(gradients)=={n for n,_ in params}
            norm2=0.
            for grad in gradients.values():
                for chunk in grad.flatten().split(1_000_000):
                    assert torch.isfinite(chunk).all()
                    norm2+=float(chunk.float().square().sum())
            assert norm2>0
            for handle in handles:handle.remove()
            loss_value=float(loss.detach());del hidden,loss,targets,ids
            gc.collect();torch.cuda.empty_cache()
            status('streaming_cpu_fp32_optimizer',gradient_norm=norm2**0.5)
            changed=0;checkpoint_bytes=0;checks=[]
            checkpoint_dir=folder/'discarded_checkpoint';checkpoint_dir.mkdir(exist_ok=True)
            # Each parameter's Adafactor state is independent; stream FP32 masters to avoid all-state residency.
            for index,(name,param) in enumerate(sorted(params,key=lambda item:item[1].numel(),reverse=True)):
                path=checkpoint_dir/(hashlib.sha256(name.encode()).hexdigest()+'.pt')
                assert not path.exists(),'Never overwrite a committed parameter checkpoint'
                grad=gradients.pop(name)
                master=torch.nn.Parameter(param.detach().to(device='cpu',dtype=torch.float32))
                master.grad=grad.float().mul_(min(1.,1/(norm2**0.5+1e-12)))
                del grad
                optimizer=torch.optim.Adafactor([master],lr=1e-5,weight_decay=0.,foreach=False)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                assert torch.isfinite(master).all()
                with torch.no_grad():param.copy_(master)
                digest=hashlib.sha256(param.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                changed+=digest!=before[name]
                chunks=[]
                # The Windows torch archive reader rejected a >2GB file; cloned chunks also avoid view-storage expansion.
                for part_index,chunk in enumerate(master.detach().flatten().split(16_000_000)):
                    chunk_path=checkpoint_dir/(path.stem+'_'+str(part_index)+'.pt')
                    torch.save(chunk.clone(),chunk_path)
                    chunks.append(chunk_path.name)
                    checkpoint_bytes+=chunk_path.stat().st_size
                torch.save({'name':name,'shape':list(master.shape),'chunks':chunks,'optimizer':optimizer.state_dict(),
                            'bf16_sha256':digest,'discarded_environment_probe':True},path)
                checkpoint_bytes+=path.stat().st_size
                checks.append({'name':name,'file':path.name,'bf16_sha256':digest,'changed':digest!=before[name]})
                del optimizer,master
                if index%40==0:status('streaming_cpu_fp32_optimizer',completed_parameters=index+1,total_parameters=len(params))
            assert changed>0 and not gradients
            status('verifying_streamed_checkpoint_restore',changed_parameters=changed)
            by_name=dict(params)
            for record in checks:
                saved=torch.load(checkpoint_dir/record['file'],map_location='cpu',weights_only=True)
                assert saved['name']==record['name'] and int(saved['optimizer']['state'][0]['step'])==1
                param=by_name[record['name']]
                restored=torch.empty(saved['shape'],dtype=torch.float32)
                offset=0
                for filename in saved['chunks']:
                    chunk=torch.load(checkpoint_dir/filename,map_location='cpu',weights_only=True)
                    restored.flatten()[offset:offset+chunk.numel()].copy_(chunk)
                    offset+=chunk.numel()
                assert offset==restored.numel()
                with torch.no_grad():param.zero_();param.copy_(restored)
                assert hashlib.sha256(param.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()==record['bf16_sha256']
                del saved,restored,chunk
            p.save(checkpoint_dir/'manifest.json',{'parameters':checks,'cpu_rng':str(folder/'cpu_rng.pt')})
            torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},folder/'cpu_rng.pt')
            receipt=dict(total_tokens=len(prompt)+len(completion),completion_tokens=len(completion),
                trainable_parameters=sum(x.numel() for _,x in params),finite_loss=loss_value,
                gradient_norm=norm2**0.5,changed_parameters=changed,total_parameter_tensors=len(params),
                checkpoint_restore_all_parameters_verified=True,checkpoint_bytes=checkpoint_bytes,
                optimizer='torch Adafactor, independent streamed FP32 CPU masters',
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,elapsed_seconds=time.monotonic()-started,
                discarded_environment_updates=1,actual_rl_updates=0,
                scope='Full-parameter CE memory/update/checkpoint smoke only. Not a direct-RLAIF training result.')
            p.save(folder/'verified.json',receipt);status('complete',receipt=receipt)
    except BaseException as error:
        status('needs_attention',error=type(error).__name__,message=str(error),automatic_retry=False)
        traceback.print_exc();raise


if __name__=='__main__':main()
