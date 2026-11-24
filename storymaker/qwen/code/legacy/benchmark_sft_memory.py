"""Temporary, isolated A/B measurement using the unchanged v5 fit implementation."""
import argparse
import hashlib
import json
import msvcrt
import os
import statistics
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from storymaker.qwen.code.legacy import sft_v5_train as t
from storymaker.qwen.code.legacy import sft_v5_prepare as p
def measure(folder, case):
    import torch
    from transformers import Trainer, TrainerCallback

    task=p.read(folder/case/'fit_task.json')
    losses=[];gradients=[];steps=[]
    original_init=Trainer.__init__
    original_ce=t.chunked_ce

    def ce(*args,**kwargs):
        value=original_ce(*args,**kwargs)
        losses.append(value.detach())
        return value

    class Measurement(TrainerCallback):
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            visual=model.get_base_model().model.visual
            assert not any(x.requires_grad for x in visual.parameters())
            if case=='optimized':visual.to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.synchronize();self.started=time.perf_counter()

        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            digest=hashlib.sha256()
            for name,param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    digest.update(name.encode());digest.update(param.grad.detach().float().cpu().numpy().tobytes())
            gradients.append(digest.hexdigest())

        def on_substep_end(self,args,state,control,**kwargs):
            if case=='optimized':torch.cuda.empty_cache()

        def on_step_end(self,args,state,control,**kwargs):
            if case=='optimized':torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory=torch.cuda.memory_stats()
            step={'step':state.global_step,'seconds':time.perf_counter()-self.started,
                'allocated_gib':torch.cuda.memory_allocated()/2**30,
                'reserved_gib':torch.cuda.memory_reserved()/2**30,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'inactive_split_gib':memory['inactive_split_bytes.all.current']/2**30}
            # Host counters run outside the timed training interval.
            script=f"$x=Get-Counter '\\GPU Process Memory(*)\\Shared Usage','\\GPU Process Memory(*)\\Dedicated Usage'; $x.CounterSamples | Where-Object {{$_.InstanceName -like 'pid_{os.getpid()}_*'}} | Select-Object Path,CookedValue | ConvertTo-Json -Compress"
            sample=subprocess.run(['powershell','-NoProfile','-Command',script],capture_output=True,text=True)
            step['host_counters']=json.loads(sample.stdout) if sample.returncode==0 and sample.stdout.strip() else None
            steps.append(step)
            p.save(folder/case/'measurement.json',{'case':case,'steps':steps,'losses':[float(x) for x in losses],'gradient_sha256':gradients})
            print(json.dumps(step),flush=True)

    def init(self,*args,**kwargs):
        kwargs['callbacks']=[*kwargs.get('callbacks',[]),Measurement()]
        original_init(self,*args,**kwargs)

    with patch.object(Trainer,'__init__',init),patch.object(t,'chunked_ce',ce),patch.object(t,'status',lambda state,**extra:print(json.dumps({'state':state,**extra}),flush=True)):
        t.run(task)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['baseline','optimized']);parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args();folder=args.folder.resolve()
    assert p.RUN.resolve() in folder.parents
    if args.case:
        measure(folder,args.case);return
    with (p.RUN/'run.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert t.gpu_ready()['ready'],'Another SFT or GPU workload is active.'
        config,records,folds=t.inputs()
        ordered=sorted(folds[1]['train_ids'],key=lambda n:len(records[n]['input_ids']))
        selected=[ordered[round(i*(len(ordered)-1)/31)] for i in range(32)]
        assert len(set(selected))==32 and len(records[selected[-1]]['input_ids'])>3500
        p.save(folder/'sample.json',{'source_ids':selected,'lengths':[len(records[n]['input_ids']) for n in selected],
            'fold':2,'test_or_validation_used':False,'config_sha256':hashlib.sha256((p.RUN/'config.json').read_bytes()).hexdigest()})
        for case in ('baseline','optimized'):
            task={'folder':str(folder/case),'train_ids':selected,'val_ids':[],
                'learning_rate':8e-5,'epochs':1,'tag':'memory_benchmark/'+case,'max_steps':8}
            p.save(folder/case/'fit_task.json',task)
            t.status('memory_benchmark',stage=case,folder=str(folder))
            with (folder/case/'stdout.log').open('wb') as out,(folder/case/'stderr.log').open('wb') as err:
                subprocess.run([config['python'],'-X','utf8','-u',str(Path(__file__).resolve()),'--folder',str(folder),'--case',case],stdout=out,stderr=err,check=True)
        before=p.read(folder/'baseline'/'measurement.json');after=p.read(folder/'optimized'/'measurement.json')
        assert before['gradient_sha256']==after['gradient_sha256'],'Gradient mismatch: do not deploy.'
        assert before['losses']==after['losses'],'Loss mismatch: do not deploy.'
        assert len(before['steps'])==len(after['steps'])==8 and len(before['losses'])==32
        result={'losses_and_gradients_bitwise_equal':True,
            'baseline_mean_seconds_after_warmup':statistics.mean(x['seconds'] for x in before['steps'][1:]),
            'optimized_mean_seconds_after_warmup':statistics.mean(x['seconds'] for x in after['steps'][1:]),
            'baseline':before['steps'][-1],'optimized':after['steps'][-1]}
        p.save(folder/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
