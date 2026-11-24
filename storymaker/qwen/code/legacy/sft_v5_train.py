"""GPU가 비면 실행할 v5 LoRA SFT. Optuna는 학습률만, 원작 5-fold와 테스트를 고정한다."""
import argparse
import gc
import hashlib
import json
import math
import msvcrt
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from storymaker.qwen.code.legacy import sft_v5_prepare as p
def status(state, **extra):
    value = {"state":state,"updated_at_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),**extra}
    p.save(p.RUN/"status.json",value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def gpu_ready():
    text = subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    free,util = map(int,text.strip().splitlines()[0].split(","))
    script = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $jobs=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and ($_.CommandLine -match 'run_benchmarks\\.py|run_5fold\\.py|sft_v5_train\\.py') } | Select-Object ProcessId,CommandLine); ConvertTo-Json -InputObject $jobs -Compress"
    result = subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,encoding="utf-8",check=True)
    if not result.stdout or not result.stdout.strip():raise RuntimeError("호스트 프로세스 조회 결과가 비어 있어 GPU 시작을 보류합니다.")
    processes = json.loads(result.stdout)
    if isinstance(processes,dict):processes = [processes]
    # Windows venv 실행기는 같은 명령의 부모 Python을 유지한다.
    others = [r for r in processes if r["ProcessId"] not in {os.getpid(),os.getppid()}]
    return {"ready":free>=13*1024 and util<=15 and not others,"free_mib":free,"utilization_percent":util,
        "conflicting_processes":others}


def chunked_ce(hidden, weight, targets, chunk_size=128):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    active = targets != -100
    hidden, targets = hidden[active],targets[active]
    if not targets.numel():raise ValueError("학습할 정답 토큰이 없습니다.")
    def part(h,w,t):return F.cross_entropy(F.linear(h,w).float(),t,reduction="sum")
    total = hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,targets.numel(),chunk_size):
        h,t = hidden[start:start+chunk_size],targets[start:start+chunk_size]
        value = checkpoint(part,h,weight,t,use_reentrant=False) if torch.is_grad_enabled() else part(h,weight,t)
        total = total+value
    return total/targets.numel()


def inputs():
    config = p.read(p.RUN/"config.json")
    for name,digest in config["dataset_sha256"].items():
        assert hashlib.sha256((p.PROCESSED/name).read_bytes()).hexdigest() == digest,name
    records = {r["source_id"]:r for r in p.read(p.RUN/"encoded_development.json")}
    folds = p.read(p.RUN/"folds.json")
    raw = p.read(p.PROCESSED/"dataset.json")
    test_ids = {r["source_id"] for r in raw if r["source_split"] == "test"}
    p.check_folds(folds,set(records),test_ids)
    assert len(records)==243 and not test_ids & set(records)
    assert all(len(r["input_ids"])<=config["max_sequence_tokens"] for r in records.values())
    assert config["search_parameters"] == ["learning_rate"] and config["epochs_per_fold"] == 5
    return config,records,folds


def worker_fit(folder, train_ids, val_ids, learning_rate, epochs, tag, max_steps, config):
    folder.mkdir(parents=True,exist_ok=True)
    task={"folder":str(folder.resolve()),"train_ids":train_ids,"val_ids":val_ids,
        "learning_rate":learning_rate,"epochs":epochs,"tag":tag,"max_steps":max_steps}
    task_file=folder/"fit_task.json"
    if task_file.exists():
        assert p.read(task_file)==task,"기존 worker 작업과 다릅니다."
    p.save(task_file,task)
    status("starting_worker",stage=tag,learning_rate=learning_rate)
    subprocess.run([config["python"],"-X","utf8","-u",str(Path(__file__).resolve()),"--fit-task",str(task_file)],check=True)
    released=gpu_ready()
    for _ in range(3):
        if released["free_mib"]>=13*1024:break
        time.sleep(2)
        released=gpu_ready()
    p.save(folder/"worker_exit_gpu.json",released)
    if released["free_mib"]<13*1024:raise RuntimeError("worker 종료 후 GPU 메모리가 충분히 반환되지 않았습니다.")
    result=p.read(folder/"result.json")
    assert result["source_ids"]==train_ids and result["learning_rate"]==learning_rate and result["epochs"]==epochs
    return result


def run(fit_task=None):
    start_config=p.read(p.RUN/"config.json")
    if datetime.now(timezone.utc)<datetime.fromisoformat(start_config["not_before"]):
        status("waiting_start_time",not_before=start_config["not_before"])
        return
    config,records,folds = inputs()
    if fit_task is None:
        first = gpu_ready()
        if not first["ready"]:
            status("waiting_gpu",gpu=first)
            return
        time.sleep(10)
        second = gpu_ready()
        if not second["ready"]:
            status("waiting_gpu",gpu=second)
            return
    import torch
    import optuna
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForImageTextToText,AutoTokenizer,Trainer,TrainerCallback,TrainingArguments,set_seed
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token = tokenizer.eos_token

    def collate(features):
        length = max(len(r["input_ids"]) for r in features)
        return {k:torch.tensor([r[k]+[padding]*(length-len(r[k])) for r in features],dtype=torch.long)
            for k,padding in (("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100))}

    class LeanTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels = inputs["labels"]
            base = model.get_base_model()
            out = base.model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],use_cache=False,return_dict=True)
            loss = chunked_ce(out.last_hidden_state[:,:-1],base.lm_head.weight,labels[:,1:],config["ce_chunk_tokens"])
            return (loss,{"loss":loss}) if return_outputs else loss

    class Progress(TrainerCallback):
        def __init__(self,tag):self.tag=tag
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            visual=model.get_base_model().model.visual
            assert not any(x.requires_grad for x in visual.parameters())
            visual.to("cpu")
            torch.cuda.empty_cache()
        def on_substep_end(self,args,state,control,**kwargs):torch.cuda.empty_cache()
        def on_step_end(self,args,state,control,**kwargs):torch.cuda.empty_cache()
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs and ("loss" in logs or "eval_loss" in logs):
                status("training",stage=self.tag,epoch=state.epoch,optimizer_step=state.global_step,
                    logs=logs,gpu_allocated_gib=round(torch.cuda.memory_allocated()/2**30,3),
                    gpu_reserved_gib=round(torch.cuda.memory_reserved()/2**30,3))

    def fresh_model():
        set_seed(config["seed"])
        model = AutoModelForImageTextToText.from_pretrained(p.MODEL,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")
        model.config.use_cache=False;model.config.text_config.use_cache=False
        lora = config["lora"]
        model = get_peft_model(model,LoraConfig(r=lora["r"],lora_alpha=lora["alpha"],lora_dropout=lora["dropout"],
            bias="none",target_modules=lora["target_modules"]))
        trainable = [(name,x.numel()) for name,x in model.named_parameters() if x.requires_grad]
        assert sum(x for _,x in trainable)==917504 and all("lora_" in name and "language_model" in name for name,_ in trainable)
        model.enable_input_require_grads()
        return model

    def fit(folder,train_ids,val_ids,learning_rate,epochs,tag,max_steps=-1):
        marker=folder/"result.json"
        if marker.exists():return p.read(marker)
        if fit_task is None:
            return worker_fit(folder,train_ids,val_ids,learning_rate,epochs,tag,max_steps,config)
        folder.mkdir(parents=True,exist_ok=True)
        status("loading_model",stage=tag,learning_rate=learning_rate)
        model=fresh_model();torch.cuda.reset_peak_memory_stats()
        checkpoints=sorted(folder.glob("checkpoint-*"),key=lambda x:int(x.name.split("-")[-1]))
        args=TrainingArguments(output_dir=str(folder),num_train_epochs=epochs,max_steps=max_steps,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=config["gradient_accumulation"],
            learning_rate=learning_rate,optim="adamw_torch",weight_decay=config["weight_decay"],lr_scheduler_type="constant",
            max_grad_norm=1.,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
            eval_strategy="epoch" if val_ids else "no",save_strategy="epoch",logging_strategy="steps",logging_steps=5,
            prediction_loss_only=True,label_names=["labels"],report_to="none",seed=42,data_seed=42,remove_unused_columns=False,
            disable_tqdm=True,save_total_limit=None)
        trainer=LeanTrainer(model=model,args=args,train_dataset=[records[n] for n in train_ids],
            eval_dataset=[records[n] for n in val_ids] if val_ids else None,data_collator=collate,callbacks=[Progress(tag)])
        trainer.model_accepts_loss_kwargs=False
        started=time.monotonic()
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        evaluations=[r for r in trainer.state.log_history if "eval_loss" in r]
        assert all(math.isfinite(r["eval_loss"]) for r in evaluations)
        if val_ids and max_steps<0:assert len(evaluations)==epochs
        trainer.save_model(str(folder/"adapter"));tokenizer.save_pretrained(folder/"adapter")
        trainer.save_state()
        result={"tag":tag,"learning_rate":learning_rate,"epochs":epochs,"train_count":len(train_ids),"validation_count":len(val_ids),
            "eval_loss_by_epoch":[r["eval_loss"] for r in evaluations],"elapsed_seconds":time.monotonic()-started,
            "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30,"optimizer_steps":trainer.state.global_step,
            "resumed_from_checkpoint":str(checkpoints[-1]) if checkpoints else None,
            "elapsed_seconds_scope":"current_invocation_only",
            "adapter":str(folder/"adapter"),"source_ids":train_ids,"test_used_for_training":False}
        p.save(marker,result)
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        return result

    if fit_task is not None:
        assert set(fit_task)=={"folder","train_ids","val_ids","learning_rate","epochs","tag","max_steps"}
        folder=Path(fit_task["folder"]).resolve()
        assert p.RUN.resolve() in folder.parents
        assert set(fit_task["train_ids"])<=set(records) and set(fit_task["val_ids"])<=set(records)
        assert not set(fit_task["train_ids"]) & set(fit_task["val_ids"])
        assert len(fit_task["train_ids"])==len(set(fit_task["train_ids"]))
        assert len(fit_task["val_ids"])==len(set(fit_task["val_ids"]))
        assert config["learning_rate_range"][0]<=fit_task["learning_rate"]<=config["learning_rate_range"][1]
        assert 1<=fit_task["epochs"]<=config["epochs_per_fold"]
        fit(**{**fit_task,"folder":folder})
        return

    # 최장 개발 작품으로 두 번 업데이트해 전체 길이의 GPU 경로를 먼저 확인한다.
    longest=max(records,key=lambda n:len(records[n]["input_ids"]))
    smoke=fit(p.RUN/"isolated_smoke",[longest],[],8e-5,1,"isolated_smoke",max_steps=2)
    assert smoke["optimizer_steps"]==2
    p.save(p.RUN/"smoke_passed.json",smoke)
    storage="sqlite:///"+(p.RUN/"optuna.sqlite3").as_posix()
    study=optuna.create_study(study_name="v5_lr_5fold",direction="minimize",storage=storage,load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=config["startup_trials"]))
    if not study.trials:study.enqueue_trial({"learning_rate":config["first_candidate"]})
    def objective(trial):
        learning_rate=trial.suggest_float("learning_rate",*config["learning_rate_range"],log=True)
        results=[]
        for fold in folds:
            folder=p.RUN/f"trial_{trial.number:04d}"/f'fold_{fold["fold"]}'
            result=fit(folder,fold["train_ids"],fold["validation_ids"],learning_rate,config["epochs_per_fold"],
                f'trial_{trial.number:04d}/fold_{fold["fold"]}')
            results.append(result)
        means=[sum(r["eval_loss_by_epoch"][e] for r in results)/5 for e in range(config["epochs_per_fold"])]
        epoch=min(range(len(means)),key=means.__getitem__)+1
        trial.set_user_attr("selected_epoch",epoch);trial.set_user_attr("mean_validation_loss_by_epoch",means)
        p.save(p.RUN/f"trial_{trial.number:04d}"/"result.json",{"learning_rate":learning_rate,
            "folds":results,"mean_validation_loss_by_epoch":means,"selected_epoch":epoch,"objective":means[epoch-1]})
        return means[epoch-1]

    for frozen in study.trials:
        if frozen.state==optuna.trial.TrialState.RUNNING:
            pending=optuna.trial.Trial(study,frozen._trial_id)
            study.tell(pending,objective(pending))
    completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed<config["n_trials"]:study.optimize(objective,n_trials=config["n_trials"]-completed,n_jobs=1)
    best=study.best_trial
    selected={"trial":best.number,"learning_rate":best.params["learning_rate"],"epoch":best.user_attrs["selected_epoch"],
        "mean_validation_ce":best.value,"heldout_test_count":61,"test_used_for_selection":False}
    p.save(p.RUN/"selection.json",selected)
    final=fit(p.RUN/"final_refit",sorted(records),[],selected["learning_rate"],selected["epoch"],"final_refit")
    status("complete",selection=selected,final_adapter=final["adapter"],completed_trials=config["n_trials"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check-ready",action="store_true")
    parser.add_argument("--fit-task",type=Path)
    args=parser.parse_args()
    if args.check_ready:
        result=gpu_ready();print(json.dumps(result,ensure_ascii=False));return
    if args.fit_task:
        assert p.RUN.resolve() in args.fit_task.resolve().parents
        with (p.RUN/"run.lock").open("r+b") as probe:
            probe.seek(0)
            try:msvcrt.locking(probe.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:pass
            else:
                probe.seek(0);msvcrt.locking(probe.fileno(),msvcrt.LK_UNLCK,1)
                raise RuntimeError("부모 orchestrator의 잠금 없이 worker를 실행하지 않습니다.")
        run(p.read(args.fit_task))
        return
    with (p.RUN/"run.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b"0");lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        except Exception as exc:
            status("failed",error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


if __name__=="__main__":main()
