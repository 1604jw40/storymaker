"""30초 간격으로 SFT를 관찰하고, 확인된 중단만 같은 체크포인트에서 복구한다."""

import argparse
import hashlib
import json
import msvcrt
import os
import sqlite3
import subprocess
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
SFT_WORKERS = ("sft_v5_train", "hpo_sft")
INTERVAL = 30
REQUIRED = (
    "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
    "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
)
TRANSIENT = (
    "cuda out of memory", "cuda error: out of memory",
    "cuda error: unspecified launch failure", "cuda error: device is lost",
    "cuda driver error", "cuda-capable device(s) is/are busy or unavailable",
    "worker 종료 후 gpu 메모리가 충분히 반환되지 않았습니다",
)
HOLD_PHASES = {
    "waiting_checkpoint", "stopping_after_saved_checkpoint", "paused_until_2300",
    "paused_waiting_user", "waiting_gpu_after_2300", "resume_launched",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optional(path):
    return p.read(path) if path.exists() else {}


def lock_held(path):
    if not path.exists():
        return False
    with path.open("r+b") as handle:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def processes(run):
    code = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        "$ErrorActionPreference='Stop'; "
        "$x=@(Get-CimInstance Win32_Process | Where-Object {"
        "$_.Name -match '^python(w)?\\.exe$'} | "
        "Select-Object ProcessId,ParentProcessId,CommandLine); "
        "ConvertTo-Json -InputObject $x -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", code],
        capture_output=True, check=True, timeout=20,
    )
    rows = json.loads(result.stdout.decode("utf-8-sig"))
    if isinstance(rows, dict):
        rows = [rows]
    scripts = [f"storymaker.qwen.code.{name}" for name in SFT_WORKERS]
    scripts += [str(p.ROOT / "qwen/code" / f"{name}.py") for name in SFT_WORKERS]
    return [r for r in rows if any(str(script).casefold() in (r.get("CommandLine") or "").casefold() for script in scripts)
            and str(run).casefold() in r["CommandLine"].casefold()
            and "--check-ready" not in r["CommandLine"] and "--prepare-from" not in r["CommandLine"]]


def gpu_sample():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    values = map(int, output.strip().splitlines()[0].split(","))
    return dict(zip(("utilization_percent", "used_mib", "free_mib", "temperature_c"), values))


def readiness(run, config):
    result = subprocess.check_output(
        p.worker_command("sft_v5_train.py", "--run-dir", run, "--check-ready", python=config["python"]),
        cwd=p.ROOT.parent, timeout=30,
    )
    return json.loads(result.decode("utf-8-sig"))


def gate(control, status, pause, pointer, run):
    if not control.get("enabled", False):
        return "disabled_by_operator"
    if not pointer.get("run_dir") or (p.ROOT / Path(pointer["run_dir"])).resolve() != run:
        return "active_run_changed"
    if (status.get("state") in {"user_paused", "paused_waiting_user"}
            or pause.get("explicit_user_resume_required")
            or pause.get("phase") in HOLD_PHASES):
        return "user_hold"
    if status.get("state") in {"lr_search_complete", "complete"}:
        return "phase_complete"
    return None


def check_checkpoint(folder, rate):
    """저장이 덜 된 최신 폴더를 발견하면 덮어쓰거나 이전 폴더로 숨기지 않는다."""
    paths = sorted(folder.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    if not paths:
        return {"checkpoint": None, "initial_step": 0, "sha256": {}}
    checkpoint = paths[-1]
    for name in REQUIRED:
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f"Incomplete checkpoint: {checkpoint / name}")
    state = p.read(checkpoint / "trainer_state.json")
    step = int(checkpoint.name.split("-")[-1])
    if state["global_step"] != step or state["epoch"] <= 0:
        raise ValueError("Checkpoint position mismatch")
    # 프로젝트가 저장한 로컬 파일만 CPU에서 검사하며 CUDA 모델을 만들지 않는다.
    import torch
    from safetensors import safe_open

    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        if not saved.keys():
            raise ValueError("Empty adapter")
        for key in saved.keys():
            saved.get_tensor(key)
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    if not optimizer["state"] or any(group["lr"] != rate for group in optimizer["param_groups"]):
        raise ValueError("Checkpoint optimizer/LR mismatch")
    for name in ("scheduler.pt", "rng_state.pth", "training_args.bin"):
        torch.load(checkpoint / name, map_location="cpu", weights_only=False)
    return {"checkpoint": str(checkpoint), "initial_step": step,
            "sha256": {str(checkpoint / name): digest(checkpoint / name) for name in REQUIRED}}


def resume_plan(run, config, launcher):
    if config.get("phase") == "hpo":
        from storymaker.qwen.code import hpo_sft
        return hpo_sft.resume_plan(run, config)
    eligible = launcher["eligible_trial_numbers"]
    if eligible != [0, 2, 3] or launcher["excluded_trial_numbers"] != [1]:
        raise ValueError("Unexpected LR study scope")
    with closing(sqlite3.connect((run / "optuna.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        trials = db.execute("SELECT trial_id,number,state FROM trials ORDER BY number").fetchall()
        pending = [row for row in trials if row[1] in eligible and row[2] != "COMPLETE"]
        if not pending or pending[0][2] not in {"RUNNING", "WAITING", "FAIL"}:
            raise ValueError("No safely resumable LR trial")
        trial_id, number, trial_state = pending[0]
        if sum(row[2] in {"RUNNING", "FAIL"} for row in pending) > 1:
            raise ValueError("Ambiguous active/failed trial")
        rate = config["candidate_learning_rates"][eligible.index(number)]
        param = db.execute(
            "SELECT param_value FROM trial_params WHERE trial_id=? AND param_name='learning_rate'",
            (trial_id,),
        ).fetchone()
        if trial_state != "WAITING" and (not param or param[0] != rate):
            raise ValueError("Optuna LR differs from approved queue")
    for fold in range(1, 6):
        folder = run / f"trial_{number:04d}" / f"fold_{fold}"
        if not (folder / "result.json").exists():
            return {"trial_id": trial_id, "trial_number": number, "trial_state": trial_state,
                    "stage": str(folder.relative_to(run)).replace("\\", "/"),
                    "learning_rate": rate, **check_checkpoint(folder, rate)}
    return {"trial_id": trial_id, "trial_number": number, "trial_state": trial_state,
            "stage": f"trial_{number:04d}", "learning_rate": rate,
            "checkpoint": None, "initial_step": 0, "sha256": {}}


def recover_failed_trial(db_path, plan):
    if plan["trial_state"] != "FAIL":
        return
    # Optuna가 worker 예외로 닫은 동일 trial만 복원한다. 새 후보를 생성하지 않는다.
    with closing(sqlite3.connect(db_path, timeout=10)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT number,state FROM trials WHERE trial_id=?", (plan["trial_id"],)).fetchone()
        if row != (plan["trial_number"], "FAIL") or db.execute(
            "SELECT COUNT(*) FROM trials WHERE state='RUNNING'"
        ).fetchone()[0]:
            raise ValueError("Optuna state changed during recovery")
        db.execute("UPDATE trials SET state='RUNNING', datetime_complete=NULL WHERE trial_id=?",
                   (plan["trial_id"],))


def can_retry(status, stderr):
    text = (status.get("error", "") + "\n" + stderr).casefold()
    transient = any(message in text for message in TRANSIENT)
    if status.get("state") == "failed" or "traceback (most recent call last)" in text:
        return transient
    return status.get("state") in {"training", "loading_model", "starting_worker", "waiting_gpu"}


def publish(run, memory, phase, **values):
    changed = memory.get("phase") != phase
    memory.update(phase=phase, updated_at_utc=now(), monitor_pid=os.getpid(), **values)
    p.save(run / "watchdog/state.json", memory)
    if changed:
        with (run / "watchdog/events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(memory, ensure_ascii=False) + "\n")


def stop_reason(run):
    pointer = optional(p.ROOT / "qwen/configs/active_run.json")
    if not pointer:
        pointer = optional(p.ROOT.parent / "outputs/sft_v5/active_run.json")
    return gate(p.read(run / "watchdog/control.json"), p.read(run / "status.json"),
                optional(run / "pause_20261110/state.json"), pointer, run)


def tick(run, memory, observe_only=False):
    reason = stop_reason(run)
    if reason:
        publish(run, memory, reason)
        return True
    status = p.read(run / "status.json")
    rows, held, gpu = processes(run), lock_held(run / "run.lock"), gpu_sample()
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["updated_at_utc"])).total_seconds()
    alerts = []
    if gpu["temperature_c"] >= 85:
        alerts.append("high_temperature")
    if age > 1200:
        alerts.append("no_recent_progress_inspect_without_killing")
    memory.update(training=status, processes=rows, run_lock_held=held, gpu=gpu, alerts=alerts)
    if rows or held:
        memory["missing_polls"] = memory["ready_polls"] = 0
        pending = memory.get("pending_resume")
        logged_rate = status.get("learning_rate", status.get("logs", {}).get("learning_rate"))
        if pending and status["updated_at_utc"] > pending["started_at_utc"] and logged_rate == pending["learning_rate"] and (
            (status.get("stage") == pending["stage"] and status.get("optimizer_step", 0) > pending["initial_step"])
            or (status.get("stage") != pending["stage"] and status.get("optimizer_step", 0) > 0
                and (run / pending["stage"] / "result.json").exists())
        ):
            pending["verified_at_utc"] = now()
            p.save(Path(pending["receipt_dir"]) / "verified.json", status)
            memory["last_verified_resume"] = memory.pop("pending_resume")
        publish(run, memory, "resume_verification_pending" if memory.get("pending_resume") else "monitoring")
        return False
    memory["missing_polls"] = memory.get("missing_polls", 0) + 1
    if observe_only or memory["missing_polls"] < 2:
        publish(run, memory, "confirming_exit")
        return False
    if memory.get("needs_attention"):
        publish(run, memory, "needs_attention")
        return False
    with (run / "stderr.log").open("rb") as handle:
        handle.seek(max(memory["stderr_offset"], handle.seek(0, 2) - 65536))
        errors = handle.read().decode("utf-8", errors="replace")
    if not can_retry(status, errors):
        publish(run, memory, "needs_attention", needs_attention="Non-transient or unclassified error", error_tail=errors[-6000:])
        return False
    attempts = [stamp for stamp in memory.get("restart_times", [])
                if time.time() - datetime.fromisoformat(stamp).timestamp() < 3600]
    if len(attempts) >= 3:
        publish(run, memory, "needs_attention", needs_attention="Three restarts within one hour")
        return False
    manifest = p.read(run / "watchdog/protected_sha256.json")
    try:
        if any(digest(path) != value for path, value in manifest.items()):
            raise ValueError("Protected data/config/code changed")
        config, launcher = p.read(run / "config.json"), p.read(run / "launcher.json")
    except Exception as exc:
        publish(run, memory, "needs_attention", needs_attention=str(exc))
        return False
    ready = readiness(run, config)
    memory["ready_polls"] = memory.get("ready_polls", 0) + 1 if ready["ready"] and gpu["temperature_c"] < 80 else 0
    if memory["ready_polls"] < 2:
        publish(run, memory, "waiting_gpu", readiness=ready)
        return False
    try:
        plan = resume_plan(run, config, launcher)
    except Exception as exc:
        publish(run, memory, "needs_attention", needs_attention=str(exc))
        return False
    if (stop_reason(run) or processes(run) or lock_held(run / "run.lock")
            or not readiness(run, config)["ready"]):
        return False
    receipt = run / "watchdog" / datetime.now(timezone.utc).strftime("resume_%Y%m%dT%H%M%SZ")
    receipt.mkdir()
    with closing(sqlite3.connect((run / "optuna.sqlite3").as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(receipt / "optuna_before.sqlite3")) as backup:
            source.backup(backup)
    p.save(receipt / "before.json", {"status": status, "launcher": launcher, "plan": plan, "error_tail": errors[-6000:]})
    recover_failed_trial(run / "optuna.sqlite3", plan)
    started = now()
    memory["stderr_offset"] = (run / "stderr.log").stat().st_size
    memory["restart_times"] = attempts + [started]
    publish(run, memory, "preparing_resume")
    script = "hpo_sft.py" if config.get("phase") == "hpo" else "sft_v5_train.py"
    with (run / "stdout.log").open("ab") as output, (run / "stderr.log").open("ab") as error:
        child = subprocess.Popen(
            p.worker_command(script, "--run-dir", run, python=config["python"]),
            cwd=p.ROOT.parent, stdin=subprocess.DEVNULL, stdout=output, stderr=error,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    p.save(run / "launcher.json", {**launcher, "pid": child.pid, "started_at_utc": started,
           "watchdog_resume_started_at_utc": started, "resume_checkpoint": plan["checkpoint"],
           "watchdog_receipt": str(receipt)})
    memory.update(missing_polls=0, ready_polls=0,
                  pending_resume={**plan, "started_at_utc": started, "pid": child.pid, "receipt_dir": str(receipt)})
    p.save(receipt / "launch.json", memory["pending_resume"])
    publish(run, memory, "resume_launched")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="observe only; never launch training")
    args = parser.parse_args()
    run = p.sft_result_path(args.run_dir)
    with (run / "watchdog/watchdog.lock").open("a+b") as handle:
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        memory = optional(run / "watchdog/state.json")
        memory.setdefault("stderr_offset", (run / "stderr.log").stat().st_size)
        while True:
            try:
                done = tick(run, memory, observe_only=args.once)
            except Exception as exc:
                publish(run, memory, "inspection_error", error=f"{type(exc).__name__}: {exc}")
                done = False
            if args.once or done:
                return
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
