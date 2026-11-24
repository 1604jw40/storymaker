"""선택 LR 고정: fold1 후보 비교 → 가장 유망한 후보의 5fold 확인 → 전체 개발 재학습."""

import argparse
import hashlib
import math
import msvcrt
import shutil
import time
import traceback
from copy import deepcopy
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code import sft_v5_train as training
CANDIDATES = [
    {"name": "dropout_010", "hyperparameters": {"lora_dropout": 0.10, "weight_decay": 0.01}},
    {"name": "weight_decay_010", "hyperparameters": {"lora_dropout": 0.05, "weight_decay": 0.10}},
]
BASELINE = {"lora_dropout": 0.05, "weight_decay": 0.01}
GRID_CANDIDATES = CANDIDATES + [
    {"name": "dropout_010_weight_decay_010", "hyperparameters": {"lora_dropout": 0.10, "weight_decay": 0.10}},
]


def approved_candidates(config):
    if config.get("screening_design") not in (None, "2x2_grid"):
        raise ValueError("Unknown HPO screening design")
    expected = GRID_CANDIDATES if config.get("screening_design") == "2x2_grid" else CANDIDATES
    if config.get("hpo_candidates") != expected:
        raise ValueError("Unexpected HPO candidate combinations")
    return expected


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_epoch(results):
    curves = [r["eval_loss_by_epoch"] for r in results]
    if not curves or any(len(c) != 10 or not all(math.isfinite(x) for x in c) for c in curves):
        raise ValueError("Expected complete ten-epoch validation curves")
    means = [sum(c[e] for c in curves) / len(curves) for e in range(10)]
    epoch = min(range(10), key=means.__getitem__) + 1
    return {"epoch": epoch, "mean_validation_ce": means[epoch - 1], "mean_validation_loss_by_epoch": means}


def screen_choice(baseline, results):
    best = min(results, key=lambda name: common_epoch([results[name]])["mean_validation_ce"])
    if common_epoch([results[best]])["mean_validation_ce"] < common_epoch([baseline])["mean_validation_ce"]:
        return best
    return None


def prepare(run, source, screen_source=None):
    """LR 산출물을 복사/참조하고 새 HPO 기록만 만든다. 기존 LR DB는 변경하지 않는다."""
    import optuna

    from storymaker.qwen.code import watch_sft as w
    if (run / "config.json").exists():
        raise ValueError("HPO run already exists; resume instead of preparing again")
    if w.lock_held(source / "run.lock") or w.lock_held(source / "watchdog/watchdog.lock") or w.processes(source):
        raise ValueError("LR training or watchdog still owns the run")
    if p.read(source / "watchdog/control.json")["enabled"]:
        raise ValueError("Disable the LR watchdog before transitioning")
    if p.read(source / "status.json")["state"] != "lr_search_complete":
        raise ValueError("LR comparison has not completed")
    selected = p.read(source / "selection.json")
    comparisons = []
    for number in (0, 2, 3):
        results = [p.read(source / f"trial_{number:04d}/fold_{f}/result.json") for f in range(1, 6)]
        if any(r["epochs"] != 10 or r["optimizer_steps"] != 490 for r in results):
            raise ValueError("Incomplete LR baseline")
        comparisons.append({"trial": number, "learning_rate": results[0]["learning_rate"], **common_epoch(results)})
    best = min(comparisons, key=lambda r: r["mean_validation_ce"])
    if any(selected[k] != best[k] for k in ("trial", "learning_rate", "epoch", "mean_validation_ce")):
        raise ValueError("LR selection does not match recomputed common-epoch means")
    if selected["test_used_for_selection"] or selected["heldout_test_count"] != 61:
        raise ValueError("Invalid test isolation")
    config = deepcopy(p.read(source / "config.json"))
    config.update(phase="hpo", version="v5-fixed-lr-dropout-weight-decay", source_lr_run=str(source),
                  fixed_learning_rate=selected["learning_rate"], selected_lr_trial=selected["trial"],
                  search_parameters=["lora_dropout", "weight_decay"], n_trials=2,
                  study_name="v5_fixed_lr_fold1_hpo", hpo_candidates=deepcopy(CANDIDATES),
                  baseline_hyperparameters=BASELINE, screening_fold=1,
                  confirmation_policy="best improving fold1 candidate only; reuse its fold1; compare full5fold with baseline",
                  objective="screen fold1; select by common-epoch mean of all5fold; test excluded")
    reused_files = []
    if screen_source is not None:
        screen_source = p.sft_result_path(screen_source)
        if screen_source in (run, source):
            raise ValueError("Invalid prior screening run")
        if w.lock_held(screen_source / "run.lock") or w.lock_held(screen_source / "watchdog/watchdog.lock") or w.processes(screen_source):
            raise ValueError("Prior screening run still has a live owner")
        if p.read(screen_source / "watchdog/control.json")["enabled"]:
            raise ValueError("Prior screening watchdog must be disabled")
        if p.read(screen_source / "status.json")["state"] != "screening_complete_for_grid_extension":
            raise ValueError("Prior screening has not completed its controlled handoff")
        previous = p.read(screen_source / "config.json")
        if previous.get("hpo_candidates") != CANDIDATES or any(previous.get(k) != v for k, v in config.items()):
            raise ValueError("Prior screening settings differ from the approved single-change run")
        for name in ("folds.json", "encoded_development.json"):
            if digest(screen_source / name) != digest(source / name):
                raise ValueError("Prior screening data/split mismatch")
        fold = p.read(source / "folds.json")[0]
        reused = {}
        for candidate in CANDIDATES:
            tag = f"screen/{candidate['name']}/fold_1"
            result_file = screen_source / tag / "result.json"
            validate_result(p.read(result_file), previous, fold["train_ids"], fold["validation_ids"],
                            candidate["hyperparameters"], 10, config_path=screen_source / "config.json")
            reused[tag] = {"result": str(result_file), "config": str(screen_source / "config.json")}
            reused_files.append(result_file)
        config.update(version="v5-fixed-lr-2x2-grid", n_trials=len(GRID_CANDIDATES),
                      study_name="v5_fixed_lr_fold1_2x2_grid", hpo_candidates=deepcopy(GRID_CANDIDATES),
                      screening_design="2x2_grid", source_screen_run=str(screen_source), reused_screen_results=reused)
        reused_files += [screen_source / name for name in ("config.json", "optuna.sqlite3", "folds.json", "encoded_development.json")]
    candidates = approved_candidates(config)
    p.save(run / "config.json", config)
    for name in ("folds.json", "encoded_development.json"):
        shutil.copy2(source / name, run / name)
    baseline_files = [source / name for name in ("config.json", "selection.json", "folds.json", "encoded_development.json", "optuna.sqlite3")]
    baseline_files += list((source / f"trial_{selected['trial']:04d}").glob("fold_*/result.json"))
    baseline_files += reused_files
    p.save(run / "baseline_sha256.json", {str(path): digest(path) for path in baseline_files})
    p.save(run / "lr_comparison.json", comparisons)
    study = optuna.create_study(study_name=config["study_name"], direction="minimize", storage="sqlite:///" + (run / "optuna.sqlite3").as_posix())
    for candidate in candidates:
        study.enqueue_trial({"candidate": candidate["name"]})
    p.save(run / "active_task.json", {"phase": "screen", "candidate": CANDIDATES[0]["name"],
            "fold": 1, "stage": "screen/dropout_010/fold_1", "trial_number": 0,
            "hyperparameters": CANDIDATES[0]["hyperparameters"], "epochs": 10})
    p.save(run / "status.json", {"state": "prepared", "updated_at_utc": w.now(), "phase": "hpo"})
    return config


def context():
    config, records, folds = training.inputs()
    source = Path(config["source_lr_run"])
    approved_candidates(config)
    if config.get("phase") != "hpo" or config["baseline_hyperparameters"] != BASELINE:
        raise ValueError("Unexpected HPO scope")
    original = p.read(source / "config.json")
    changed = {"version", "search_parameters", "n_trials", "study_name", "objective"}
    if any(config[k] != v for k, v in original.items() if k not in changed):
        raise ValueError("HPO changed a fixed training setting")
    if config["fixed_learning_rate"] != p.read(source / "selection.json")["learning_rate"]:
        raise ValueError("HPO must use the selected LR")
    for filename in ("baseline_sha256.json", "watchdog/protected_sha256.json"):
        for path, expected in p.read(p.RUN / filename).items():
            if digest(path) != expected:
                raise ValueError(f"Protected input changed: {path}")
    baseline = [p.read(source / f"trial_{config['selected_lr_trial']:04d}/fold_{f}/result.json") for f in range(1, 6)]
    for result, fold in zip(baseline, folds):
        if result["source_ids"] != fold["train_ids"] or result["validation_count"] != len(fold["validation_ids"]):
            raise ValueError("Baseline split mismatch")
    return config, records, folds, baseline


def validate_result(result, config, train_ids, val_ids, params, epochs, *, config_path=None):
    reused = config.get("reused_screen_results", {}).get(result.get("tag"))
    if reused:
        if result != p.read(reused["result"]):
            raise ValueError("Reused screening result differs from its immutable source")
        config_path = Path(reused["config"])
    expected = {"learning_rate": config["fixed_learning_rate"], "epochs": epochs,
                "source_ids": train_ids, "train_count": len(train_ids), "validation_count": len(val_ids),
                "hyperparameters": params, "run_config_sha256": digest(config_path or p.RUN / "config.json"),
                "optimizer_steps": math.ceil(len(train_ids) / config["gradient_accumulation"]) * epochs,
                "test_used_for_training": False}
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Cached fit result does not match this task/config")
    if val_ids:
        common_epoch([result])
    elif result["eval_loss_by_epoch"]:
        raise ValueError("Final refit must not select using validation/test")
    if not (Path(result["adapter"]) / "adapter_model.safetensors").is_file():
        raise ValueError("Completed fit adapter missing")


def resume_plan(run, config):
    """감시자는 현재 HPO 작업만 이어가며 LR DB와 제외 trial에는 접근하지 않는다."""
    import optuna
    from storymaker.qwen.code import watch_sft as w
    active = p.read(run / "active_task.json")
    candidates = {c["name"]: c["hyperparameters"] for c in config["hpo_candidates"]}
    phase, name, fold = active["phase"], active["candidate"], active["fold"]
    if phase == "screen":
        if fold != 1 or name not in candidates or active["epochs"] != 10:
            raise ValueError("Invalid screening task")
        stage = f"screen/{name}/fold_1"
        expected_params = candidates[name]
    elif phase == "confirmation":
        if name != p.read(run / "screening.json")["promising_candidate"] or fold not in range(2, 6):
            raise ValueError("Invalid confirmation task")
        stage, expected_params = f"confirmation/{name}/fold_{fold}", candidates[name]
        if active["epochs"] != 10:
            raise ValueError("Confirmation must use ten epochs")
    elif phase == "final_refit":
        selection = p.read(run / "selection.json")
        stage, expected_params = "final_refit", selection["hyperparameters"]
        if active["epochs"] != selection["epoch"] or fold is not None or name != selection["candidate"]:
            raise ValueError("Invalid final refit task")
    else:
        raise ValueError("Unknown HPO phase")
    if active["stage"] != stage or active["hyperparameters"] != expected_params:
        raise ValueError("Active HPO task differs from its approved stage")
    plan = {"stage": stage, "learning_rate": config["fixed_learning_rate"], "trial_state": "COMPLETE"}
    if phase == "screen":
        study = optuna.load_study(study_name=config["study_name"], storage="sqlite:///" + (run / "optuna.sqlite3").as_posix())
        trials = study.trials
        if len(trials) != len(approved_candidates(config)) or sum(t.state.name in {"RUNNING", "FAIL"} for t in trials) > 1:
            raise ValueError("Ambiguous HPO study")
        trial = trials[active["trial_number"]]
        if trial.params.get("candidate", trial.system_attrs.get("fixed_params", {}).get("candidate")) != name:
            raise ValueError("HPO trial differs from active candidate")
        plan.update(trial_id=trial._trial_id, trial_number=trial.number, trial_state=trial.state.name)
    saved = w.check_checkpoint(run / stage, config["fixed_learning_rate"])
    if saved["checkpoint"]:
        import torch
        checkpoint = Path(saved["checkpoint"])
        if p.read(checkpoint / "adapter_config.json")["lora_dropout"] != expected_params["lora_dropout"]:
            raise ValueError("Checkpoint dropout mismatch")
        args = torch.load(checkpoint / "training_args.bin", map_location="cpu", weights_only=False)
        if args.weight_decay != expected_params["weight_decay"]:
            raise ValueError("Checkpoint weight decay mismatch")
    return {**plan, **saved}


def fit(config, records, folds, params, phase, name, fold=None, trial_number=None, epochs=10):
    if fold is None:
        stage, train_ids, val_ids = "final_refit", sorted(records), []
    else:
        split = folds[fold - 1]
        stage = f"{phase}/{name}/fold_{fold}"
        train_ids, val_ids = split["train_ids"], split["validation_ids"]
    p.save(p.RUN / "active_task.json", {"phase": phase, "candidate": name, "fold": fold,
           "stage": stage, "trial_number": trial_number, "hyperparameters": params, "epochs": epochs})
    folder = p.RUN / stage
    reused = config.get("reused_screen_results", {}).get(stage)
    if reused:
        result = p.read(reused["result"])
        validate_result(result, config, train_ids, val_ids, params, epochs)
        if (folder / "result.json").exists() and p.read(folder / "result.json") != result:
            raise ValueError("Local reused result differs from source")
        p.save(folder / "result.json", result)
        p.save(folder / "reuse.json", {**reused, "source_result_sha256": digest(reused["result"]),
                                      "source_config_sha256": digest(reused["config"]), "new_gpu_training": False})
    elif (folder / "result.json").exists():
        result = p.read(folder / "result.json")
    else:
        resume_plan(p.RUN, config)
        result = training.worker_fit(folder, train_ids, val_ids, config["fixed_learning_rate"],
                                     epochs, stage, -1, config, hyperparameters=params)
    validate_result(result, config, train_ids, val_ids, params, epochs)
    return result


def run():
    import optuna

    config, records, folds, baseline = context()
    if p.read(p.RUN / "status.json")["state"] == "complete":
        return
    for _ in range(2):
        ready = training.gpu_ready()
        if not ready["ready"]:
            training.status("waiting_gpu", gpu=ready)
            return
        time.sleep(5)
    study = optuna.load_study(study_name=config["study_name"], storage="sqlite:///" + (p.RUN / "optuna.sqlite3").as_posix())
    candidates = approved_candidates(config)
    names = [c["name"] for c in candidates]
    if len(study.trials) != len(names) or any(t.state.name in {"FAIL", "PRUNED"} for t in study.trials):
        raise ValueError("HPO study requires inspection before resuming")
    for frozen in study.trials:
        if frozen.state.name == "COMPLETE":
            continue
        trial = optuna.trial.Trial(study, frozen._trial_id) if frozen.state.name == "RUNNING" else study.ask()
        name = trial.suggest_categorical("candidate", names)
        params = next(c["hyperparameters"] for c in candidates if c["name"] == name)
        try:
            result = fit(config, records, folds, params, "screen", name, 1, trial.number)
            score = common_epoch([result])
            trial.set_user_attr("hyperparameters", params)
            trial.set_user_attr("selected_epoch", score["epoch"])
            if f"screen/{name}/fold_1" in config.get("reused_screen_results", {}):
                trial.set_user_attr("reused_from", config["reused_screen_results"][f"screen/{name}/fold_1"])
            study.tell(trial, score["mean_validation_ce"])
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
    screened = {name: p.read(p.RUN / f"screen/{name}/fold_1/result.json") for name in names}
    for candidate in candidates:
        validate_result(screened[candidate["name"]], config, folds[0]["train_ids"], folds[0]["validation_ids"], candidate["hyperparameters"], 10)
    promising = screen_choice(baseline[0], screened)
    p.save(p.RUN / "screening.json", {"fixed_fold": 1, "baseline": common_epoch([baseline[0]]),
           "candidates": {name: common_epoch([r]) for name, r in screened.items()},
           "promising_candidate": promising, "test_used_for_selection": False})
    selected = {"candidate": "baseline", "hyperparameters": BASELINE, **common_epoch(baseline)}
    confirmed = {"baseline": common_epoch(baseline)}
    if promising:
        params = next(c["hyperparameters"] for c in candidates if c["name"] == promising)
        results = [screened[promising]] + [fit(config, records, folds, params, "confirmation", promising, f) for f in range(2, 6)]
        candidate_score = common_epoch(results)
        confirmed[promising] = candidate_score
        if candidate_score["mean_validation_ce"] < selected["mean_validation_ce"]:
            selected = {"candidate": promising, "hyperparameters": params, **candidate_score}
    p.save(p.RUN / "confirmation.json", confirmed)
    selected.update(learning_rate=config["fixed_learning_rate"], source_lr_trial=config["selected_lr_trial"],
                    heldout_test_count=61, test_used_for_selection=False)
    p.save(p.RUN / "selection.json", selected)
    final = fit(config, records, folds, selected["hyperparameters"], "final_refit", selected["candidate"], epochs=selected["epoch"])
    p.save(p.RUN / "final_result.json", {"selection": selected, "fit": final,
           "dataset_sha256": config["dataset_sha256"], "folds_sha256": digest(p.RUN / "folds.json"),
           "encoded_development_sha256": digest(p.RUN / "encoded_development.json"),
           "model": config["model"], "model_revision": config["model_revision"],
           "test_used_for_training": False, "test_used_for_selection": False,
           "generation_quality_evaluated": False, "rag_api_reward_rl_executed": False})
    training.status("complete", selection=selected, final_adapter=final["adapter"], phase="hpo_and_final_refit_complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--reuse-screen-from", type=Path)
    args = parser.parse_args()
    p.RUN = p.sft_result_path(args.run_dir)
    if args.prepare_from:
        prepare(p.RUN, args.prepare_from.resolve(), args.reuse_screen_from.resolve() if args.reuse_screen_from else None)
        return
    if args.reuse_screen_from:
        raise ValueError("--reuse-screen-from requires --prepare-from")
    with (p.RUN / "run.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run()
        except Exception as exc:
            training.status("failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
