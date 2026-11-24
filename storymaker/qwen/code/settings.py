"""발표·재현용 SFT 설정. 실행 중인 실험은 outputs/.../config.json을 고정해서 읽는다."""

from copy import deepcopy


# ==================== 1. 교차검증·Optuna ====================
# 각 후보는 같은 5-fold에서 학습한다. 최저 fold 하나가 아닌 동일 에포크의 평균 CE로 선택한다.
SEARCH = {
    "epochs_per_fold": 10,
    "n_trials": 3,
    "seed": 42,
    "learning_rate_range": [2e-5, 3e-4],
    "search_parameters": ["learning_rate"],
    "sampler": "fixed_queue",
    "candidate_learning_rates": [8e-5, 5e-5, 2e-4],
    "defer_final_refit": True,
    "startup_trials": 1,
    "first_candidate": 8e-5,
}

# ==================== 2. 모델·LoRA ====================
# 이번 실험은 LR만 탐색한다. rank/dropout 등은 고정하며 새 실험 없이 중간에 바꾸지 않는다.
MODEL = {"dtype": "bfloat16", "quantization": False}
LORA = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
}

# ==================== 3. 배치·옵티마이저 ====================
# GPU 1개에서 micro_batch 1 × 누적 4 = 보통 4편마다 업데이트한다.
OPTIMIZER = {
    "micro_batch": 1,
    "eval_micro_batch": 1,
    "gradient_accumulation": 4,
    "weight_decay": 0.01,
    "scheduler": "constant",
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}

# ==================== 4. 전체 문맥·메모리 ====================
# 정답을 자르지 않는다. 긴 vocabulary logits만 나눠 같은 CE를 계산한다.
MEMORY = {
    "truncation": False,
    "ce_chunk_tokens": 128,
    "gradient_checkpointing": True,
    "checkpoint_use_reentrant": False,
    "logging_steps": 5,
}


def hyperparameters(epochs=10):
    assert epochs in (5, 10)
    return deepcopy(
        {
            **SEARCH,
            **MODEL,
            **OPTIMIZER,
            **MEMORY,
            "lora": LORA,
            "epochs_per_fold": epochs,
        }
    )


def training_arguments(
    config, folder, learning_rate, epochs, max_steps, has_validation
):
    """고정 config를 실제 Trainer 인자로 연결한다. 체크포인트 재개 시 LR 스케줄은 constant다."""
    defaults = {**OPTIMIZER, **MEMORY}
    return {
        "output_dir": str(folder),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": config["micro_batch"],
        "per_device_eval_batch_size": config.get(
            "eval_micro_batch", defaults["eval_micro_batch"]
        ),
        "gradient_accumulation_steps": config["gradient_accumulation"],
        "learning_rate": learning_rate,
        "optim": "adamw_torch",
        "weight_decay": config["weight_decay"],
        "lr_scheduler_type": config["scheduler"],
        **{
            key: config.get(key, defaults[key])
            for key in ("max_grad_norm", "adam_beta1", "adam_beta2", "adam_epsilon")
        },
        "bf16": True,
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {
            "use_reentrant": config.get("checkpoint_use_reentrant", False)
        },
        "eval_strategy": "epoch" if has_validation else "no",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": config.get("logging_steps", 5),
        "prediction_loss_only": True,
        "label_names": ["labels"],
        "report_to": "none",
        "seed": config["seed"],
        "data_seed": config["seed"],
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "save_total_limit": None,
    }
