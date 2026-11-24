"""학습 완료 어댑터로 교훈 하나를 JSON 이야기로 생성한다. API 호출과 가중치 갱신은 없다."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from storymaker import project as p
# ==================== 생성 하이퍼파라미터 ====================
DECODING = {"max_new_tokens": 4096, "temperature": 0.7, "top_p": 0.9}


# ==================== 1. 학습과 같은 입력 형식 ====================
def messages_for(moral):
    moral = moral.strip()
    if not moral:
        raise ValueError("교훈이 비어 있습니다.")
    system = (p.ROOT / "data/source/system_prompt.txt").read_text("utf-8")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"moral": moral}, ensure_ascii=False)},
    ]


# ==================== 2. 생성 결과의 형식 확인 ====================
def parse_story(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(value, dict) or set(value) != {"title", "characters", "scenes"}:
        return value, ["invalid_top_level_fields"]
    if not isinstance(value["characters"], list) or not isinstance(
        value["scenes"], list
    ):
        return value, ["characters_and_scenes_must_be_arrays"]
    if not value["scenes"]:
        return value, ["empty_scenes"]
    if not isinstance(value["title"], str) or not value["title"].strip():
        return value, ["invalid_title"]
    characters = value["characters"]
    if not all(
        isinstance(row, dict)
        and all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ["character_id", "name", "goal", "traits"]
        )
        for row in characters
    ):
        return value, ["invalid_characters"]
    ids = {row["character_id"] for row in characters}
    if len(ids) != len(characters):
        return value, ["duplicate_character_id"]
    problems = []
    for number, scene in enumerate(value["scenes"], 1):
        if not isinstance(scene, dict):
            problems.append(f"invalid_scene_{number}")
            continue
        references = scene.get("character_ids")
        if not isinstance(references, list) or not all(
            isinstance(item, str) and item in ids for item in references
        ):
            problems.append(f"invalid_character_reference_{number}")
        if scene.get("scene_no") != number:
            problems.append(f"invalid_scene_number_{number}")
        if not all(
            isinstance(scene.get(key), str)
            for key in ["setting", "text", "visual_note"]
        ):
            problems.append(f"invalid_scene_fields_{number}")
        mode, body = scene.get("text_mode"), scene.get("text")
        if mode not in {"read_aloud", "wordless"} or (
            (mode == "wordless" and body != "")
            or (
                mode == "read_aloud" and (not isinstance(body, str) or not body.strip())
            )
        ):
            problems.append(f"invalid_text_mode_{number}")
    return value, problems


# ==================== 3. 로컬 모델 추론 ====================
def generate(moral, adapter, max_new_tokens=DECODING["max_new_tokens"]):
    messages = messages_for(moral)
    from storymaker.qwen.code.sft_v5_train import gpu_ready

    if not gpu_ready()["ready"]:
        raise RuntimeError("다른 GPU 학습이 진행 중이거나 여유 메모리가 부족합니다.")
    if adapter is not None and not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"학습 완료 어댑터가 없습니다: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    config = p.read(p.RUN / "config.json")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        config["model"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model = (PeftModel.from_pretrained(base, str(adapter)) if adapter is not None else base).eval()
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    # 어댑터 로딩의 난수 소비와 무관하게 두 모델의 생성 seed를 맞춘다.
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=DECODING["temperature"],
            top_p=DECODING["top_p"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            use_cache=True,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = result[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    story, problems = parse_story(text)
    ended = bool(generated.numel() and int(generated[-1]) == tokenizer.eos_token_id)
    if not ended:
        problems.append("no_eos_before_limit")
    return {
        "moral": moral,
        "adapter": str(adapter) if adapter is not None else None,
        "model_variant": "sft" if adapter is not None else "base",
        "raw_text": text,
        "story": story,
        "format_problems": problems,
        "generated_tokens": generated.numel(),
        "generated_token_ids": generated.tolist(),
        "prompt_tokens": inputs["input_ids"].shape[1],
        "generation_seconds": seconds,
        "tokens_per_second": generated.numel() / seconds,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "ended_with_eos": ended,
        "decoding": {**DECODING, "max_new_tokens": max_new_tokens},
        "seed": config["seed"],
        "base_model_revision": config["model_revision"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_quality_evaluation": "not_run",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=p.RUN)
    parser.add_argument("--moral", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--base", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=p.ROOT / "outputs/generations/example.json"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DECODING["max_new_tokens"]
    )
    args = parser.parse_args()
    p.RUN = args.run_dir.resolve()
    if (p.ROOT / "outputs").resolve() not in p.RUN.parents:
        raise ValueError("Run must remain inside project outputs")
    if args.base and args.adapter is not None:
        raise ValueError("--base cannot be combined with --adapter")
    if args.max_new_tokens <= 0:
        raise ValueError("생성 토큰 상한은 양수여야 합니다.")
    adapter = None if args.base else (args.adapter or p.RUN / "final_refit/adapter")
    result = generate(args.moral, adapter, args.max_new_tokens)
    p.save(args.output, result)
    print(f"생성 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
