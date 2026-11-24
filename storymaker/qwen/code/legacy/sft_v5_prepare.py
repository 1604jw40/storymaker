"""CPU 전처리: v5 채팅·정답 마스크·기존 5-fold·길이 통계를 검증한다."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/sft_v5"
DATA = ROOT / "outputs/picturebook_dataset_v5"
PROJECT = ROOT
PROCESSED = PROJECT / "data/processed/v5"
MODEL = PROJECT / "qwen/models/original"
PYTHON = Path(sys.executable)
OLD_FOLDS = PROJECT / "5fold_학습률_00008/결과/폴드_구성.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def stats(values):
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),
        mean=statistics.mean(values),maximum=max(values),total=sum(values))


def encode(tokenizer, row):
    messages = row["messages"]
    assert [m["role"] for m in messages] == ["system","user","assistant"]
    assert set(json.loads(messages[1]["content"])) == {"moral"}
    assert set(json.loads(messages[2]["content"])) == {"title","characters","scenes"}
    prefix = tokenizer.apply_chat_template(messages[:2],tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_dict=True)["input_ids"]
    full = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
        enable_thinking=False,return_dict=True)["input_ids"]
    assert full[:len(prefix)] == prefix and len(full) > len(prefix),row["source_id"]
    assert tokenizer.eos_token_id in full[len(prefix):],row["source_id"]
    return {"source_id":row["source_id"],"input_ids":full,"attention_mask":[1]*len(full),
        "labels":[-100]*len(prefix)+full[len(prefix):],"prompt_tokens":len(prefix)}


def check_folds(folds, dev_ids, test_ids):
    seen = []
    assert len(folds) == 5 and not dev_ids & test_ids
    for fold in folds:
        train,val = set(fold["train_ids"]),set(fold["validation_ids"])
        assert len(train) == len(fold["train_ids"]) and len(val) == len(fold["validation_ids"])
        assert not train & val and train | val == dev_ids
        assert not (train | val) & test_ids
        seen.extend(fold["validation_ids"])
    assert len(seen) == len(set(seen)) == 243 and set(seen) == dev_ids


def main():
    from transformers import AutoTokenizer
    data = read(DATA/"dataset.json")
    by_id = {r["source_id"]:r for r in data}
    assert len(data) == len(by_id) == 304
    chats = {}
    for split,expected in (("train",194),("validation",49),("test",61)):
        rows = [json.loads(s) for s in (DATA/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == expected
        for row in rows:
            n = row["source_id"]
            assert by_id[n]["source_split"] == split and n not in chats
            assert json.loads(row["messages"][2]["content"]) == by_id[n]["output"]
            chats[n] = row
    dev_ids = {n for n,r in by_id.items() if r["source_split"] != "test"}
    test_ids = set(by_id)-dev_ids
    folds = read(OLD_FOLDS)
    check_folds(folds,dev_ids,test_ids)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    encoded = {n:encode(tokenizer,row) for n,row in chats.items()}
    count = lambda text:len(tokenizer.encode(text,add_special_tokens=False))
    raw = read(PROJECT/"data/raw/이솝_우화_영어대조_전체.json")
    original = {r["number"]:r for r in raw}
    lengths = {"scope":"304편 기술 통계; 테스트 자료는 학습·하이퍼파라미터 선택에 사용하지 않음",
        "tokenizer":"Qwen3.5-4B 기존 고정 원본; API 과금 토큰과 다름",
        "original_story_text":stats([count(original[n]["story"]) for n in by_id]),
        "v5_story_text":stats([count("\n".join(s["text"] for s in r["output"]["scenes"])) for r in data]),
        "v5_visual_notes":stats([count("\n".join(s["visual_note"] for s in r["output"]["scenes"])) for r in data]),
        "v5_output_json":stats([count(chats[n]["messages"][2]["content"]) for n in by_id]),
        "full_chat_including_template":stats([len(r["input_ids"]) for r in encoded.values()]),
        "assistant_loss_tokens_including_end":stats([sum(t!=-100 for t in r["labels"]) for r in encoded.values()]),
        "scene_counts":stats([len(r["output"]["scenes"]) for r in data])}
    maximum = max(len(encoded[n]["input_ids"]) for n in dev_ids)
    context = math.ceil(maximum/256)*256
    assert context <= 4096,"4096을 넘는 작품을 자르지 않습니다. 길이를 검토하세요."
    files = ["dataset.json","train.jsonl","validation.jsonl","test.jsonl"]
    config = {"version":"v5-sft-lr-5fold","dataset":str(PROCESSED),"dataset_source":str(DATA),
        "dataset_sha256":{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in files},
        "model":str(MODEL),"model_id":"Qwen/Qwen3.5-4B","model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "python":str(PYTHON),"not_before":"2026-11-08T21:00:00+09:00","epochs_per_fold":5,"n_trials":3,"seed":42,
        "learning_rate_range":[2e-5,3e-4],"search_parameters":["learning_rate"],
        "sampler":"TPE","startup_trials":1,"first_candidate":8e-5,
        "objective":"min over common epochs of unweighted mean 5-fold validation CE; no test selection",
        "dtype":"bfloat16","quantization":False,"lora":{"r":8,"alpha":16,"dropout":0.05,
            "target_modules":r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"},
        "micro_batch":1,"gradient_accumulation":4,"weight_decay":0.01,"scheduler":"constant",
        "max_sequence_tokens":context,"truncation":False,"loss":"assistant-only token CE; checkpointed chunks, same objective",
        "ce_chunk_tokens":128,"test_count":61,"development_count":243,
        "legacy_v1_v4_policy":"archive only; excluded from SFT, RAG and discriminator",
        "discriminator_policy":"future v5 model completions only; preserve source_id group and scoring reasons",
        "api_calls_during_sft":False,"rag_policy":"only fold train_ids; held-out source and test excluded",
        "final_refit":"fresh base + LoRA on all development 243 at selected LR/common epoch; test excluded"}
    if (RUN/"config.json").exists():assert read(RUN/"config.json") == config,"기존 설정을 덮어쓰지 않습니다."
    save(RUN/"config.json",config)
    save(RUN/"folds.json",folds)
    save(RUN/"encoded_development.json",[encoded[n] for n in sorted(dev_ids)])
    save(RUN/"token_statistics.json",lengths)
    save(RUN/"inspection_lion_mouse.json",{"actual_chat_record":chats[4],"input":by_id[4]["input"],
        "output":by_id[4]["output"],"full_chat_tokens":len(encoded[4]["input_ids"]),
        "prompt_masked_tokens":encoded[4]["prompt_tokens"],"assistant_loss_tokens":sum(x!=-100 for x in encoded[4]["labels"])})
    save(RUN/"status.json",{"state":"prepared_waiting_gpu","epochs_per_fold":5,"n_trials":3,"test_used_for_training":False})
    notes = ("v5 SFT 운영\nPython: "+str(PYTHON)+"\n실행기: "+str(ROOT/"storymaker.qwen.code.legacy.sft_v5_train.py")+
        "\n현재 상태: CPU 전처리 완료. GPU smoke 미실행. LLaDA 종료 및 GPU 여유 확인 후 시작.\n"
        "5-fold는 기존 원작 분할과 동일: 개발 243, 테스트 61 고정. 폴드당 5에포크, 우선 학습률 3후보.\n"
        "Optuna 검색은 learning_rate만 2e-5~3e-4 로그 범위. 첫 후보 8e-5, TPE 초기 1회 후 적응 탐색.\n"
        "이전 Optuna 점수·이전 어댑터는 재사용하지 않는다. 후보·폴드마다 새 원본·LoRA·옵티마이저.\n"
        "입력 system+교훈, 정답 title/characters/scenes 전체 JSON. system/user/pad는 -100, assistant와 종료 토큰만 CE.\n"
        "512 제한 사용 금지. 전체 길이를 검증하고 자르지 않는다. BF16·비양자화·기존 LoRA 설정.\n"
        "긴 JSON의 메모리를 줄이기 위해 LM head CE를 분할·체크포인트한다. dense CE와 손실·기울기 동등 검사 필요.\n"
        "GPU 시작 조건: 호스트 CIM에서 run_benchmarks.py 종료, 13GiB 이상 여유·사용률 15% 이하 연속 확인.\n"
        "다른 GPU 작업을 종료하지 않는다. 준비·검사 완료 후 --check-ready로 확인하고 숨김 실행한다.\n"
        "run.lock 및 호스트 PID/명령줄로 중복 방지. os.kill(pid,0) 사용 금지.\n"
        "API·RAG·판별자 학습은 SFT 후 별도 단계. 이전 버전은 학습·RAG·판별자에서 제외.\n"
        "자동 감시 ID: v5-sft-gpu. 유휴 대기 중 변경 없으면 알림 없이 유지. 시작/실패/완료 때 알림.\n")
    (RUN/"OPERATOR_NOTES.txt").write_text(notes,encoding="utf-8")
    print(json.dumps({"config":config,"token_statistics":lengths},ensure_ascii=False),flush=True)


if __name__ == "__main__":main()
