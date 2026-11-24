"""두 API 생성 후보를 교훈 → 그림책 시나리오 SFT 샘플로 가공한다. API 호출 없음."""
import hashlib
import json
from pathlib import Path

from storymaker.data_augmentation.code import augment as a
SOURCE_DIR = a.ROOT / "outputs" / "deliverables_20_scenes"
OUTPUT_DIR = a.ROOT / "outputs" / "sft_20_scenes_v1"
FIXED_CONDITIONS = {
    "language": "ko", "age_min": 5, "age_max": 7,
    "reading_mode": "보호자와 함께 읽기",
    "scene_unit": "한 펼침면에 배치할 하나의 주요 시각 장면",
}
SYSTEM_PROMPT = """당신은 만 5~7세가 보호자와 함께 읽는 한국어 우화 그림책의 작가다.
입력 JSON의 moral과 target_scene_count를 바탕으로 그림책 시나리오를 작성한다.
교훈은 인물의 선택과 그 결과를 통해 드러낸다. 글은 자연스러운 한국어 낭독 문장으로 쓴다.
각 장면은 한 펼침면에서 보여줄 하나의 주요 시각적 순간이다. 장면 번호는 1부터 연속으로 매긴다.
출력은 설명이나 코드 블록 없이 다음 필드만 가진 JSON 객체다.
title: 제목.
characters: character_id, name, goal, traits를 가진 등장인물 객체 배열.
scenes: scene_no, character_ids, setting, text_mode, text, visual_note를 가진 장면 객체 배열.
character_ids에는 해당 장면에 등장하는 인물의 character_id를 넣는다. 인물이 없으면 빈 배열이다.
setting은 장면의 장소와 시간, visual_note는 그 순간의 인물·행동·표정·소품 등 그림에 필요한 정보를 담는다.
text_mode는 read_aloud 또는 wordless다. wordless의 text는 빈 문자열이고, read_aloud의 text는 비어 있으면 안 된다.
장면 수는 target_scene_count와 일치시킨다. 장면 수를 채우려고 같은 행동과 의미를 반복하지 않는다.
인물 외형과 공간·소품의 연속성, 글과 그림의 일치를 유지한다."""


def convert(record):
    conditions = record["conditions"]
    if {k: conditions.get(k) for k in FIXED_CONDITIONS} != FIXED_CONDITIONS:
        raise ValueError("공통 프롬프트와 다른 독자·언어·장면 조건")
    if record["status"] != "completed" or record["plan"]["suitability"] == "hold":
        raise ValueError("생성이 완료된 비보류 후보가 필요합니다.")
    if record["source_split"] not in ("train", "validation", "test"):
        raise ValueError("원작 분할이 지정되지 않았습니다.")
    number = record["source"]["number"]
    if type(number) is not int or number < 1:
        raise ValueError("원작 ID 오류")
    plan = a.Plan.model_validate(record["plan"])
    story = a.Story.model_validate(record["story"])
    review = a.Review.model_validate(record["review"])
    a.validate_plan(plan, record["source"], conditions["target_scene_count"])
    a.validate_story(story, plan)
    a.validate_review(review, story)
    example = {
        "id": f"aesop_{number:04d}_scenes_sft_v1",
        "source_id": number,
        "source_split": record["source_split"],
        "input": {"moral": plan.target_moral, "target_scene_count": conditions["target_scene_count"]},
        "output": {
            "title": story.title,
            "characters": [{"character_id": c.character_id, "name": c.source_name,
                            "goal": c.goal, "traits": c.traits} for c in plan.characters],
            "scenes": [{"scene_no": s.scene_no, "character_ids": p.character_ids,
                        "setting": p.setting, "text_mode": s.text_mode,
                        "text": s.text, "visual_note": s.visual_note}
                       for p, s in zip(plan.scenes, story.scenes)],
        },
    }
    metadata = {"id": example["id"], "source_id": number, "source_split": record["source_split"],
                "human_review": record["human_review"], "api_recommendation": record["api_recommendation"],
                "review": review.model_dump(),
                "scene_editorial_notes": [{"scene_no": s.scene_no, "reader_discovery": s.reader_discovery,
                                           "page_turn_role": s.page_turn_role} for s in plan.scenes]}
    return example, metadata


def chat_record(example):
    return {**{k: example[k] for k in ("id", "source_id", "source_split")}, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(example["input"], ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)},
    ]}


def main():
    paths = sorted(SOURCE_DIR.glob("*.json"))
    if len(paths) != 2:
        raise ValueError("이번 가공 범위는 전달된 JSON 두 편입니다.")
    converted = []
    for path in paths:
        content = path.read_bytes()
        example, metadata = convert(json.loads(content))
        metadata.update(source_file=str(path.relative_to(a.ROOT)), source_sha256=hashlib.sha256(content).hexdigest())
        converted.append((path, example, metadata))
    if len({e["id"] for _, e, _ in converted}) != len(converted):
        raise ValueError("중복 학습 샘플 ID")
    grouped = {}
    for path, example, _ in converted:
        a.save_json(OUTPUT_DIR / path.name, example)
        grouped.setdefault(example["source_split"], []).append(chat_record(example))
    for split, rows in grouped.items():
        path = OUTPUT_DIR / f"{split}.jsonl"
        temp = path.with_suffix(".jsonl.tmp")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        a.replace_with_retry(temp, path)
    a.save_json(OUTPUT_DIR / "review_metadata.json", [m for _, _, m in converted])
    (OUTPUT_DIR / "README.txt").write_text(
        "교훈 → 그림책 시나리오 SFT 샘플 2편\n\n"
        "개별 JSON: 관리용 ID·원작 ID·원래 분할 + input + output.\n"
        "input: moral, target_scene_count. output: title, characters, scenes.\n"
        "독자·언어·장면 단위는 train.jsonl의 공통 system 메시지에 지정했다.\n"
        "기준: 한국어, 만 5~7세 보호자 동반 읽기, 한 장면은 한 펼침면의 주요 시각적 순간.\n"
        "plan과 story는 장면 번호로 합쳤고, 그림 정보는 최종 visual_note를 사용했다.\n"
        "source_name은 name으로 바꾸었으며 본문·그림 지시·인물 설정은 다시 생성하지 않았다.\n\n"
        "train.jsonl: 한 줄이 한 작품이며, messages의 system/user/assistant 대화 형식이다.\n"
        "id/source_id/source_split은 관리 정보다. 모델에는 messages만 전달한다.\n"
        "원작 본문, 분석, 품질 점수, API 비용과 로그는 messages에 포함하지 않았다.\n"
        "두 원작의 기존 분할은 모두 train이다. 별도의 검증·테스트 샘플은 만들지 않았다.\n"
        "같은 원작의 증강본은 같은 분할에 묶는다.\n"
        "실제 학습 시 선택한 모델의 토크나이저·대화 템플릿을 적용하고 assistant 응답에 학습 손실을 계산하도록 설정해야 한다.\n"
        "모델별 학습 코드 연결, 토큰 길이 확인 및 실제 학습은 수행하지 않았다.\n\n"
        "품질 상태: 두 편 모두 human_review=pending, api_recommendation=revise인 초안이다.\n"
        "형식 연결·소규모 실험용 샘플이며 최종 품질 승인을 의미하지 않는다.\n"
        "남은 지적과 장면의 편집 의도는 review_metadata.json에 보존했다.\n"
        "원문 분석·창작 보완·API 이력은 review_metadata.json의 source_file에서 확인한다.\n"
        "원본 파일은 변경하지 않았고 추가 API 호출도 하지 않았다.\n\n"
        "재생성: python prepare_sft.py\n검사: python -m unittest -v test_prepare_sft.py\n",
        encoding="utf-8")
    for path, example, metadata in converted:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["source_sha256"]
        assert json.loads((OUTPUT_DIR / path.name).read_text(encoding="utf-8")) == example
    for split, rows in grouped.items():
        saved = [json.loads(line) for line in (OUTPUT_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert saved == rows
    print(json.dumps({"output": str(OUTPUT_DIR), "examples": len(converted),
                      "split_counts": {k: len(v) for k, v in grouped.items()},
                      "human_approved": sum(m["human_review"] == "approved" for _, _, m in converted),
                      "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
