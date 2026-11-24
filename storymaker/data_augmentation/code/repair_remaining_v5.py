"""v4의 남은 7편만 직접 편집하고 검수 API로 확인한다. 이전 버전은 보존한다."""
import hashlib
import json
import msvcrt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import review_dataset_v4 as v4
BASE = v4.ROOT
ROOT = a.ROOT / "outputs" / "picturebook_dataset_v5"
TARGETS = [52, 61, 67, 87, 153, 245, 290]
CORE = {
    52: "돌을 던지는 놀이로 개구리들이 죽고, 살아남은 개구리가 놀이와 죽음의 차이를 항의함",
    61: "쐐기풀에 쏘인 소년이 엄마에게 호소하고, 엄마의 조언으로 마무리됨; 새 청소 성공 결말 없음",
    67: "늙은 군마의 방앗간 배치·한탄과 형편이 달라질 수 있다는 주인의 응답을 유지함",
    87: "발을 잇는 줄 때문에 생쥐가 익사하고, 매가 생쥐와 묶인 개구리를 함께 잡아먹음",
    153: "편애받던 새끼는 지나친 포옹으로 죽고, 돌봄받지 못한 새끼는 살아 자라남",
    245: "산토끼 포획·참새의 조롱·참새의 포획과 죽어 가는 산토끼의 마지막 말을 유지함",
    290: "벼룩의 해를 축소하는 변명과 사람의 거절을 유지하며, 기존 각색의 비노골적인 치명적 마무리를 보존함",
}
README = """우화 그림책 v5 — 잔여 7편 편집·재검수

v4의 52·61·67·87·153·245·290번만 수정했다. 나머지 297편의 교훈·출력·검수는 v4와 동일하다.
비용 절약을 위해 에이전트가 반복 구간·표현을 직접 편집하고 새 검수 API만 호출한다. 생성 API는 호출하지 않는다.
연령 적합성·교훈 표현·핵심 서사 보존의 v4 검수 기준을 그대로 사용하며 점수를 낮춰 통과시키지 않는다.
61번은 원문의 직접적인 쐐기풀 잡기 지시를 안전한 방법과 노력에 대한 조언으로 각색했다.
쐐기풀에 쏘인 결과와 엄마에게 호소해 조언을 듣는 흐름은 유지하고, 이전에 덧붙인 청소 성공 결말은 제거했다.
각색의 범위와 원작 핵심 결과는 before_after.json 및 record의 editorial_adaptations에 기록한다.
all_candidates.json에는 304편 전체, dataset.json·JSONL에는 집중 검수 통과 후보만 포함한다.
before_after.html은 7편의 전후 비교이며 전체 본문·그림 설명도 펼쳐 볼 수 있다.
장면 수는 고정하지 않는다. 입력은 moral, 출력은 title·characters·scenes 구조다.
기존 원작별 분할과 이전 버전, 이전 과금 미확인 예약을 보존했다. v2 이후 누적 100달러 상한을 유지한다.
자동 검수는 사람 편집·출판 승인·그림 품질·실제 학습 성능의 검증과 다르다.
"""


def edit(number, candidate):
    result = candidate.model_copy(deep=True)
    original = {s.scene_no: s.model_copy(deep=True) for s in candidate.output.scenes}
    changed = {n: s.model_copy(deep=True) for n, s in original.items()}
    notes = []
    highlights = []
    keep = list(original)
    if number == 52:
        keep = [1, 2, 4, 5, 8, 10, 11, 13, 14]
        changed[8].visual_note = "비어 있는 연잎 곁에서 살아남은 개구리가 물 위로 고개를 내밀어요. 상처나 죽은 몸은 그리지 않아요."
        changed[13].text = "“너희에게는 놀이지만, 우리에게는 죽음이란다.” 이번에는 소년들의 웃음소리가 들리지 않았어요."
        changed[13].visual_note = "소년들이 입을 다문 채 말하는 개구리를 바라봐요. 뒤쪽 물풀과 연잎은 조용해요."
        notes = ["추가 투척·숨은 개구리 찾기와 사체 재확인을 덜고, 개구리들의 죽음과 중단 요청·핵심 대사는 유지"]
        highlights = [(7, None, "추가 공격 준비 삭제"), (9, None, "다시 겨누는 공격 삭제"), (13, 13, "항의 뒤 죽은 친구들의 재묘사 삭제")]
    elif number == 61:
        rows = [
            (1, ["boy"], "집 곁 담장 아래, 오전",
             "소년이 담장 아래 풀을 들여다보았어요. 잎에 난 가는 털이 궁금했지요. ‘살짝만 만져 볼까?’ 손끝이 잎을 스치자 따끔했어요. 소년은 놀라 손을 홱 거두었어요.",
             "소년이 담장 아래 쐐기풀에서 손을 급히 떼고 있어요. 풀을 잡거나 다시 만지는 모습은 없어요."),
            (2, ["boy", "mother"], "집 문 앞, 잠시 뒤",
             "손이 따끔거려 소년은 집으로 달려갔어요.\n“엄마, 풀을 살짝 건드렸을 뿐인데 무척 아파요.”\n엄마는 하던 일을 멈추고 소년의 손을 살펴보았어요.",
             "소년이 아픈 손을 내밀고, 엄마는 몸을 낮추어 소년의 말을 듣고 있어요."),
            (3, ["boy", "mother"], "집 안 창가, 오전",
             "“어디에 있던 풀이니?”\n소년이 담장을 가리키자 엄마가 말했어요.\n“쐐기풀이구나. 살짝 닿아도 따끔할 수 있어. 모르는 풀은 맨손으로 만지지 말고 먼저 어른에게 물어보렴.”",
             "엄마와 소년이 창밖 담장 쪽을 바라봐요. 소년은 아픈 손을 몸 가까이에 두고 있어요."),
            (4, ["boy", "mother"], "집 안, 같은 오전",
             "“그러면 무슨 일이든 살살 해야 해요?” 소년이 물었어요.\n“안전하게 하는 방법을 먼저 알고, 하기로 한 일에는 힘을 다해야지. 이를테면 빗자루질은 잎을 스치기만 하지 말고 바닥을 따라 밀어야 한단다.”\n엄마는 소년 곁에서 차분히 이야기했어요.",
             "엄마가 소년과 눈을 맞추며 이야기해요. 소년은 아픈 손을 쉬게 하며 듣고 있어요.")
        ]
        result.output.scenes = [b.Scene(scene_no=n, character_ids=ids, setting=setting,
            text_mode="read_aloud", text=text, visual_note=visual) for n, ids, setting, text, visual in rows]
        result.moral = "안전하게 하는 방법을 먼저 알고, 하기로 한 일에는 힘을 다해야 해요."
        result.output.characters[0].goal = "따끔해진 손을 엄마에게 보이고, 어떻게 해야 하는지 알고 싶어요."
        result.output.characters[1].goal = "아이의 말을 듣고 안전한 방법과 일에 힘을 다하는 태도를 알려 주고 싶어요."
        notes = ["소년의 쐐기풀 접촉·통증·엄마에게 호소·엄마의 조언이라는 원작의 사건과 마무리 구조 보존",
            "원문의 '다음에는 힘주어 꽉 잡아 보렴'은 맨손 접촉 지시가 되므로 의도적으로 안전한 방법·노력에 관한 대사로 각색",
            "빗자루는 엄마가 드는 짧은 비유일 뿐 실제 청소 과제·재도전·성공·통증 회복 결말은 추가하지 않음"]
        highlights = [(5, 3, "직접 잡기 지시를 안전한 조언으로 각색"), (7, None, "새 청소 성공 결말 삭제")]
        keep = None
    elif number == 67:
        keep = list(range(1, 18))
        result.moral = "형편이 달라졌을 때 지난날과 비교만 하면 지금의 어려움이 더 크게 느껴질 수 있어요."
        notes = ["교훈을 낮아진 처지에 순응하라는 명령에서 과거와의 비교에만 매이는 태도에 대한 표현으로 좁힘",
            "원작에 없는 작업 재개·태도 변화의 완결을 덜고 주인의 조언을 들은 뒤 열린 반응으로 마무리"]
        highlights = [(19, None, "작업 재개와 성장의 완결 삭제")]
    elif number == 87:
        keep = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16, 18]
        changed[14].visual_note = "매가 날개를 기울여 수면을 향해 내려와요. 생쥐의 발에서 이어진 덩굴 끝에 개구리가 있어요."
        notes = ["거듭되는 중단 요청·매듭 당기기·허공에서 버둥거리는 묘사를 삭제",
            "익사와 매의 포식, 두 발을 이은 줄 때문에 개구리도 끌려가는 인과는 명확하고 비노골적으로 유지"]
        highlights = [(15, None, "반복되는 매듭 탈출 시도 삭제"), (17, None, "공중 몸부림과 위기 재설명 삭제")]
    elif number == 153:
        keep = [1, 4, 5, 6, 8, 9, 10, 11, 15, 16]
        changed[8].text = "나루는 굵은 가지에 배를 붙이고 손을 뻗었어요. 열매 하나가 톡 떨어졌지요. 두 손으로 받아 작게 베어 물었어요."
        changed[10].text = "바람이 불자 나루는 잎이 무성한 곳으로 들어갔어요. 넓은 잎이 바람을 가려 주었지요. 나루는 그 안에 몸을 웅크렸어요."
        notes = ["편애는 처음 한 번과 보리의 움직임 제한으로 전달하고, 나루의 거절·돌봄 부재 회상을 반복하지 않음",
            "보리의 치명적 결과와 나루의 성장은 유지하며 방임 자체를 좋은 양육으로 칭찬하지 않음"]
        highlights = [(3, None, "추가 거절 경험 삭제"), (10, 10, "거절 회상 대신 쉴 곳을 찾는 행동"), (13, None, "돌봄 부재 재강조 삭제")]
    elif number == 245:
        keep = [1, 2, 3, 4, 5, 7, 9, 12, 14, 15, 16]
        changed[12].text = "참새는 낮은 돌 위로 폴짝 올라가 여전히 산토끼 쪽만 바라보았어요. 풀밭 옆 마른 나무에 앉아 있던 매가 그 참새를 발견했어요. 매는 날개를 기울였어요."
        changed[12].visual_note = "마른 가지의 매가 낮은 돌 위 참새를 바라봐요. 참새의 고개는 산토끼 쪽으로 돌아가 있어요."
        changed[16].text = "죽어 가던 산토끼가 남은 힘을 모아 말했어요. “내가 붙잡혔다고 웃더니…… 너도 같은 일을 당했구나.” 산토끼는 더 말을 잇지 못했어요."
        notes = ["조롱은 핵심 대사 한 번만 남기고 애원 뒤의 추가 조롱과 추격 반복을 덜어냄",
            "참새가 돌로 옮겨 가는 연결을 남겨 위치를 일치시키고 두 동물의 포획·산토끼의 치명적 결과와 마지막 말 유지"]
        highlights = [(10, None, "추가 애원 삭제"), (11, None, "날개를 자랑하며 조롱하는 대사 삭제"), (13, None, "세 번째 조롱 삭제")]
    elif number == 290:
        changed[14].text = changed[14].text.replace(" 너를 죽이겠다.", "")
        notes = ["직접적인 살해 위협 대사만 삭제; 해를 축소하는 변명·거절과 짧고 비노골적인 치명적 마무리는 유지"]
        highlights = [(14, 14, "직접적인 살해 위협 삭제")]
    else:
        raise ValueError("승인되지 않은 수정 대상")
    if keep is not None:
        result.output.scenes = [changed[n] for n in keep]
    mapping = {old: index for index, old in enumerate(keep, 1)} if keep is not None else {}
    for index, scene in enumerate(result.output.scenes, 1):
        scene.scene_no = index
    v4.v3.validate(result)
    excerpts = []
    for old, after, reason in highlights:
        new_no = mapping[after] if after is not None and keep is not None else after
        excerpts.append({"before_scene_no": old, "after_scene_no": new_no,
            "before_text": original[old].text,
            "after_text": result.output.scenes[new_no-1].text if new_no else None, "reason": reason})
    assert result.model_dump() != candidate.model_dump()
    return result, notes, excerpts


def export(budget, baseline_cost):
    summary = v4.export(budget, baseline_cost, root=ROOT, dataset_version="v5")
    summary["baseline_v2_v3_v4_cost_usd"] = summary.pop("baseline_v2_v3_cost_usd")
    summary["new_review_targets"] = TARGETS
    summary["new_reviews_completed"] = sum(v4.load(ROOT/"work"/f"{n:04d}"/"record.json")["audit_status"] == "completed" for n in TARGETS)
    summary["inherited_v4_passes"] = 297
    a.save_json(ROOT/"summary.json", summary)
    (ROOT/"README.txt").write_text(README, encoding="utf-8")
    metadata = v4.load(ROOT/"review_metadata.json")
    for row in metadata:
        row["review_origin"] = "v5" if row["source_id"] in TARGETS else "v4"
    a.save_json(ROOT/"review_metadata.json", metadata)
    return summary


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw, labels, _ = b.sources()
    previous = {n: v4.load(BASE/"work"/f"{n:04d}"/"record.json") for n in labels}
    assert {n for n, r in previous.items() if r["focus_verdict"] != "pass"} == set(TARGETS)
    ledgers = [v4.v3.v2.ROOT/"cost_ledger.json", v4.v3.ROOT/"cost_ledger.json", BASE/"cost_ledger.json"]
    assert all(x["status"] in ("completed","unknown") for p in ledgers for x in v4.load(p).values())
    baseline_cost = sum(x["usd"] for p in ledgers for x in v4.load(p).values())
    assert abs(baseline_cost-v4.load(BASE/"summary.json")["cumulative_upper_usd"]) < 1e-8
    patches = {n: edit(n, b.Candidate.model_validate(previous[n]["candidate"])) for n in TARGETS}
    protected = [*BASE.glob("work/*/record.json"), *ledgers, a.SOURCE,
        *[a.SPLITS/f"{s}.jsonl" for s in ("train","validation","test")],
        *[BASE/name for name in ("dataset.json","all_candidates.json","train.jsonl","validation.jsonl","test.jsonl","review_metadata.json")]]
    config = {"version":"remaining-seven-v5","model":a.MODEL,"effort":"medium","targets":TARGETS,
        "baseline_cost_usd":baseline_cost,"cumulative_budget_usd":100,"review_prompt_hash":a.digest(v4.REVIEW_PROMPT),
        "edited_candidate_hashes":{str(n):a.digest(value[0].model_dump()) for n,value in patches.items()},
        "protected_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
        "safety_source_61":"https://www.rhs.org.uk/weeds/nettles-as-weeds"}
    if (ROOT/"config.json").exists() and v4.load(ROOT/"config.json") != config:
        raise ValueError("보존 대상·편집본·검수 기준이 이전 실행과 다릅니다.")
    a.save_json(ROOT/"config.json", config)
    for n, old in previous.items():
        path = ROOT/"work"/f"{n:04d}"/"record.json"
        if path.exists():
            continue
        record = {key:old[key] for key in ("source","source_split","reference_moral","candidate","focus_review","focus_verdict","editorial_adaptations")}
        record.update(baseline_candidate_digest=a.digest(old["candidate"]), audit_status="completed", calls={},
            human_review="pending", review_origin="v4")
        if n in TARGETS:
            revised, notes, excerpts = patches[n]
            record.update(candidate=revised.model_dump(), previous_focus_review=old["focus_review"],
                editorial_adaptations=old["editorial_adaptations"]+notes, edit_highlights=excerpts,
                preserved_core=CORE[n], audit_status="pending", review_origin="v5")
        a.save_json(path, record)
    a.save_json(ROOT/"excluded_unlabeled.json", v4.load(BASE/"excluded_unlabeled.json"))
    a.save_json(ROOT/"review_prompt.json", {"prompt":v4.REVIEW_PROMPT,"schema":v4.FocusReview.model_json_schema()})
    return baseline_cost


def check(number, budget, key):
    path = ROOT/"work"/f"{number:04d}"/"record.json"
    record = v4.load(path)
    candidate = b.Candidate.model_validate(record["candidate"])
    try:
        review, usage = budget.stage(path.parent/"calls/remaining_seven_review.json", v4.REVIEW_PROMPT,
            {"source":{k:record["source"][k] for k in ("title","story")},
             "candidate":candidate.model_dump(), "editorial_adaptations":record["editorial_adaptations"]},
            v4.FocusReview, 2400, key, effort="medium")
        v4.validate_review(review,candidate)
        record.update(audit_status="completed",focus_review=review.model_dump(),focus_verdict=v4.verdict(review))
        record["calls"]["remaining_seven_review"] = usage
        record.pop("audit_error",None)
    except Exception as exc:
        record.update(audit_status="failed",audit_error=f"{type(exc).__name__}: {exc}")
    a.save_json(path,record)
    return record


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (BASE/"run.lock").open("r+b") as base_lock,(ROOT/"run.lock").open("a+b") as lock:
        for handle in (base_lock,lock):
            handle.seek(0)
            if not handle.read(1):handle.write(b"0");handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        baseline_cost = prepare()
        budget = b.Budget(ROOT,100-baseline_cost)
        key = yaml.safe_load(a.KEY_FILE.read_text(encoding="utf-8-sig"))["openai_api_key"].strip()
        if not key.startswith("sk-"):raise ValueError("기존 API 키가 없습니다.")
        runtime={"pid":os.getpid(),"status":"running","started_at_utc":datetime.now(timezone.utc).isoformat()}
        a.save_json(ROOT/"runtime.json",runtime)
        try:
            jobs=[n for n in TARGETS if v4.load(ROOT/"work"/f"{n:04d}"/"record.json")["audit_status"]=="pending"]
            with ThreadPoolExecutor(max_workers=7) as pool:
                for future in as_completed([pool.submit(check,n,budget,key) for n in jobs]):
                    r=future.result()
                    print(f'{r["source"]["number"]}: {r["audit_status"]} / {r.get("focus_verdict")}',flush=True)
            summary=export(budget,baseline_cost)
            for p,digest in v4.load(ROOT/"config.json")["protected_sha256"].items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:
            runtime.update(status="stopped",finished_at_utc=datetime.now(timezone.utc).isoformat())
            a.save_json(ROOT/"runtime.json",runtime)


if __name__=="__main__":
    main()
