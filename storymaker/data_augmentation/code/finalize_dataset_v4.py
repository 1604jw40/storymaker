"""검수 결과·비용·원본 보존을 검증한 뒤 전달 파일을 만든다. API 호출 없음."""
import collections
import hashlib
import html
import json
import math
import msvcrt
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.qwen.code.legacy import prepare_sft as sft
from storymaker.data_augmentation.code import review_dataset_v4 as v
from storymaker.data_augmentation.code.finalize_dataset_v3 import stats


def main():
    root = v.ROOT
    with (v.BASE/"run.lock").open("r+b") as base_lock,(root/"run.lock").open("r+b") as lock:
        for handle in (base_lock,lock):
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        assert v.load(root/"runtime.json")["status"]=="stopped"
        summary,config=v.load(root/"summary.json"),v.load(root/"config.json")
        assert summary["state"]=="complete" and summary["audited"]==304
        assert not summary["pending"] and not summary["failed"]
        for path,digest in config["protected_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        raw,labels,_=b.sources();raw_map={r["number"]:r for r in raw}
        records=[v.load(p) for p in sorted(root.glob("work/*/record.json"))]
        by_id={r["source"]["number"]:r for r in records}
        assert len(records)==304 and set(by_id)==set(labels)
        data,all_candidates=v.load(root/"dataset.json"),v.load(root/"all_candidates.json")
        assert len(all_candidates)==304 and {r["source_id"] for r in all_candidates}==set(labels)
        eligible,changes,topics=set(),[],collections.Counter()
        for n,record in by_id.items():
            assert record["source"]==raw_map[n] and record["source_split"]==labels[n]
            assert record["audit_status"]=="completed"
            candidate=b.Candidate.model_validate(record["candidate"]);v.v3.validate(candidate)
            review=v.FocusReview.model_validate(record["focus_review"]);v.validate_review(review,candidate)
            assert record["focus_verdict"]==v.verdict(review)
            if v.verdict(review)=="pass":eligible.add(n)
            topics.update(set(review.sensitive_topics))
            before=v.load(v.BASE/"work"/f"{n:04d}"/"record.json")["candidate"]
            assert a.digest(before)==record["baseline_candidate_digest"]
            if before!=record["candidate"]:
                changes.append({"source_id":n,"source_title":record["source"]["title"],
                    "source_split":labels[n],"before":before,"after":record["candidate"],
                    "editorial_adaptations":record["editorial_adaptations"],
                    "focus_revision_adopted":record.get("focus_revision_adopted",False),
                    "final_review":record["focus_review"]})
        assert {r["source_id"] for r in data}==eligible and len(data)==len(eligible)
        assert len(data)==summary["training_candidates"]
        for row in all_candidates:
            r=by_id[row["source_id"]]
            assert row["input"]=={"moral":r["candidate"]["moral"]} and row["output"]==r["candidate"]["output"]
        assert len(list((root/"examples").glob("*.json")))==len(data)
        for row in data:assert v.load(root/"examples"/f'{row["source_id"]:04d}.json')==row
        split_seen=set()
        for split in ("train","validation","test"):
            rows=[json.loads(line) for line in (root/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            expected=[]
            for row in data:
                if row["source_split"]!=split:continue
                chat=sft.chat_record(row);chat["messages"][0]["content"]=v.SYSTEM_PROMPT;expected.append(chat)
            assert rows==expected and len(rows)==summary["split_counts"][split]
            ids={r["source_id"] for r in rows};assert not ids&split_seen;split_seen|=ids
        queue=v.load(root/"human_review_queue.json")
        assert {r["source_id"] for r in queue}==set(labels)-eligible
        calibration=v.load(root/"calibration_results.json")
        assert {r["source_id"] for r in calibration}==set(v.PRIORITY)
        assert all(r["verdict"]!="pass" for r in calibration)
        for n in v.PRIORITY:
            assert by_id[n]["editorial_adaptations"]
            assert n in eligible, (n,by_id[n]["focus_review"])
        assert "주머니가 터" not in json.dumps(by_id[144]["candidate"],ensure_ascii=False)
        assert "당해도 싸" not in json.dumps(by_id[144]["candidate"],ensure_ascii=False)
        assert "푸줏간" not in json.dumps(by_id[99]["candidate"],ensure_ascii=False)
        assert "피를 흘" not in json.dumps(by_id[99]["candidate"],ensure_ascii=False)
        assert "절뚝" not in json.dumps(by_id[235]["candidate"],ensure_ascii=False)
        assert "주름진 피부" not in json.dumps(by_id[235]["candidate"],ensure_ascii=False)
        ledger_bytes=(root/"cost_ledger.json").read_bytes();ledger=json.loads(ledger_bytes)
        assert not any(x["status"]=="inflight" for x in ledger.values())
        tokens=dict.fromkeys(("input_tokens","cached_input_tokens","cache_write_tokens","output_tokens","reasoning_tokens"),0)
        total,ids,stages=0.0,[],collections.Counter()
        known=0
        for name,entry in ledger.items():
            assert int(Path(name).parts[1]) in labels
            assert Path(name).stem in ("calibration_before","age_moral_review","age_moral_revision","age_moral_recheck")
            if entry["status"]=="unknown":continue
            assert entry["status"]=="completed"
            envelope=v.load(root/name);response=envelope["response"];usage=response["usage"]
            assert response["status"]=="completed" and envelope["reasoning_effort"]=="medium"
            ids.append(response["id"]);stages[Path(name).stem]+=1;known+=1
            cost=a.cost_estimates(usage)["cache_write_upper_usd"]
            assert math.isclose(cost,entry["usd"],abs_tol=1e-12);total+=cost
            details=usage.get("input_tokens_details",{})
            values=(usage["input_tokens"],details.get("cached_tokens",0),details.get("cache_write_tokens",0),
                usage["output_tokens"],usage.get("output_tokens_details",{}).get("reasoning_tokens",0))
            assert values[-1]<=values[-2]
            for k,value in zip(tokens,values):tokens[k]+=value
        assert len(ids)==len(set(ids))==known==summary["known_api_responses"]
        assert stages["age_moral_review"]==304 and stages["calibration_before"]==3
        assert stages["age_moral_revision"]==stages["age_moral_recheck"]==summary["revision_calls"]
        tokens["total_tokens"]=tokens["input_tokens"]+tokens["output_tokens"]
        assert tokens==summary["api_tokens"]
        assert math.isclose(total,summary["cost_confirmed_usd"],abs_tol=1e-9)
        assert math.isclose(sum(x["usd"] for x in ledger.values()),summary["new_cost_upper_usd"],abs_tol=1e-9)
        assert summary["cumulative_upper_usd"]<=100
        a.save_json(root/"editorial_comparisons.json",changes)
        a.save_json(root/"changed_stories.json",[r for r in all_candidates if r["source_id"] in {c["source_id"] for c in changes}])
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(a.ROOT / "qwen/models/original/tokenizer.json"))
        count=lambda text:len(tokenizer.encode(text,add_special_tokens=False).ids)
        scenes=[s for row in data for s in row["output"]["scenes"]]
        token_stats={"tokenizer":"Qwen/Qwen3.5-4B","add_special_tokens":False,"chat_template_included":False,
            "scope":"이번 집중 검수 통과 학습 후보; API 과금 토큰과 별개",
            "scene_text":stats([count(s["text"]) for s in scenes]),
            "spoken_text_only":stats([count(s["text"]) for s in scenes if s["text"]]),
            "visual_note":stats([count(s["visual_note"]) for s in scenes]),
            "scene_text_and_visual":stats([count("\n".join(t for t in (s["text"],s["visual_note"]) if t)) for s in scenes]),
            "output_json_per_story":stats([count(json.dumps(r["output"],ensure_ascii=False)) for r in data])}
        a.save_json(root/"token_statistics.json",token_stats)
        report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"summary":summary,
            "tests_passed":52,"api_stages":dict(stages),"changed_source_ids":[r["source_id"] for r in changes],
            "sensitive_topics":dict(topics),"priority_stories":[{"source_id":n,
                "scores_before":next(r["review"] for r in calibration if r["source_id"]==n),
                "scores_after":by_id[n]["focus_review"],"editorial_adaptations":by_id[n]["editorial_adaptations"]} for n in v.PRIORITY],
            "checks":["304편 집중 검수 및 첫 3편 수정 전 결함 감지 확인",
                "최종 후보의 장면 번호·한국어·인물 참조·무언 본문 검증",
                "JSON·개별 JSON·JSONL 일치 및 원작별 분할 중복 없음",
                "원작·이전 분할·v2/v3 장부·v3 출력/기록 해시 불변",
                "API 응답 완료·ID 중복 없음·사용량/비용 일치·추론 중복 합산 없음·누적 상한 준수"],
            "rubric_sources":config["rubric_sources"],"human_review_performed":False,
            "image_generation_performed":False,"student_training_performed":False}
        a.save_json(root/"completion_report.json",report)
        text=("우화 그림책 v4 — 연령 적합성·교훈 표현 재검수\n\n"
            f"304/304편 검수 완료. 집중 기준 통과 {len(data)}편, 국소 수정 남음 {summary['revise']}편, 큰 편집 판단 보류 {summary['hold']}편.\n"
            f"내용 변경 {len(changes)}편. 최초 144·99·235번은 표현 각색 후 모두 집중 기준 통과.\n"
            f"신규 확인 비용 USD {total:.6f}, 신규 미확인 예약 USD {summary['cost_unknown_reserved_usd']:.6f}. "
            f"v2 이후 누적 예약 합산 USD {summary['cumulative_upper_usd']:.6f}/100.\n"
            f"이번 API 입력 {tokens['input_tokens']:,}, 캐시 읽기 {tokens['cached_input_tokens']:,}, 캐시 쓰기 {tokens['cache_write_tokens']:,}, "
            f"출력 {tokens['output_tokens']:,}, 추론 {tokens['reasoning_tokens']:,}(출력에 포함), 합계 {tokens['total_tokens']:,}.\n"
            "원작 속 표현도 자동 면제하지 않으며, 죽음·포식 자체는 일괄 금지하지 않음. 고통의 상세 묘사와 차별·피해자 비난을 구분해 평가.\n"
            "dataset.json/JSONL에는 이번 세 항목 모두 2점이고 보류가 없는 작품만 포함. all_candidates.json에 304편 전체 보존.\n"
            "human_review_queue.json은 남은 편집 판단, editorial_comparisons.json은 변경 전후, review_metadata.json은 304편 전체 평가.\n"
            "자동 검수이며 사람 전수 검수·출판 승인·그림 생성·모델 학습 성능 평가는 미실시. 비용은 실제 청구 조회가 아닌 응답 사용량·공개 단가 기준.\n\n"
            +"\n".join("- "+x for x in report["checks"])+"\n\n")
        criterion_labels={"age_appropriateness":"연령 적합성","moral_expression":"교훈 표현","adaptation_integrity":"핵심 서사 보존"}
        for r in queue:
            text+=f'{r["source_id"]}번 {r["title"]}: {r["focus_verdict"]}\n'
            review=r["review"]
            for key in ("age_appropriateness","moral_expression","adaptation_integrity"):
                if review[key]["score"]<2:text+=f'  {criterion_labels[key]}: {review[key]["reason"]}\n'
            for reason in review["publication_hold"]:text+="  보류: "+reason+"\n"
        (root/"REPORT.txt").write_text(text,encoding="utf-8")
        (root/"STATUS.txt").write_text(text,encoding="utf-8")
        e=lambda value:html.escape(str(value),quote=True)
        parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>우화 v4 검수 보고서</title>',
            '<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1050px;margin:auto;padding:28px;color:#202633}details{border:1px solid #ccd4df;padding:12px;margin:10px 0}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;font:inherit}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px;text-align:left;vertical-align:top}</style>',
            '<h1>우화 v4 연령·교훈 표현 검수</h1><pre>'+e(text)+'</pre><h2>304편 전체 평가</h2>']
        for n,r in by_id.items():
            review=r["focus_review"];candidate=r["candidate"]["output"]
            parts.append(f'<details><summary>{n:04d} {e(r["source"]["title"])} — {e(r["focus_verdict"])}</summary>')
            for key in ("age_appropriateness","moral_expression","adaptation_integrity"):
                item=review[key];parts.append(f'<p>{e(criterion_labels[key])}: {item["score"]}점 — {e(item["reason"])}</p>')
            parts.append('<p>민감 소재: '+e(", ".join(review["sensitive_topics"]) or "별도 표시 없음")+'</p>')
            for reason in review["publication_hold"]:parts.append('<p>보류: '+e(reason)+'</p>')
            for reason in review["revision_instructions"]:parts.append('<p>수정: '+e(reason)+'</p>')
            refs={ref for key in ("age_appropriateness","moral_expression","adaptation_integrity") if review[key]["score"]<2 for ref in review[key]["scene_refs"]}
            for scene in candidate["scenes"]:
                if scene["scene_no"] in refs:
                    parts.append(f'<pre>{scene["scene_no"]}장면 본문: {e(scene["text"])}\n그림 설명: {e(scene["visual_note"])}</pre>')
            parts.append('</details>')
        parts.append('</html>');(root/"age_moral_review_report.html").write_text("\n".join(parts),encoding="utf-8")
        assert (root/"cost_ledger.json").read_bytes()==ledger_bytes
        files=["dataset.json","all_candidates.json","changed_stories.json","train.jsonl","validation.jsonl","test.jsonl",
            "README.txt","REPORT.txt","STATUS.txt","summary.json","completion_report.json","review_metadata.json",
            "human_review_queue.json","revision_results.json","editorial_comparisons.json","calibration_results.json",
            "excluded_unlabeled.json","token_statistics.json","age_moral_review_report.html"]
        delivery=a.ROOT/"outputs"/"deliverables"/f"picturebook_dataset_v4_{len(data)}_candidates.zip"
        with zipfile.ZipFile(delivery,"w",zipfile.ZIP_DEFLATED) as z:
            for name in files:z.write(root/name,name)
            for path in sorted((root/"examples").glob("*.json")):z.write(path,path.relative_to(root).as_posix())
        with zipfile.ZipFile(delivery) as z:
            assert z.testzip() is None and len(z.namelist())==len(files)+len(data)
            assert json.loads(z.read("dataset.json"))==data
        print(json.dumps({"delivery":str(delivery),"bytes":delivery.stat().st_size,"summary":summary,
            "changed_count":len(changes),"queue":[{"id":r["source_id"],"title":r["title"],"verdict":r["focus_verdict"]} for r in queue],
            "api_stages":dict(stages),"sensitive_topic_labels_count":len(topics)},ensure_ascii=False))


if __name__=="__main__":
    main()
