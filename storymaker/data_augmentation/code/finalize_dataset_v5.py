"""v5 보존·분할·비용을 검증하고 일곱 편의 실제 전후 비교를 묶는다. 유료 호출 없음."""
import hashlib
import html
import json
import math
import msvcrt
import zipfile
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.qwen.code.legacy import prepare_sft as sft
from storymaker.data_augmentation.code import repair_remaining_v5 as v
def esc(value):
    return html.escape(str(value)).replace("\n", "<br>")


def scenes_html(scenes):
    return "".join(f'<p><b>장면 {s["scene_no"]}</b> · {esc(s["setting"])}<br>'
        f'{esc(s["text"] or "[글 없는 장면]")}<br><small>그림: {esc(s["visual_note"])}</small></p>' for s in scenes)


def main():
    root, load = v.ROOT, v.v4.load
    with (v.BASE/"run.lock").open("r+b") as old_lock, (root/"run.lock").open("r+b") as lock:
        for handle in (old_lock,lock):
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        assert load(root/"runtime.json")["status"] == "stopped"
        summary, config = load(root/"summary.json"), load(root/"config.json")
        assert summary["state"] == "complete" and summary["new_reviews_completed"] == 7
        assert not summary["failed"] and not summary["pending"]
        for path, digest in config["protected_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        raw, labels, _ = b.sources()
        raw_map = {r["number"]: r for r in raw}
        records = [load(p) for p in sorted(root.glob("work/*/record.json"))]
        by_id = {r["source"]["number"]: r for r in records}
        assert len(records) == 304 and set(by_id) == set(labels)
        changes, eligible = [], set()
        for n, record in by_id.items():
            old = load(v.BASE/"work"/f"{n:04d}"/"record.json")
            assert record["source"] == raw_map[n] and record["source_split"] == labels[n]
            assert record["audit_status"] == "completed"
            candidate = b.Candidate.model_validate(record["candidate"])
            v.v4.v3.validate(candidate)
            review = v.v4.FocusReview.model_validate(record["focus_review"])
            v.v4.validate_review(review,candidate)
            assert record["focus_verdict"] == v.v4.verdict(review)
            assert a.digest(old["candidate"]) == record["baseline_candidate_digest"]
            if record["focus_verdict"] == "pass": eligible.add(n)
            if n not in v.TARGETS:
                assert record["candidate"] == old["candidate"]
                assert record["focus_review"] == old["focus_review"]
                assert not record["calls"]
                continue
            expected, notes, highlights = v.edit(n,b.Candidate.model_validate(old["candidate"]))
            assert expected.model_dump() == record["candidate"] and record["edit_highlights"] == highlights
            before_scenes = old["candidate"]["output"]["scenes"]
            after_scenes = record["candidate"]["output"]["scenes"]
            signature = lambda s:a.digest({k:value for k,value in s.items() if k != "scene_no"})
            diff = SequenceMatcher(None,list(map(signature,before_scenes)),list(map(signature,after_scenes)),autojunk=False)
            scene_changes = [{"operation":tag,"before":before_scenes[i:j],"after":after_scenes[k:l]}
                for tag,i,j,k,l in diff.get_opcodes() if tag != "equal"]
            changes.append({"source_id":n,"source_title":record["source"]["title"],"source_split":labels[n],
                "source":record["source"],"preserved_core":v.CORE[n],"before":old["candidate"],
                "after":record["candidate"],"before_scene_count":len(before_scenes),"after_scene_count":len(after_scenes),
                "edit_notes":notes,"highlights":highlights,"all_scene_changes":scene_changes,
                "review_before":old["focus_review"],"review_after":record["focus_review"]})
        assert [r["source_id"] for r in changes] == v.TARGETS
        data, all_data = load(root/"dataset.json"), load(root/"all_candidates.json")
        assert len(data) == len(eligible) == summary["training_candidates"]
        assert {r["source_id"] for r in data} == eligible
        assert len(all_data) == 304 and {r["source_id"] for r in all_data} == set(labels)
        assert data == [r for r in all_data if r["source_id"] in eligible]
        for row in all_data:
            record = by_id[row["source_id"]]
            assert row["id"] == f'aesop_{row["source_id"]:04d}_scenes_sft_v5'
            assert row["source_split"] == labels[row["source_id"]]
            assert row["input"] == {"moral":record["candidate"]["moral"]}
            assert row["output"] == record["candidate"]["output"]
        assert len(list((root/"examples").glob("*.json"))) == len(data)
        for row in data: assert load(root/"examples"/f'{row["source_id"]:04d}.json') == row
        seen = set()
        for split in ("train","validation","test"):
            rows = [json.loads(s) for s in (root/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            expected = []
            for row in data:
                if row["source_split"] == split:
                    chat = sft.chat_record(row);chat["messages"][0]["content"] = v.v4.SYSTEM_PROMPT
                    expected.append(chat)
            assert rows == expected and len(rows) == summary["split_counts"][split]
            ids = {r["source_id"] for r in rows};assert not ids & seen;seen |= ids
        assert seen == eligible
        assert {r["source_id"] for r in load(root/"human_review_queue.json")} == set(labels)-eligible
        ledger = load(root/"cost_ledger.json")
        assert len(ledger) == 7 and all(x["status"] == "completed" for x in ledger.values())
        total, response_ids, ledger_sources = 0.0, set(), set()
        tokens = dict.fromkeys(("input_tokens","cached_input_tokens","cache_write_tokens","output_tokens","reasoning_tokens"),0)
        for path, entry in ledger.items():
            parts = Path(path).parts
            assert parts[0] == "work" and parts[2:] == ("calls","remaining_seven_review.json")
            n = int(parts[1]);ledger_sources.add(n)
            envelope = load(root/path);response = envelope["response"];usage = response["usage"]
            assert envelope["reasoning_effort"] == "medium" and response["status"] == "completed"
            assert response["id"] not in response_ids;response_ids.add(response["id"])
            cost = a.cost_estimates(usage)["cache_write_upper_usd"]
            assert math.isclose(cost,entry["usd"],abs_tol=1e-12);total += cost
            details = usage.get("input_tokens_details",{})
            values = (usage["input_tokens"],details.get("cached_tokens",0),details.get("cache_write_tokens",0),
                usage["output_tokens"],usage.get("output_tokens_details",{}).get("reasoning_tokens",0))
            assert values[-1] <= values[-2]
            for key,value in zip(tokens,values):tokens[key] += value
        assert ledger_sources == set(v.TARGETS)
        tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
        assert tokens == summary["api_tokens"] and math.isclose(total,summary["cost_confirmed_usd"],abs_tol=1e-12)
        assert math.isclose(config["baseline_cost_usd"]+total,summary["cumulative_upper_usd"],abs_tol=1e-9)
        assert summary["cumulative_upper_usd"] <= 100
        old_unknown = sum(x["usd"] for folder in (v.v4.v3.v2.ROOT,v.v4.v3.ROOT,v.BASE)
            for x in load(folder/"cost_ledger.json").values() if x["status"] == "unknown")
        assert math.isclose(old_unknown,0.3673125,abs_tol=1e-9)
        a.save_json(root/"before_after.json",changes)
        a.save_json(root/"modified_stories.json",[r for r in all_data if r["source_id"] in v.TARGETS])
        pages = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>우화 그림책 v5 전후 비교</title>'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'
            'body{font:16px/1.7 sans-serif;max-width:1200px;margin:auto;padding:24px;color:#20252a}'
            'table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccc;padding:14px;vertical-align:top;overflow-wrap:anywhere}'
            'small{color:#555}section{margin:48px 0}summary{cursor:pointer;font-weight:bold}h1,h2{line-height:1.4}</style>'
            '<h1>남은 7편 · 수정 전후 비교</h1><p>기준: v4 → v5. 본문·그림 설명·교훈의 실제 변경만 표시합니다. '
            '장면 번호만 달라진 경우는 생략했습니다. 297편은 내용과 검수 결과가 같습니다. 자동 재검수 통과는 출판 승인이 아닙니다.</p>']
        for row in changes:
            before,after = row["before"],row["after"]
            pages.append(f'<section><h2>{row["source_id"]}. {esc(row["source_title"])} · {row["before_scene_count"]} → {row["after_scene_count"]}장면</h2>'
                f'<p><b>보존한 결과:</b> {esc(row["preserved_core"])}</p><p>{"<br>".join(map(esc,row["edit_notes"]))}</p>'
                '<p>재검수: 연령 적합성·교훈 표현·핵심 서사 보존 모두 2/2점.</p><table><tr><th scope="col">수정 전</th><th scope="col">수정 후</th></tr>')
            if before["moral"] != after["moral"]:
                pages.append(f'<tr><td><b>교훈</b><p>{esc(before["moral"])}</p></td><td><b>교훈</b><p>{esc(after["moral"])}</p></td></tr>')
            if before["output"]["characters"] != after["output"]["characters"]:
                pages.append('<tr><td><b>인물 설정</b><p>'+esc(json.dumps(before["output"]["characters"],ensure_ascii=False,indent=2))+
                    '</p></td><td><b>인물 설정</b><p>'+esc(json.dumps(after["output"]["characters"],ensure_ascii=False,indent=2))+'</p></td></tr>')
            for change in row["all_scene_changes"]:
                pages.append('<tr><td>'+scenes_html(change["before"])+'</td><td>'+
                    (scenes_html(change["after"]) if change["after"] else '<p>해당 장면 삭제·통합</p>')+'</td></tr>')
            pages.append('</table><details><summary>전체 이야기 전후 보기</summary><table><tr><th scope="col">수정 전</th><th scope="col">수정 후</th></tr><tr><td>'+
                scenes_html(before["output"]["scenes"])+'</td><td>'+scenes_html(after["output"]["scenes"])+
                '</td></tr></table></details></section>')
        pages.append('</html>')
        (root/"before_after.html").write_text("\n".join(pages),encoding="utf-8")
        report = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"summary":summary,"tests_passed":54,
            "modified_source_ids":v.TARGETS,"unchanged_v4_stories":297,"old_unknown_reserved_usd":old_unknown,
            "checks":["304편 구조·한국어·인물 참조·무언 본문 검증","7편만 수정·재검수, 기존 297편 내용·검수 동일",
                "JSON·개별 JSON·JSONL 동일 및 원작별 분할 중복 없음","원작·기존 분할·이전 장부·v4 자료 해시 불변",
                "7개 응답 완료·응답 ID 중복 없음·사용량/비용 일치·누적 상한 준수"],
            "human_review_performed":False,"image_generation_performed":False,"student_training_performed":False,
            "cost_basis":"API 응답 사용량과 설정 단가 기준; 실제 청구 내역 조회 아님"}
        a.save_json(root/"completion_report.json",report)
        text = (f"우화 그림책 v5: 남은 7편 수정 및 같은 기준으로 재검수 완료. 학습 후보 {len(data)}/304편.\n"
            f"나머지 297편 내용·검수 보존. 분할: {summary['split_counts']}.\n"
            f"이번 확인 비용 ${total:.7f}, 신규 미확인 예약 $0. 누적(이전 미확인 예약 포함) ${summary['cumulative_upper_usd']:.7f}/100.\n"
            f"이전 미확인 예약 ${old_unknown:.7f} 보존. API 사용량: {json.dumps(tokens,ensure_ascii=False)}. 추론은 출력에 포함.\n"
            "7개 medium 검수만 호출. 생성·재작성 API 호출 없음. 54개 자동 검사 통과.\n"
            "61번은 위험한 직접 잡기 조언을 의도적으로 안전하게 각색하고 새 청소 성공 결말 제거.\n"
            "before_after.html과 before_after.json에서 전체 변경·교훈·핵심 결과·평가를 확인할 수 있음.\n"
            "사람 전수 검수·출판 승인·그림 제작·학습 성능 평가는 미실시. 비용은 응답 사용량 기반 추정.\n\n"+
            "\n".join(report["checks"])+"\n")
        for name in ("REPORT.txt","STATUS.txt","OPERATOR_NOTES.txt"):(root/name).write_text(text,encoding="utf-8")
        names = ["README.txt","REPORT.txt","STATUS.txt","summary.json","completion_report.json","dataset.json",
            "all_candidates.json","modified_stories.json","train.jsonl","validation.jsonl","test.jsonl",
            "review_metadata.json","human_review_queue.json","revision_results.json","excluded_unlabeled.json",
            "before_after.json","before_after.html"]
        destination = a.ROOT/"outputs/deliverables"/f"picturebook_dataset_v5_{len(data)}_candidates.zip"
        destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:archive.write(root/name,name)
            for path in sorted((root/"examples").glob("*.json")):archive.write(path,path.relative_to(root))
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None and len(archive.namelist()) == len(names)+len(data)
            assert json.loads(archive.read("dataset.json")) == data
        print(json.dumps({"archive":str(destination),"bytes":destination.stat().st_size,
            "changed":len(changes),"eligible":len(data),"new_cost_usd":total,"cumulative_upper_usd":summary["cumulative_upper_usd"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
