"""유료 호출 없이 v3의 보존·분할·품질·비용을 검증하고 전달 파일을 만든다."""
import hashlib
import json
import math
import msvcrt
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.qwen.code.legacy import prepare_sft as sft
from storymaker.data_augmentation.code import revise_dataset_v3 as v
from storymaker.data_augmentation.code.repair_dataset import scores


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    return {"count": len(values), "total": sum(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    root = v.ROOT
    with (v.BASE / "run.lock").open("r+b") as base_lock, (root / "run.lock").open("r+b") as lock:
        for handle in (base_lock, lock):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        assert read(root / "runtime.json")["status"] == "stopped"
        summary, config = read(root / "summary.json"), read(root / "config.json")
        assert summary["state"] == "complete" and summary["corrections_completed"] == 139
        assert not summary["corrections_failed"]
        for path, expected in config["protected_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        raw, labels, _ = b.sources()
        raw_map = {r["number"]: r for r in raw}
        records = [read(p) for p in sorted(root.glob("work/*/record.json"))]
        record_map = {r["source"]["number"]: r for r in records}
        assert len(records) == 304 and set(record_map) == set(labels)
        targets = set(config["target_source_ids"])
        assert len(targets) == 139
        changes, unchanged = [], 0
        for n, record in record_map.items():
            before = read(v.BASE / "work" / f"{n:04d}" / "record.json")
            assert record["source"] == raw_map[n] and record["source_split"] == labels[n]
            candidate, review = b.Candidate.model_validate(record["candidate"]), a.Review.model_validate(record["review"])
            v.validate(candidate)
            a.validate_review(review, candidate.output)
            assert b.recommendation(review) == record["api_recommendation"]
            if n not in targets:
                assert record["correction_status"] == "unchanged" and not record["calls"]
                assert record["candidate"] == before["candidate"] and record["review"] == before["review"]
                unchanged += 1
                continue
            assert record["correction_status"] == "completed"
            assert 1 <= len(record["attempts"]) <= 2
            assert all(name.startswith("flow_revision") or name in ("flow_local_review", "flow_local_review2") for name in record["calls"])
            changes.append({"source_id": n, "source_title": record["source"]["title"],
                "source_split": labels[n], "scene_count_before": len(before["candidate"]["output"]["scenes"]),
                "scene_count_after": len(candidate.output.scenes),
                "content_changed": before["candidate"] != record["candidate"],
                "selected_attempt": record["selected_attempt"],
                "scores_before": scores(before["review"]), "scores_after": scores(record["review"]),
                "moral_before": before["candidate"]["moral"], "moral_after": candidate.moral,
                "review_before": before["review"], "review_after": record["review"],
                "agent_local_edit": record.get("agent_local_edit")})
        assert unchanged == 165
        dataset, candidates = read(root / "dataset.json"), read(root / "all_candidates.json")
        assert len(candidates) == 304 and {r["source_id"] for r in candidates} == set(labels)
        major = {r["source"]["number"] for r in records if v.v2.needs_rewrite(a.Review.model_validate(r["review"]))}
        assert {r["source_id"] for r in dataset} == set(labels) - major
        assert len(dataset) == summary["dataset_examples"]
        a.save_json(root / "revised_stories.json", [row for row in candidates if row["source_id"] in targets])
        for row in candidates:
            n = row["source_id"]
            assert row["input"] == {"moral": record_map[n]["candidate"]["moral"]}
            assert row["output"] == record_map[n]["candidate"]["output"]
            assert row["source_split"] == labels[n]
        assert len(list((root / "examples").glob("*.json"))) == len(dataset)
        for row in dataset:
            assert read(root / "examples" / f'{row["source_id"]:04d}.json') == row
        split_ids = set()
        for split in ("train", "validation", "test"):
            rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            expected = []
            for example in dataset:
                if example["source_split"] == split:
                    row = sft.chat_record(example)
                    row["messages"][0]["content"] = v.SYSTEM_PROMPT
                    expected.append(row)
            assert rows == expected and len(rows) == summary["split_counts"][split]
            ids = {r["source_id"] for r in rows}
            assert not split_ids & ids
            split_ids |= ids
        ledger_bytes = (root / "cost_ledger.json").read_bytes()
        ledger = json.loads(ledger_bytes)
        assert not any(x["status"] == "inflight" for x in ledger.values())
        confirmed, known_calls = 0.0, 0
        tokens = dict.fromkeys(("input_tokens", "cached_input_tokens", "cache_write_tokens",
                              "output_tokens", "reasoning_tokens"), 0)
        for name, entry in ledger.items():
            assert int(Path(name).parts[1]) in targets
            assert Path(name).stem in {"flow_revision1", "flow_revision1_review", "flow_revision2", "flow_revision2_review", "flow_local_review", "flow_local_review2"}
            if entry["status"] != "completed":
                assert entry["status"] == "unknown"
                continue
            envelope = read(root / name)
            usage = envelope["response"]["usage"]
            assert envelope["reasoning_effort"] == "medium"
            cost = a.cost_estimates(usage)["cache_write_upper_usd"]
            assert math.isclose(entry["usd"], cost, abs_tol=1e-12)
            confirmed += cost
            known_calls += 1
            details = usage.get("input_tokens_details", {})
            values = (usage["input_tokens"], details.get("cached_tokens", 0), details.get("cache_write_tokens", 0),
                usage["output_tokens"], usage.get("output_tokens_details", {}).get("reasoning_tokens", 0))
            assert values[-1] <= values[-2]
            for k, value in zip(tokens, values):
                tokens[k] += value
        tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
        assert tokens == summary["api_tokens"]
        assert math.isclose(confirmed, summary["cost_confirmed_usd"], abs_tol=1e-9)
        assert math.isclose(sum(x["usd"] for x in ledger.values()), summary["new_cost_upper_usd"], abs_tol=1e-9)
        assert summary["cumulative_v2_v3_upper_usd"] <= 100
        assert known_calls == len(list(root.glob("work/*/calls/*.json")))
        tokenizer = Tokenizer.from_file(str(a.ROOT / "qwen/models/original/tokenizer.json"))
        def count(text):
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        scenes = [s for row in dataset for s in row["output"]["scenes"]]
        revised_scenes = [s for row in dataset if row["source_id"] in targets for s in row["output"]["scenes"]]
        token_statistics = {"tokenizer": "Qwen/Qwen3.5-4B", "add_special_tokens": False,
            "chat_template_included": False, "scope": "학습 후보 전체; API 과금 토큰과 별개",
            "scene_text": stats([count(s["text"]) for s in scenes]),
            "spoken_text_only": stats([count(s["text"]) for s in scenes if s["text"]]),
            "visual_note": stats([count(s["visual_note"]) for s in scenes]),
            "scene_text_and_visual": stats([count("\n".join(x for x in (s["text"],s["visual_note"]) if x)) for s in scenes]),
            "output_json_per_story": stats([count(json.dumps(row["output"], ensure_ascii=False)) for row in dataset]),
            "revised_139_scene_text_and_visual": stats([count("\n".join(x for x in (s["text"],s["visual_note"]) if x)) for s in revised_scenes]),
            "wordless_scene_count": sum(s["text_mode"] == "wordless" for s in scenes)}
        a.save_json(root / "token_statistics.json", token_statistics)
        a.save_json(root / "revision_results.json", sorted(changes, key=lambda row:row["source_id"]))
        revised_counts = [row["scene_count_after"] for row in changes]
        baseline_summary = read(v.BASE / "summary.json")
        report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary,
            "validation": ["304편 원작·분할·장면 번호·인물 참조·무언 본문 검증",
                "기존 통과 165편의 교훈·출력·검수 및 v2 보호 파일 해시 불변",
                "수정 대상 139편만 신규 API 호출, 각 최대 2회 자동 수정·검수; 추가 국소 편집은 별도 이력",
                "JSON·개별 JSON·JSONL 내용 일치, 분할 중복 없음, 학습 입력의 고정 장면 수 제거",
                "신규 응답 사용량·비용·캐시 대조, 추론 중복 합산 없음, 누적 100달러 상한 준수"],
            "tests_passed": 46, "revised_scene_count_statistics": stats(revised_counts),
            "content_changed_count": sum(row["content_changed"] for row in changes),
            "agent_local_edit_source_ids": [row["source_id"] for row in changes if row["agent_local_edit"]],
            "unchanged_pass_count": unchanged,
            "prior_v2_confirmed_usd": baseline_summary["cost_confirmed_usd"],
            "prior_v2_unknown_reserved_usd": baseline_summary["cost_unknown_reserved_usd"],
            "new_unknown_requests": [name for name,x in ledger.items() if x["status"]=="unknown"],
            "human_review_performed": False, "model_training_performed": False, "image_generation_performed": False}
        a.save_json(root / "completion_report.json", report)
        text = (
            "우화 그림책 v3 — 이야기 흐름에 맞춘 수정 결과\n\n"
            f"대상 139편 수정·재검수 종료. 기존 통과 165편의 교훈·출력 유지. 내용 변경 {report['content_changed_count']}편.\n"
            f"최종 학습 후보 {len(dataset)}편: 전 항목 충족 {summary['all_criteria_pass']}, 국소 첨삭 {summary['minor_revision']}. "
            f"큰 문제 제외 {summary['major_revision']}편.\n"
            f"수정 139편의 장면 수: 최소 {min(revised_counts)}, 중앙값 {statistics.median(revised_counts)}, 최대 {max(revised_counts)}. "
            "학습 input은 moral만 포함하며 장면 수는 output.scenes를 따른다.\n"
            f"이번 확인 비용 USD {confirmed:.6f}, 이번 미확인 예약 USD {summary['cost_unknown_reserved_usd']:.6f}. "
            f"v2 포함 누적 상한 집계 USD {summary['cumulative_v2_v3_upper_usd']:.6f}/100.\n"
            f"이번 API 입력 {tokens['input_tokens']:,}, 캐시 읽기 {tokens['cached_input_tokens']:,}, "
            f"캐시 쓰기 {tokens['cache_write_tokens']:,}, 출력 {tokens['output_tokens']:,}, "
            f"추론 {tokens['reasoning_tokens']:,}(출력에 포함), 합계 {tokens['total_tokens']:,}.\n"
            "이전 v2 과금 미확인 예약은 유지한다. 비용은 응답 사용량·단가 기준이며 실제 청구 조회가 아니다.\n"
            "남은 지적은 remaining_tasks.json/review_metadata.json, 작품별 전후 비교는 revision_results.json에 있다.\n"
            "사람 전수 검수·실제 모델 학습·그림 제작은 미실시.\n\n"
            + "\n".join("- "+item for item in report["validation"]) + "\n")
        (root / "REPORT.txt").write_text(text, encoding="utf-8")
        (root / "STATUS.txt").write_text(text, encoding="utf-8")
        assert (root / "cost_ledger.json").read_bytes() == ledger_bytes
        delivery = a.ROOT / "outputs" / "deliverables" / f"picturebook_dataset_v3_{len(dataset)}_candidates.zip"
        files = ["dataset.json","all_candidates.json","revised_stories.json","train.jsonl","validation.jsonl","test.jsonl",
            "README.txt","REPORT.txt","STATUS.txt","summary.json","completion_report.json","revision_results.json",
            "review_metadata.json","excluded_quality.json","excluded_unlabeled.json","remaining_tasks.json","token_statistics.json"]
        files += [name for name in ("quality_audit.json", "QUALITY_AUDIT.txt", "illustration_spec_example.json") if (root / name).exists()]
        with zipfile.ZipFile(delivery, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                z.write(root / name, name)
            for path in sorted((root / "examples").glob("*.json")):
                z.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(delivery) as z:
            assert z.testzip() is None and len(z.namelist()) == len(files) + len(dataset)
            assert json.loads(z.read("dataset.json")) == dataset
        print(json.dumps({"delivery":str(delivery),"bytes":delivery.stat().st_size,
            "summary":summary,"revised_scene_counts":report["revised_scene_count_statistics"],
            "content_changed_count":report["content_changed_count"],"token_statistics":token_statistics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
