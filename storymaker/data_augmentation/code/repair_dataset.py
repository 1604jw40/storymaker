"""전수 생성 후 남은 예산으로 문제 후보를 한 번 더 보완하고 전후 결과를 남긴다."""
import argparse
import hashlib
import json
import msvcrt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
POST_PROMPT = b.FIX_PROMPT + """
전수 생성 후의 추가 보완이다. review에서 지적한 문제를 우선 수정한다.
합치라는 권고가 있으면 반복 장면을 합치고, 원작의 동기·단서·선택·관계 변화가 드러나는 독립적인 행동으로 장면을 재배분한다.
20장면은 유지하되, 같은 표정의 확대나 마지막 교훈 대사를 나누어 수를 채우지 않는다.
원작의 핵심 결과를 바꾸는 새 사건은 넣지 않는다. 각 장면의 글·그림 지시는 필요한 만큼 간결하게 쓴다."""


def scores(review):
    return {k: v["score"] for k, v in review.items() if isinstance(v, dict) and "score" in v}


def quality_rank(review):
    values = scores(review)
    return (len(review["blocking_issues"]), sum(v == 0 for v in values.values()),
            -values["moral_and_fidelity"], -values["causality"], -values["age_and_read_aloud"],
            -sum(values.values()))


def priority(record):
    if record["status"] != "completed":
        return (-1, 0, 0, record["source"]["number"])
    review = record["review"]
    values = scores(review)
    return (0 if review["blocking_issues"] or 0 in values.values() else 1,
            -len(review["blocking_issues"]), -sum(v == 0 for v in values.values()),
            values["moral_and_fidelity"] + values["causality"] + values["age_and_read_aloud"],
            sum(values.values()), record["source"]["number"])


def group_problems(records):
    return {group: [record for record in records if priority(record)[0] == group] for group in (-1, 0, 1)}


def repair(record, lesson, root, budget, key, *, stage_prefix="post_repair", prompt=None):
    number = record["source"]["number"]
    work = root / "work" / f"{number:04d}"
    archived = work / "before_post_repair.json"
    if not archived.exists():
        a.save_json(archived, record)
    before = record.get("review") or record.get("initial_review")
    result = {"source_id": number, "title": record["source"]["title"], "source_split": record["source_split"],
              "before_status": record["status"], "before_scores": scores(before) if before else None,
              "before_blocking_issues": before["blocking_issues"] if before else None}
    payload = {"source": {k: record["source"][k] for k in ("title", "story")},
               "reference_moral": lesson, "conditions": b.CONDITIONS}
    if "candidate" in record:
        payload["candidate"] = record["candidate"]
    if before:
        payload["review"] = before
    if record.get("error"):
        payload["error"] = record["error"]
    try:
        candidate, generation = budget.stage(work / "calls" / f"{stage_prefix}.json", prompt or POST_PROMPT,
            payload, b.Candidate, 7000, key)
        b.validate(candidate)
        review, evaluation = budget.stage(work / "calls" / f"{stage_prefix}_review.json", b.REVIEW_PROMPT,
            {"source": payload["source"], "candidate": candidate.model_dump()}, a.Review, 2500, key)
        a.validate_review(review, candidate.output)
        after = review.model_dump()
        adopt = before is None or quality_rank(after) <= quality_rank(before)
        improved = before is None or quality_rank(after) < quality_rank(before)
        result.update(status="completed", after_scores=scores(after), after_blocking_issues=review.blocking_issues,
                      after_revision_instructions=review.revision_instructions,
                      adopted=adopt, improved=improved,
                      resolved=adopt and b.recommendation(review) == "pass")
        if adopt:
            record.update(status="completed", candidate=candidate.model_dump(), review=after,
                          api_recommendation=b.recommendation(review))
            record.pop("error", None)
        elif record["status"] != "completed" and before and "candidate" in record:
            # 보완 응답만 실패한 경우 검증된 초기 후보를 버리지 않는다.
            previous = b.Candidate.model_validate(record["candidate"])
            b.validate(previous)
            a.validate_review(a.Review.model_validate(before), previous.output)
            record.update(status="completed", review=before,
                          api_recommendation=b.recommendation(a.Review.model_validate(before)))
            record.pop("error", None)
            result["recovered_initial_candidate"] = True
        record.setdefault("calls", {}).update({stage_prefix: generation, f"{stage_prefix}_review": evaluation})
        record["post_repair"] = {"attempted": True, "adopted": adopt, "improved": improved,
                                 "previous_record": f"work/{number:04d}/before_post_repair.json"}
        a.save_json(work / "record.json", record)
    except b.BudgetExceeded as error:
        result.update(status="budget_wait", error=str(error))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    return result


def restore_cached_results(results, lessons, root, budget):
    restored = []
    for result in results:
        if result["status"] == "failed" and "기존 캐시와 설정이 다릅니다" in result.get("error", ""):
            number = result["source_id"]
            work = root / "work" / f"{number:04d}"
            if not all((work / path).exists() for path in
                       ("before_post_repair.json", "calls/post_repair.json", "calls/post_repair_review.json")):
                raise ValueError("집계 복구에 필요한 이전 후보·응답이 없습니다.")
            original = json.loads((work / "before_post_repair.json").read_text(encoding="utf-8"))
            result = repair(original, lessons[number], root, budget, "cached-only")
            if result["status"] != "completed":
                raise ValueError("저장된 응답으로 집계 복구 실패: " + str(number))
            result["metadata_restored_from_cache"] = True
        restored.append(result)
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-usd", type=float, default=38.3329241)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    if args.budget_usd <= 0:
        raise ValueError("양수 예산이 필요합니다.")
    root = b.ROOT
    with (root / "run.lock").open("r+b") as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        raw, labels, lessons = b.sources()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if hashlib.sha256(a.SOURCE.read_bytes()).hexdigest() != config["source_sha256"] or any(
            hashlib.sha256((a.SPLITS / f"{split}.jsonl").read_bytes()).hexdigest() != digest
            for split, digest in config["split_hashes"].items()):
            raise ValueError("1차 생성 이후 원문 또는 분할이 변경됐습니다.")
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("work/*/record.json"))]
        if len(records) != len(labels):
            raise ValueError("1차 전수 처리가 끝난 뒤 실행합니다.")
        budget = b.Budget(root, args.budget_usd)
        results_path = root / "post_repair_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
        results = restore_cached_results(results, lessons, root, budget)
        a.save_json(results_path, results)
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("work/*/record.json"))]
        attempted = {r["source_id"] for r in results if r["status"] in ("completed", "failed")}
        problems = sorted([r for r in records if r["source"]["number"] not in attempted and
            (r["status"] != "completed" or r["api_recommendation"] != "pass")], key=priority)
        a.save_json(root / "post_repair_plan.json", [{"source_id": r["source"]["number"],
            "title": r["source"]["title"], "status": r["status"], "priority": list(priority(r))} for r in problems])
        a.save_json(root / "post_repair_prompt.json", {"prompt": POST_PROMPT})
        a.save_json(root / "post_repair_config.json", {"model": a.MODEL, "budget_usd": args.budget_usd,
            "prompt_hash": a.digest(POST_PROMPT), "review_prompt_hash": a.digest(b.REVIEW_PROMPT),
            "source_sha256": config["source_sha256"], "acceptance": "품질 순위가 나빠지지 않으면 반영; 이전 버전은 보존"})
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            key = yaml.safe_load(a.KEY_FILE.read_text(encoding="utf-8"))["openai_api_key"].strip()
        if not key:
            raise ValueError("API 키가 필요합니다.")
        print(f"추가 보완 대상 {len(problems)}편 / 신규 실행 누적 상한 사용 ${budget.total():.4f}", flush=True)
        # 먼저 심각한 문제 그룹을 모두 마친 뒤 경미한 문제에 예산을 쓴다.
        groups = group_problems(problems)
        for jobs in groups.values():
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(repair, r, lessons[r["source"]["number"]], root, budget, key) for r in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    results = [r for r in results if r["source_id"] != result["source_id"]] + [result]
                    a.save_json(results_path, sorted(results, key=lambda r: r["source_id"]))
                    if result["status"] == "completed":
                        b.export(root, raw, labels, budget)
                    if result["status"] != "budget_wait":
                        print(f'{result["source_id"]}: {result["status"]} / 개선 {result.get("improved", False)} / 해결 {result.get("resolved", False)} / ${budget.total():.4f}', flush=True)
        summary = b.export(root, raw, labels, budget)
        print(json.dumps({**summary, "post_attempts_completed": sum(r["status"] == "completed" for r in results),
            "post_improved": sum(r.get("improved", False) for r in results),
            "post_resolved": sum(r.get("resolved", False) for r in results),
            "post_budget_wait": sum(r["status"] == "budget_wait" for r in results),
            "post_failed": [r["source_id"] for r in results if r["status"] == "failed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
