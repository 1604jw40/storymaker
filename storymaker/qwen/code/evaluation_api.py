"""고정된 원본/SFT 출력을 블라인드 평가한다. 요청 전 비용 확보, 실패 자동 재전송 없음."""
import argparse
import json
import msvcrt
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model

from storymaker.qwen.code import evaluation_generate as g
from storymaker import project as p
CRITERIA = ("moral", "causality", "emotion", "read_aloud", "scene_density", "text_visual", "age_fit")


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    score: int | None = Field(ge=0, le=4)
    scene_nos: list[int]
    evidence: str = Field(min_length=1)
    weakness: str
    minimal_fix: str
    uncertainty: str


Scores = create_model("Scores", __config__=ConfigDict(extra="forbid", strict=True), **{k: (Metric, ...) for k in CRITERIA})


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    A: Scores
    B: Scores
    preference: Literal["A", "B", "tie", "not_evaluable"]
    comparison_reason: str = Field(min_length=1)
    human_review_needed: bool
    human_review_reason: str


def now():
    return datetime.now(timezone.utc).isoformat()


def semantic_eligible(story):
    parsed, _ = g.generation.parse_story(story["raw_text"])
    readable = (isinstance(parsed, dict) and isinstance(parsed.get("scenes"), list) and bool(parsed["scenes"])
                and all(isinstance(s, dict) and isinstance(s.get("text"), str) for s in parsed["scenes"]))
    return bool(story["ended_with_eos"] and readable), parsed


def validate_judgment(value, payload):
    result = PairJudgment.model_validate(value)
    for label in ("A", "B"):
        story = payload[label]
        valid, parsed = semantic_eligible(story)
        scenes = {s["scene_no"] for s in parsed["scenes"] if type(s.get("scene_no")) is int} if valid else set()
        for metric in getattr(result, label).model_dump().values():
            assert (metric["score"] is not None) == valid, "Score/N/A differs from the fixed failure policy"
            assert set(metric["scene_nos"]) <= scenes, "Evidence refers to a missing scene"
    if all(not semantic_eligible(payload[k])[0] for k in ("A", "B")):
        assert result.preference == "not_evaluable", "Both outputs are invalid"
    return result.model_dump()


def request_body(settings, rubric, payload):
    return {"model": settings["model"], "store": False, "service_tier": "default",
            "reasoning": {"effort": settings["reasoning_effort"]},
            "max_output_tokens": settings.get("max_output_tokens", settings["max_output_tokens_initial"]),
            "input": [{"role": "system", "content": rubric},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "text": {"format": {"type": "json_schema", "name": "PairJudgment", "strict": True,
                                "schema": PairJudgment.model_json_schema()}}}


def cost_upper(usage, prices):
    inp, out = usage["input_tokens"], usage["output_tokens"]
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    assert all(type(n) is int and n >= 0 for n in (inp, out, cached)) and cached <= inp
    # 캐시 쓰기 TTL이 사용량에서 구분되지 않아도 12시간 단가로 비용 상한을 확보한다.
    return (Decimal(inp - cached) * Decimal(str(prices["input_cache_write_upper"]))
            + Decimal(cached) * Decimal(str(prices["cached_input"]))
            + Decimal(out) * Decimal(str(prices["output"]))) / 1_000_000


def spent_upper(ledger):
    previous = ledger.get("previous_protocol_calls", [])
    assert all(c.get("charge_status") == "usage_known" for c in previous), "Prior API charge is unresolved"
    return sum((Decimal(c["cost_upper_usd"]) for c in ledger["calls"] + previous), Decimal(0))


def freeze_pilot(output, manifest, pairs, report_path, settings_path, rubric_path):
    assert len(pairs) == 3
    report = p.read(report_path)
    assert report["phase"] == "pilot" and report["completed_pairs"] == 3 and report["all_responses_complete"]
    stories = [pair["payload"][label] for pair in pairs for label in ("A", "B")]
    for story in stories:
        parsed, problems = g.generation.parse_story(story["raw_text"])
        expected = problems + ([] if story["ended_with_eos"] else ["no_eos_before_limit"])
        assert story["format_problems"] == expected, "Pilot format/EOS classification mismatch"
    frozen = {"generation_manifest_sha256": manifest["sha256"], "frozen_at_utc": now()}
    frozen["pilot_output_checks"] = {"outputs": len(stories),
        "strict_format_failures": sum(bool(s["format_problems"]) for s in stories),
        "semantic_unscorable": sum(not semantic_eligible(s)[0] for s in stories),
        "policy": "Technical validation passed; model output failures remain in evaluation denominators, not resampled."}
    for name, path in {"pilot_api_report": report_path, "api_settings": settings_path, "rubric": rubric_path}.items():
        frozen[name], frozen[name + "_sha256"] = str(path), g.digest(path)
    p.save(output / "protocol_frozen.json", frozen)


def call_pair(directory, body, payload, settings, key):
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory.parent / "budget_ledger.json"
    ledger = p.read(ledger_path) if ledger_path.exists() else {"calls": []}
    signature = g.signature(body)
    for call in ledger["calls"]:
        if call["state"] != "completed":
            raise RuntimeError("Unresolved API attempt; do not resend automatically")
    existing = next((c for c in ledger["calls"] if c["directory"] == str(directory)), None)
    if existing:
        assert existing["request_sha256"] == signature, "Existing API request settings changed"
        envelope = p.read(directory / "response.json")
        assert envelope["request_sha256"] == signature
        return validate_judgment(p.read(directory / "judgment.json"), payload)
    assert not (directory / "response.json").exists(), "Response without ledger requires inspection"
    headers = {"Authorization": "Bearer " + key}
    counted = requests.post(settings["endpoint"] + "/input_tokens", headers=headers,
                            json={k: body[k] for k in ("model", "input", "text", "reasoning")},
                            timeout=(20, 120), allow_redirects=False)
    p.save(directory / "token_count.json", {"status_code": counted.status_code, "body": counted.text})
    if counted.status_code != 200:
        raise RuntimeError("Input token counting failed; generation request was not sent")
    count = counted.json()["input_tokens"]
    assert type(count) is int and 0 < count <= 272000, "Only short-context pricing is configured"
    prices = settings["short_context_prices_usd_per_million"]
    maximum = cost_upper({"input_tokens": count, "output_tokens": body["max_output_tokens"]}, prices)
    spent = spent_upper(ledger)
    if spent + maximum > Decimal(str(settings["budget_usd"])):
        raise RuntimeError("Budget insufficient for this request's maximum cost")
    call = {"directory": str(directory), "request_sha256": signature, "state": "reserved",
            "reserved_at_utc": now(), "input_tokens": count, "cost_upper_usd": str(maximum)}
    ledger["calls"].append(call)
    p.save(directory / "request.json", {"request_sha256": signature, "body": body})
    p.save(ledger_path, ledger)
    started = time.monotonic()
    try:
        response = requests.post(settings["endpoint"], headers=headers, json=body,
                                 timeout=(20, 1200), allow_redirects=False)
    except requests.RequestException:
        call.update(state="uncertain", reason="network_error_charge_unknown", finished_at_utc=now())
        p.save(ledger_path, ledger)
        raise RuntimeError("API connection failed; charge unresolved and no automatic retry") from None
    envelope = {"request_sha256": signature, "status_code": response.status_code,
                "seconds": time.monotonic() - started, "finished_at_utc": now(), "raw_body": response.text}
    p.save(directory / "response.json", envelope)
    call.update(state="needs_inspection", finished_at_utc=now())
    p.save(ledger_path, ledger)
    if response.status_code != 200:
        raise RuntimeError("API HTTP error preserved; no automatic retry")
    raw = response.json()
    usage = raw["usage"]
    actual_upper = cost_upper(usage, prices)
    call.update(response_id=raw.get("id"), usage=usage, cost_upper_usd=str(actual_upper))
    p.save(ledger_path, ledger)
    assert usage["input_tokens"] == count and usage["output_tokens"] <= body["max_output_tokens"]
    assert actual_upper <= maximum and raw["model"] == settings["model"]
    assert raw.get("service_tier") == "default", "Unexpected service tier"
    if raw.get("status") != "completed":
        raise RuntimeError("Incomplete response preserved; inspect before proceeding")
    texts = [c["text"] for item in raw.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    assert len(texts) == 1, "Missing structured answer or refusal"
    judgment = validate_judgment(json.loads(texts[0]), payload)
    p.save(directory / "judgment.json", judgment)
    call["state"] = "completed"
    p.save(ledger_path, ledger)
    return judgment


def evaluate(run, split, execute):
    context = g.completed_context(run)
    output = run / "evaluation"
    manifest = g.manifest_for(context)
    assert p.read(output / "generation_manifest.json") == manifest
    if split == "test":
        g.require_frozen_protocol(output, manifest)
    settings_path = p.ROOT / "docs/test61_api_settings.json"
    rubric_path = p.ROOT / "docs/test61_rubric.txt"
    settings, rubric = p.read(settings_path), rubric_path.read_text("utf-8")
    assert settings["model"] == "gpt-6.1-sol" and settings["reasoning_effort"] == "xhigh"
    assert settings["service_tier"] == "default" and settings["store"] is False and not settings["tools"]
    assert settings["max_automatic_retries"] == 0 and settings["budget_usd"] == 10
    jobs = g.jobs_for(run, output, context, manifest, split)
    pairs = []
    for index in range(0, len(jobs), 2):
        pair = jobs[index:index + 2]
        payload, hashes = {"moral": pair[0]["moral"]}, {}
        mapping = {}
        for label, job in zip(("A", "B"), pair):
            path = Path(job["output"])
            cached = p.read(path)
            assert cached["state"] == "completed" and cached["job"] == job
            result = cached["generation"]
            g.validate_generation(result, job, context)
            payload[label] = {k: result[k] for k in ("raw_text", "format_problems", "ended_with_eos")}
            hashes[str(path)] = g.digest(path)
            mapping[label] = job["variant"]
        pairs.append({"source_id": pair[0]["source_id"], "payload": payload, "mapping": mapping,
                      "generation_sha256": hashes})
    identity = {"generation_manifest_sha256": manifest["sha256"], "settings_sha256": g.digest(settings_path),
                "rubric_sha256": g.digest(rubric_path), "api_code_sha256": g.digest(Path(__file__)),
                "schema": PairJudgment.model_json_schema()}
    api_manifest = output / "api_manifest.json"
    if api_manifest.exists():
        assert p.read(api_manifest) == identity, "API evaluation protocol changed"
    else:
        p.save(api_manifest, identity)
    plan = {"phase": split, "pairs": len(pairs), "budget_usd": settings["budget_usd"], "execute": execute,
            "max_output_tokens": settings.get("max_output_tokens", settings["max_output_tokens_initial"])}
    p.save(output / f"api_{split}_plan.json", plan)
    if not execute:
        return plan
    pricing = p.read(p.ROOT / "reports/api_pricing_check.json")
    assert pricing["checked_client_date"] == datetime.now(timezone(timedelta(hours=9))).date().isoformat(), "Recheck official prices before execution"
    assert pricing["model"] == settings["model"] and pricing["prices_usd_per_million"] == settings["short_context_prices_usd_per_million"]
    key = yaml.safe_load(Path(settings["credential_file"]).read_text("utf-8-sig"))["openai_api_key"]
    assert isinstance(key, str) and key.strip()
    completed = []
    for pair in pairs:
        directory = output / "api_calls" / f"{split}_{pair['source_id']}"
        p.save(output / "api_status.json", {"state": "evaluating", "phase": split, "completed_pairs": len(completed), "total_pairs": len(pairs), "source_id": pair["source_id"], "updated_at_utc": now()})
        p.save(directory / "pair.json", pair)
        judgment = call_pair(directory, request_body(settings, rubric, pair["payload"]), pair["payload"], settings, key)
        assert all(g.digest(path) == sha for path, sha in pair["generation_sha256"].items())
        completed.append({"source_id": pair["source_id"], "mapping": pair["mapping"], "directory": str(directory), "judgment": judgment, "generation_sha256": pair["generation_sha256"]})
    report_path = output / f"api_{split}_report.json"
    p.save(report_path, {"phase": split, "completed_pairs": len(completed), "all_responses_complete": True,
                        "api_manifest": identity, "pairs": completed, "human_review_performed": False})
    if split == "pilot":
        freeze_pilot(output, manifest, pairs, report_path, settings_path, rubric_path)
    p.save(output / "api_status.json", {"state": "phase_complete", "phase": split, "completed_pairs": len(completed), "updated_at_utc": now()})
    return {**plan, "completed_pairs": len(completed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--execute", action="store_true", help="Send the planned API evaluation requests")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (p.ROOT / "outputs").resolve() not in run.parents:
        raise ValueError("Run outside project outputs")
    assert Path(p.read(p.ROOT.parent / "outputs/sft_v5/active_run.json")["run_dir"]).resolve() == run, "Use the active SFT run"
    with (p.ROOT / "outputs/evaluation_api.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            print(json.dumps(evaluate(run, args.split, args.execute), ensure_ascii=False))
        except Exception as exc:
            p.save(run / "evaluation/api_status.json", {"state": "needs_attention", "phase": args.split, "error_type": type(exc).__name__, "message": str(exc), "updated_at_utc": now()})
            raise


if __name__ == "__main__":
    main()
