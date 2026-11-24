"""고정 test61 결과 집계. 작품 단위 재표집과 개발 v5의 긴 동일 구절 후보를 기록한다."""
import argparse
import difflib
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from storymaker.qwen.code import evaluation_api as api
from storymaker.qwen.code import evaluation_generate as g
from storymaker import project as p
def interval(values, repeats=4000):
    if len(values) < 2:
        return None
    rng = random.Random(42)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [samples[int((repeats - 1) * q)] for q in (.025, .975)]


def narrative(story):
    text = " ".join([story.get("title", "")] + [s.get("text", "") for s in story.get("scenes", [])])
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def long_matches(development, outputs, minimum=40):
    width = minimum // 2
    texts = {row["source_id"]: narrative(row["output"]) for row in development}
    index = defaultdict(set)
    for number, text in texts.items():
        for start in range(0, len(text) - width + 1, width):
            index[text[start:start + width]].add(number)
    matches, checked = [], 0
    for (number, variant), result in outputs.items():
        story, problems = g.generation.parse_story(result["raw_text"])
        if not (isinstance(story, dict) and isinstance(story.get("scenes"), list)
                and all(isinstance(s, dict) and isinstance(s.get("text"), str) for s in story["scenes"])):
            continue
        text = narrative(story)
        checked += 1
        candidates = set()
        for start in range(len(text) - width + 1):
            candidates.update(index.get(text[start:start + width], ()))
        for source in sorted(candidates):
            found = difflib.SequenceMatcher(None, text, texts[source], autojunk=False).find_longest_match()
            if found.size >= minimum:
                matches.append({"source_id": number, "variant": variant, "development_source_id": source,
                                "normalized_contiguous_characters": found.size,
                                "matched_text": text[found.a:found.a + found.size]})
    return {"minimum_normalized_characters": minimum, "checked_outputs": checked,
            "normalization": "NFKC; letters/digits only; title and scene text; JSON keys excluded",
            "matches": sorted(matches, key=lambda x: -x["normalized_contiguous_characters"]),
            "interpretation": "Review candidates, not proof of memorization; common fable motifs and phrases may overlap. Semantic paraphrase duplication is not measured."}


def summarize(report, test_ids, outputs, development):
    assert report["phase"] == "test" and report["all_responses_complete"] and report["completed_pairs"] == 61
    rows = report["pairs"]
    assert len(rows) == len({row["source_id"] for row in rows}) == len(test_ids) == 61
    assert {row["source_id"] for row in rows} == set(test_ids)
    assert set(outputs) == {(number, variant) for number in test_ids for variant in ("base", "sft")}
    scores, passed = {}, {}
    for row in rows:
        assert set(row["mapping"]) == {"A", "B"} and set(row["mapping"].values()) == {"base", "sft"}
        for label, variant in row["mapping"].items():
            item = row["judgment"][label]
            scores[row["source_id"], variant] = item
            generated = outputs[row["source_id"], variant]
            passed[row["source_id"], variant] = int(not generated["format_problems"] and generated["ended_with_eos"]
                and all(item[c]["score"] is not None and item[c]["score"] >= 3 for c in api.CRITERIA))
    models, paired = {}, {}
    for variant in ("base", "sft"):
        generated = [outputs[number, variant] for number in test_ids]
        metrics = {}
        for criterion in api.CRITERIA:
            values = [scores[number, variant][criterion]["score"] for number in test_ids
                      if scores[number, variant][criterion]["score"] is not None]
            metrics[criterion] = {"mean_valid_only": statistics.mean(values) if values else None,
                                  "valid_n": len(values), "below_three_count": sum(v < 3 for v in values)}
        models[variant] = {"total_n": 61, "format_valid_count": sum(not r["format_problems"] for r in generated),
            "eos_count": sum(r["ended_with_eos"] for r in generated),
            "quality_pass_count": sum(passed[number, variant] for number in test_ids),
            "quality_pass_denominator": 61, "criteria": metrics}
    for criterion in api.CRITERIA:
        values = [scores[number, "sft"][criterion]["score"] - scores[number, "base"][criterion]["score"]
                  for number in test_ids if all(scores[number, v][criterion]["score"] is not None for v in ("base", "sft"))]
        paired[criterion] = {"paired_valid_n": len(values), "mean_sft_minus_base": statistics.mean(values) if values else None,
                             "bootstrap_95_interval": interval(values)}
    pass_differences = [passed[number, "sft"] - passed[number, "base"] for number in test_ids]
    preferences = Counter()
    examples = []
    for row in rows:
        choice = row["judgment"]["preference"]
        preferences[row["mapping"].get(choice, choice)] += 1
        if row["judgment"]["human_review_needed"]:
            examples.append({"source_id": row["source_id"], "reason": row["judgment"]["human_review_reason"],
                             "directory": row["directory"]})
    return {"test_source_count": 61, "models": models, "paired_criteria": paired,
            "quality_pass_rate_sft_minus_base": {"difference": statistics.mean(pass_differences),
                "bootstrap_95_interval": interval(pass_differences), "denominator": 61},
            "api_preferences": dict(preferences), "human_review_candidates": examples,
            "copy_check": long_matches(development, outputs),
            "bootstrap": {"unit": "source_id", "repeats": 4000, "seed": 42,
                          "method": "paired percentile resampling; criteria only jointly scorable pairs; quality pass includes all61"},
            "limitations": ["Single decoding seed and one output per model/source; no multiple training seeds.",
                "Valid-only means have a different denominator from full-set quality pass rates.",
                "API judge opinion, not human/expert certification; same-family data curation/judging may bias preferences.",
                "No images were generated; text/visual-note consistency only.",
                "If test diagnoses inform RL design, these61 are no longer an unseen RL final evaluation set."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = p.sft_result_path(args.run_dir)
    output = p.sft_result_path(args.output_dir or run / "evaluation")
    context = g.completed_context(run)
    manifest = g.manifest_for(context)
    assert p.read(output / "generation_manifest.json") == manifest
    g.require_frozen_protocol(output, manifest)
    report = p.read(output / "api_test_report.json")
    assert report["api_manifest"]["generation_manifest_sha256"] == manifest["sha256"]
    assert report["api_manifest"]["api_code_sha256"] == g.digest(Path(api.__file__))
    outputs = {}
    for job in g.jobs_for(run, output, context, manifest, "test"):
        record = p.read(job["output"])
        assert record["state"] == "completed" and record["job"] == job
        g.validate_generation(record["generation"], job, context)
        outputs[job["source_id"], job["variant"]] = record["generation"]
    for pair in report["pairs"]:
        assert all(g.digest(path) == expected for path, expected in pair["generation_sha256"].items())
        stored = p.read(Path(pair["directory"]) / "pair.json")
        assert stored["mapping"] == pair["mapping"] and stored["source_id"] == pair["source_id"]
        for label, variant in pair["mapping"].items():
            assert stored["payload"][label] == {k: outputs[pair["source_id"], variant][k] for k in ("raw_text", "format_problems", "ended_with_eos")}
        assert pair["judgment"] == p.read(Path(pair["directory"]) / "judgment.json")
        api.validate_judgment(pair["judgment"], stored["payload"])
    result = summarize(report, manifest["identity"]["test_ids"], outputs,
                       [r for r in context["books"] if r["source_split"] != "test"])
    ledger = p.read(output / "api_calls/budget_ledger.json")
    assert len(ledger["calls"]) == len({c["directory"] for c in ledger["calls"]}) == 64
    assert all(call["state"] == "completed" for call in ledger["calls"])
    result.update(generation_manifest_sha256=manifest["sha256"], report_code_sha256=g.digest(Path(__file__)),
                  api_cost_upper_usd=str(api.spent_upper(ledger)),
                  api_evaluation_calls=len(ledger["calls"]),
                  api_request_attempts=len(ledger["calls"]) + len(ledger.get("previous_protocol_calls", [])),
                  excluded_calibration_attempts=len(ledger.get("previous_protocol_calls", [])), human_review_performed=False)
    p.save(output / "evaluation_summary.json", result)
    print(output / "evaluation_summary.json")


if __name__ == "__main__":
    main()
