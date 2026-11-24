import copy
import json
import unittest

from storymaker.qwen.code import evaluation_report as report
class EvaluationReportTests(unittest.TestCase):
    def fixture(self):
        ids = list(range(1, 62))
        rows, outputs = [], {}
        for number in ids:
            mapping = {"A": "base", "B": "sft"} if number % 2 else {"A": "sft", "B": "base"}
            judgment = {"preference": next(k for k, v in mapping.items() if v == "sft"), "human_review_needed": False}
            for label, variant in mapping.items():
                judgment[label] = {c: {"score": 3 if variant == "sft" else 2} for c in report.api.CRITERIA}
                outputs[number, variant] = {"format_problems": [], "ended_with_eos": True, "raw_text": "not a parsed story"}
            rows.append({"source_id": number, "mapping": mapping, "judgment": judgment})
        return {"phase": "test", "completed_pairs": 61, "all_responses_complete": True, "pairs": rows}, ids, outputs

    def test_mapping_and_source_denominators(self):
        data, ids, outputs = self.fixture()
        result = report.summarize(data, ids, outputs, [])
        self.assertEqual(result["api_preferences"], {"sft": 61})
        self.assertEqual(result["models"]["sft"]["quality_pass_count"], 61)
        self.assertEqual(result["models"]["base"]["quality_pass_count"], 0)
        for metric in result["paired_criteria"].values():
            self.assertEqual(metric["paired_valid_n"], 61)
            self.assertEqual(metric["mean_sft_minus_base"], 1)
            self.assertEqual(metric["bootstrap_95_interval"], [1, 1])

    def test_failure_stays_in_full_denominator_but_not_paired_mean(self):
        data, ids, outputs = self.fixture()
        outputs[1, "sft"]["format_problems"] = ["invalid_json"]
        for metric in data["pairs"][0]["judgment"]["B"].values():
            metric["score"] = None
        result = report.summarize(data, ids, outputs, [])
        self.assertEqual(result["models"]["sft"]["quality_pass_count"], 60)
        self.assertEqual(result["models"]["sft"]["quality_pass_denominator"], 61)
        self.assertEqual(result["paired_criteria"]["moral"]["paired_valid_n"], 60)

    def test_duplicate_source_is_rejected(self):
        data, ids, outputs = self.fixture()
        data["pairs"][-1]["source_id"] = 1
        with self.assertRaises(AssertionError):
            report.summarize(data, ids, outputs, [])
        self.assertIsNone(report.interval([1]))

    def test_readable_schema_failure_keeps_semantic_pair_but_fails_product_pass(self):
        data, ids, outputs = self.fixture()
        outputs[1, "sft"]["format_problems"] = ["invalid_characters"]
        result = report.summarize(data, ids, outputs, [])
        self.assertEqual(result["models"]["sft"]["format_valid_count"], 60)
        self.assertEqual(result["models"]["sft"]["quality_pass_count"], 60)
        self.assertEqual(result["models"]["sft"]["quality_pass_denominator"], 61)
        self.assertEqual(result["paired_criteria"]["moral"]["paired_valid_n"], 61)

    def test_long_phrase_finds_shifted_match_without_json_key_matches(self):
        phrase = "가나다라마바사아자차카타파하" * 4
        story = {"title": "이야기", "characters": [], "scenes": [{"scene_no": 1, "character_ids": [],
                 "setting": "숲", "text_mode": "read_aloud", "text": "서문" + phrase, "visual_note": "숲"}]}
        development = [{"source_id": 99, "output": copy.deepcopy(story)}]
        story["scenes"][0]["text"] = "다른시작" + phrase
        outputs = {(1, "sft"): {"raw_text": json.dumps(story)}}
        result = report.long_matches(development, outputs)
        self.assertEqual(result["matches"][0]["development_source_id"], 99)
        self.assertGreaterEqual(result["matches"][0]["normalized_contiguous_characters"], len(phrase))
        story["scenes"][0]["text"] = "전혀다른짧은문장"
        self.assertEqual(report.long_matches(development, {(1, "sft"): {"raw_text": json.dumps(story)}})["matches"], [])


if __name__ == "__main__":
    unittest.main()
