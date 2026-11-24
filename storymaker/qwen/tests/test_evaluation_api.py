import copy
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from storymaker.qwen.code import evaluation_api as api
from storymaker import project as p
def response(value, status=200):
    return SimpleNamespace(status_code=status, text=json.dumps(value), json=lambda: value)


class APIEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "api_calls/pilot_1"
        self.settings = {"endpoint": "https://api.openai.com/v1/responses", "model": "gpt-6.1-sol",
                         "reasoning_effort": "xhigh", "max_output_tokens_initial": 12288,
                         "budget_usd": 10, "short_context_prices_usd_per_million":
                         {"input": 2, "output": 10, "cached_input": .1, "input_cache_write_upper": 5}}
        story = {"title": "이야기", "characters": [], "scenes": [{"scene_no": 1, "character_ids": [],
                 "setting": "숲", "text_mode": "read_aloud", "text": "같이 걸었어요.", "visual_note": "숲길"}]}
        self.payload = {"moral": "친절", "A": {"raw_text": json.dumps(story), "format_problems": [], "ended_with_eos": True},
                        "B": {"raw_text": "broken JSON", "format_problems": ["invalid_json", "no_eos_before_limit"], "ended_with_eos": False}}
        self.judgment = {"preference": "A", "comparison_reason": "A만 유효함", "human_review_needed": True, "human_review_reason": "실패 확인"}
        for label in ("A", "B"):
            self.judgment[label] = {criterion: {"score": 3 if label == "A" else None,
                "scene_nos": [1] if label == "A" else [], "evidence": "장면1" if label == "A" else "형식 실패",
                "weakness": "", "minimal_fix": "", "uncertainty": ""} for criterion in api.CRITERIA}
        self.body = api.request_body(self.settings, "fixed rubric", self.payload)
        self.raw = {"id": "test-response", "model": "gpt-6.1-sol", "service_tier": "default", "status": "completed",
                    "usage": {"input_tokens": 1200, "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 30}},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(self.judgment)}]}]}

    def call(self):
        return api.call_pair(self.directory, self.body, self.payload, self.settings, "do-not-log-this-key")

    def test_scene_evidence_and_na_policy_are_enforced(self):
        api.validate_judgment(self.judgment, self.payload)
        wrong = copy.deepcopy(self.judgment)
        wrong["A"]["moral"]["scene_nos"] = [2]
        with self.assertRaisesRegex(AssertionError, "missing scene"):
            api.validate_judgment(wrong, self.payload)
        wrong = copy.deepcopy(self.judgment)
        wrong["B"]["moral"]["score"] = 3
        with self.assertRaisesRegex(AssertionError, "N/A"):
            api.validate_judgment(wrong, self.payload)

    def test_usage_cost_does_not_add_reasoning_twice(self):
        self.assertEqual(api.cost_upper(self.raw["usage"], self.settings["short_context_prices_usd_per_million"]), Decimal(".007"))

    def test_reviewed_output_cap_revision_preserves_initial_metadata(self):
        self.settings["max_output_tokens"] = 16384
        self.assertEqual(api.request_body(self.settings, "rubric", self.payload)["max_output_tokens"], 16384)
        self.assertEqual(self.settings["max_output_tokens_initial"], 12288)

    def test_prior_known_calibration_cost_is_included_in_budget(self):
        p.save(self.directory.parent / "budget_ledger.json", {"calls": [], "previous_protocol_calls": [
            {"cost_upper_usd": "9.9", "charge_status": "usage_known"}]})
        with patch.object(api.requests, "post", return_value=response({"input_tokens": 1200})) as post:
            with self.assertRaisesRegex(RuntimeError, "Budget insufficient"):
                self.call()
            self.assertEqual(post.call_count, 1)
        with self.assertRaisesRegex(AssertionError, "unresolved"):
            api.spent_upper({"calls": [], "previous_protocol_calls": [{"cost_upper_usd": "1", "charge_status": "unknown"}]})

    def test_no_paid_request_without_budget(self):
        self.settings["budget_usd"] = .00001
        with patch.object(api.requests, "post", return_value=response({"input_tokens": 1200})) as post:
            with self.assertRaisesRegex(RuntimeError, "Budget insufficient"):
                self.call()
            self.assertEqual(post.call_count, 1)
        self.assertFalse((self.directory.parent / "budget_ledger.json").exists())

    def test_token_count_failure_does_not_send_evaluation(self):
        with patch.object(api.requests, "post", return_value=response({"error": "count failed"}, 400)) as post:
            with self.assertRaisesRegex(RuntimeError, "counting failed"):
                self.call()
            self.assertEqual(post.call_count, 1)

    def test_reservation_precedes_request_and_cache_avoids_new_charge(self):
        def post(url, **kwargs):
            if url.endswith("input_tokens"):
                return response({"input_tokens": 1200})
            ledger = p.read(self.directory.parent / "budget_ledger.json")
            self.assertEqual(ledger["calls"][0]["state"], "reserved")
            return response(self.raw)
        with patch.object(api.requests, "post", side_effect=post) as network:
            self.assertEqual(self.call(), self.judgment)
            self.assertEqual(network.call_count, 2)
            self.assertEqual(self.call(), self.judgment)
            self.assertEqual(network.call_count, 2)
        ledger = p.read(self.directory.parent / "budget_ledger.json")
        self.assertEqual(ledger["calls"][0]["cost_upper_usd"], "0.007")
        self.assertTrue(all("do-not-log-this-key" not in file.read_text("utf-8") for file in Path(self.temp.name).rglob("*.json")))

    def test_timeout_keeps_reservation_and_never_resends(self):
        with patch.object(api.requests, "post", side_effect=[response({"input_tokens": 1200}), requests.Timeout()]) as network:
            with self.assertRaisesRegex(RuntimeError, "charge unresolved"):
                self.call()
            with self.assertRaisesRegex(RuntimeError, "Unresolved API attempt"):
                self.call()
            self.assertEqual(network.call_count, 2)
        self.assertEqual(p.read(self.directory.parent / "budget_ledger.json")["calls"][0]["state"], "uncertain")

    def test_incomplete_response_and_usage_are_preserved(self):
        self.raw["status"] = "incomplete"
        with patch.object(api.requests, "post", side_effect=[response({"input_tokens": 1200}), response(self.raw)]) as network:
            with self.assertRaisesRegex(RuntimeError, "Incomplete response"):
                self.call()
            with self.assertRaisesRegex(RuntimeError, "Unresolved API attempt"):
                self.call()
            self.assertEqual(network.call_count, 2)
        self.assertEqual(json.loads(p.read(self.directory / "response.json")["raw_body"])["status"], "incomplete")
        self.assertEqual(p.read(self.directory.parent / "budget_ledger.json")["calls"][0]["usage"]["output_tokens"], 100)

    def test_changed_request_cannot_reuse_old_result(self):
        with patch.object(api.requests, "post", side_effect=[response({"input_tokens": 1200}), response(self.raw)]) as network:
            self.call()
            self.body["reasoning"]["effort"] = "low"
            with self.assertRaisesRegex(AssertionError, "settings changed"):
                self.call()
            self.assertEqual(network.call_count, 2)

    def test_api_refuses_training_run_before_reading_key_or_network(self):
        run = Path(self.temp.name) / "run"
        p.save(run / "status.json", {"state": "training"})
        with patch.object(api.requests, "post") as network:
            with self.assertRaisesRegex(ValueError, "has not completed"):
                api.evaluate(run, "pilot", True)
            network.assert_not_called()

    def test_pilot_keeps_model_failures_but_rejects_wrong_classification(self):
        output = Path(self.temp.name)
        report, settings, rubric = output / "report.json", output / "storymaker.qwen.code.settings.json", output / "rubric.txt"
        p.save(report, {"phase": "pilot", "completed_pairs": 3, "all_responses_complete": True})
        p.save(settings, self.settings)
        rubric.write_text("fixed rubric", encoding="utf-8")
        pairs = [{"payload": self.payload}] * 3
        api.freeze_pilot(output, {"sha256": "fixed"}, pairs, report, settings, rubric)
        self.assertEqual(p.read(output / "protocol_frozen.json")["pilot_output_checks"]["strict_format_failures"], 3)
        self.assertEqual(p.read(output / "protocol_frozen.json")["pilot_output_checks"]["semantic_unscorable"], 3)
        self.payload["B"]["format_problems"] = []
        with self.assertRaisesRegex(AssertionError, "classification mismatch"):
            api.freeze_pilot(output, {"sha256": "fixed"}, pairs, report, settings, rubric)

    def test_readable_traits_array_keeps_semantic_scores_and_strict_failure(self):
        parsed = json.loads(self.payload["A"]["raw_text"])
        parsed["characters"] = [{"character_id": "c1", "name": "친구", "goal": "돕기", "traits": ["친절"]}]
        self.payload["A"]["raw_text"] = json.dumps(parsed)
        self.payload["A"]["format_problems"] = ["invalid_characters"]
        self.assertTrue(api.semantic_eligible(self.payload["A"])[0])
        api.validate_judgment(self.judgment, self.payload)
        self.assertEqual(self.payload["A"]["format_problems"], ["invalid_characters"])

    def test_good_pilot_freezes_inputs_for_test_generation(self):
        output = Path(self.temp.name)
        payload = copy.deepcopy(self.payload)
        payload["B"] = copy.deepcopy(payload["A"])
        report, settings, rubric = output / "report.json", output / "storymaker.qwen.code.settings.json", output / "rubric.txt"
        p.save(report, {"phase": "pilot", "completed_pairs": 3, "all_responses_complete": True})
        p.save(settings, self.settings)
        rubric.write_text("fixed rubric", encoding="utf-8")
        api.freeze_pilot(output, {"sha256": "fixed"}, [{"payload": payload}] * 3, report, settings, rubric)
        api.g.require_frozen_protocol(output, {"sha256": "fixed"})
        rubric.write_text("changed rubric", encoding="utf-8")
        with self.assertRaises(AssertionError):
            api.g.require_frozen_protocol(output, {"sha256": "fixed"})


if __name__ == "__main__":
    unittest.main()
