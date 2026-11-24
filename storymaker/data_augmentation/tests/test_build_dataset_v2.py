import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import build_dataset_v2 as v
class NewDatasetTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads(v.STYLE_FILE.read_text(encoding="utf-8"))
        self.candidate = b.Candidate(moral=self.example["input"]["moral"], output=self.example["output"])
        self.source = {"number": 4, "title": "사자와 생쥐", "story": "사자가 생쥐를 놓아주고 생쥐가 밧줄을 끊었다."}

    def review(self, score=2):
        grade = {"score": 2, "scene_refs": [1], "reason": "실제 장면 근거"}
        data = {k: copy.deepcopy(grade) for k in list(a.Review.model_fields)[:6]}
        data["character_consistency"]["score"] = score
        return a.Review.model_validate({**data, "blocking_issues": [],
            "revision_instructions": [] if score == 2 else ["감정과 선택의 연결을 고친다."]})

    def test_fresh_source_payload_and_minor_issue_does_not_spend_on_rewrite(self):
        budget = Mock()
        budget.stage.side_effect = [(self.candidate, {}), (self.review(1), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            record = v.process(self.source, "train", "교훈", budget, "new prompt", "test-key")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["api_recommendation"], "revise")
        self.assertEqual(budget.stage.call_count, 2)
        self.assertNotIn("candidate", budget.stage.call_args_list[0].args[2])
        self.assertEqual(budget.stage.call_args_list[0].args[2]["source"]["story"], self.source["story"])
        self.assertEqual([c.kwargs["effort"] for c in budget.stage.call_args_list], ["medium", "medium"])

    def test_major_issue_gets_exactly_one_high_rewrite(self):
        revised = self.candidate.model_copy(deep=True)
        revised.output.scenes[0].text = "사자는 편안히 잠을 잤어요."
        budget = Mock()
        budget.stage.side_effect = [(self.candidate, {}), (self.review(0), {}),
                                    (revised, {}), (self.review(), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            record = v.process(self.source, "train", "교훈", budget, "new prompt", "test-key")
        self.assertTrue(record["revision_adopted"])
        self.assertEqual(record["candidate"], revised.model_dump())
        self.assertEqual([c.kwargs["effort"] for c in budget.stage.call_args_list],
                         ["medium", "medium", "high", "medium"])

    def test_authorized_retry_uses_separate_generation_history(self):
        budget = Mock()
        budget.stage.side_effect = [(self.candidate, {}), (self.review(), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            record = v.process(self.source, "train", "교훈", budget, "new prompt", "test-key",
                               generation_stage="generate_retry1")
            self.assertFalse((Path(directory) / "work/0004/calls/generate.json").exists())
        self.assertEqual(budget.stage.call_args_list[0].args[0].name, "generate_retry1.json")
        self.assertEqual(record["status"], "completed")
        self.assertIn("generate_retry1", record["calls"])
        self.assertNotIn("generate", record["calls"])

    def test_budget_deferred_rewrite_preserves_initial_candidate(self):
        budget = Mock()
        budget.stage.side_effect = [(self.candidate, {}), (self.review(0), {}), b.BudgetExceeded("cap")]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            record = v.process(self.source, "train", "교훈", budget, "new prompt", "test-key")
        self.assertEqual(record["candidate"], self.candidate.model_dump())
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["revision_deferred"])
        self.assertEqual(budget.stage.call_count, 3)

    def test_truncated_first_response_can_rewrite_once_from_source(self):
        budget = Mock()
        budget.stage.side_effect = [ValueError("API 응답 미완료: incomplete"),
                                   (self.candidate, {}), (self.review(), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            record = v.process(self.source, "train", "교훈", budget, "new prompt", "test-key")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(budget.stage.call_count, 3)
        self.assertNotIn("candidate", budget.stage.call_args_list[1].args[2])
        self.assertIn("error", budget.stage.call_args_list[1].args[2])

    def test_export_uses_v2_training_prompt_and_does_not_double_count_reasoning(self):
        raw, labels, _ = b.sources()
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            root = Path(directory)
            a.save_json(root / "work/0004/record.json", {"source": self.source,
                "source_split": labels[4], "status": "completed", "human_review": "pending",
                "candidate": self.candidate.model_dump(), "review": self.review().model_dump(),
                "api_recommendation": "pass", "calls": {}})
            a.save_json(root / "work/0004/calls/generate.json", {"reasoning_effort": "medium",
                "seconds": 2, "response": {"usage": {"input_tokens": 100, "output_tokens": 80,
                    "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 50}}}})
            summary = v.export(raw, labels, b.Budget(root, 100))
            rows = [json.loads(line) for line in (root / f"{labels[4]}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["id"], "aesop_0004_scenes_sft_v2")
            self.assertEqual(rows[0]["messages"][0]["content"], v.SYSTEM_PROMPT)
            self.assertEqual(json.loads(rows[0]["messages"][2]["content"]), self.candidate.output.model_dump())
            self.assertEqual(summary["api_tokens"]["total_tokens"], 180)
            self.assertEqual(summary["api_tokens"]["reasoning_tokens"], 50)
            self.assertEqual(summary["reused_existing"], 0)
            missing = {**self.source, "number": 71}
            a.save_json(root / "work/0071/record.json", {"source": missing, "source_split": "test",
                "status": "failed", "human_review": "pending", "error": "unknown request"})
            budget = b.Budget(root, 100)
            name = str((root / "work/0071/calls/generate.json").relative_to(root))
            budget.entries[name] = {"status": "unknown", "usd": 0.14728}
            summary = v.export([self.source, missing], {4: "train", 71: "test"}, budget)
            self.assertEqual(summary["blocked_source_ids"], [71])
            self.assertEqual(summary["automatic_remaining_sources"], 0)
            self.assertIsNone(summary["estimate"]["remaining_seconds"])

    def test_effort_is_sent_and_cached_responses_cannot_cross_efforts(self):
        raw = {"id": "resp-test", "status": "completed", "model": a.MODEL,
            "usage": {"input_tokens": 10, "output_tokens": 20, "input_tokens_details": {}},
            "output": [{"type": "message", "content": [{"type": "output_text",
                "text": self.candidate.model_dump_json()}]}]}
        response = Mock(ok=True)
        response.json.return_value = raw
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(a.requests, "post", return_value=response) as post:
            path = Path(directory) / "calls/generate.json"
            a.api_stage(path, "prompt", {}, b.Candidate, 12000, "test-key", effort="medium")
            a.api_stage(path, "prompt", {}, b.Candidate, 12000, "test-key", effort="medium")
            with self.assertRaisesRegex(ValueError, "캐시"):
                a.api_stage(path, "prompt", {}, b.Candidate, 12000, "test-key", effort="high")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["json"]["reasoning"], {"effort": "medium"})
            self.assertEqual(post.call_args.kwargs["timeout"], (20, 240))
            a.api_stage(Path(directory) / "calls/revision.json", "prompt", {},
                        b.Candidate, 16000, "test-key", effort="high")
            self.assertEqual(post.call_args.kwargs["timeout"], (20, 600))

    def test_unknown_request_is_not_retried_but_untouched_work_can_continue(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory:
            root = Path(directory)
            budget = b.Budget(root, 100)
            unknown = root / "work/0023/calls/revision1.json"
            name = str(unknown.relative_to(root))
            budget.entries[name] = {"status": "unknown", "usd": 0.22}
            a.save_json(budget.path, budget.entries)
            def stage(path, *args, **kwargs):
                a.save_json(path, {"response": {"usage": {"input_tokens": 100,
                    "output_tokens": 100, "input_tokens_details": {"cache_write_tokens": 0}}}})
                return self.candidate, {}
            with patch.object(a, "api_stage", side_effect=stage) as api:
                with self.assertRaisesRegex(RuntimeError, "재전송"):
                    budget.stage(unknown, "prompt", {}, b.Candidate, 16000, "test-key", effort="high")
                api.assert_not_called()
                budget.stage(root / "work/0024/calls/generate.json", "prompt", {},
                             b.Candidate, 12000, "test-key", effort="medium")
                self.assertEqual(api.call_count, 1)
            self.assertEqual(budget.entries[name], {"status": "unknown", "usd": 0.22})
            self.assertAlmostEqual(budget.total(), 0.2212)


if __name__ == "__main__":
    unittest.main()
