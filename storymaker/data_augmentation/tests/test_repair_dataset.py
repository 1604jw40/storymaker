import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import repair_dataset as r
class RepairTests(unittest.TestCase):
    def record(self, root):
        b.seed_existing(root, a.split_labels())
        path = root / "work/0004/record.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_quality_rank_prioritizes_serious_problems(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            record = self.record(Path(temp))
            bad = copy.deepcopy(record)
            bad["review"]["blocking_issues"] = ["중대한 문제"]
            failed = {**record, "status": "failed"}
            self.assertLess(r.priority(failed), r.priority(bad))
            self.assertLess(r.priority(bad), r.priority(record))
            self.assertGreater(r.quality_rank(bad["review"]), r.quality_rank(record["review"]))
            pacing = copy.deepcopy(record)
            pacing["review"]["interest_and_pacing"]["score"] = 0
            self.assertLess(r.priority(bad), r.priority(pacing))

    def test_success_keeps_old_version_and_adopts_improvement(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            record = self.record(root)
            old = copy.deepcopy(record)
            candidate = b.Candidate.model_validate(record["candidate"])
            review = a.Review.model_validate(record["review"])
            review.causality.score = 2
            review.revision_instructions = []
            budget = Mock()
            budget.stage.side_effect = [(candidate, {}), (review, {})]
            result = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key")
            self.assertTrue(result["improved"])
            self.assertTrue(result["resolved"])
            self.assertTrue(result["adopted"])
            self.assertEqual(json.loads((root / "work/0004/before_post_repair.json").read_text(encoding="utf-8")), old)
            saved = json.loads((root / "work/0004/record.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["api_recommendation"], "pass")
            self.assertEqual(saved["human_review"], "pending")

    def test_budget_failure_preserves_current_candidate(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            record = self.record(root)
            old = copy.deepcopy(record)
            budget = Mock()
            budget.stage.side_effect = [(b.Candidate.model_validate(record["candidate"]), {}), b.BudgetExceeded("상한")]
            result = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key")
            self.assertEqual(result["status"], "budget_wait")
            self.assertEqual(json.loads((root / "work/0004/record.json").read_text(encoding="utf-8")), old)

    def test_worse_revision_does_not_replace_current_candidate(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            record = self.record(root)
            old = copy.deepcopy(record)
            candidate = b.Candidate.model_validate(record["candidate"])
            candidate.output.scenes[0].visual_note = "변경된 그림 지시."
            review = a.Review.model_validate(record["review"])
            review.causality.score = 0
            budget = Mock()
            budget.stage.side_effect = [(candidate, {}), (review, {})]
            result = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key")
            self.assertFalse(result["adopted"])
            saved = json.loads((root / "work/0004/record.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["candidate"], old["candidate"])
            self.assertEqual(saved["review"], old["review"])

    def test_failed_revision_can_recover_valid_initial_candidate(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            record = self.record(root)
            record["initial_review"] = record.pop("review")
            record.update(status="failed", error="보완 응답의 언어 오류")
            a.save_json(root / "work/0004/record.json", record)
            old_candidate = copy.deepcopy(record["candidate"])
            review = a.Review.model_validate(record["initial_review"])
            review.causality.score = 0
            budget = Mock()
            budget.stage.side_effect = [(b.Candidate.model_validate(old_candidate), {}), (review, {})]
            result = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key")
            self.assertFalse(result["adopted"])
            self.assertTrue(result["recovered_initial_candidate"])
            saved = json.loads((root / "work/0004/record.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["candidate"], old_candidate)
            self.assertEqual(saved["review"], saved["initial_review"])

    def test_group_membership_does_not_change_after_repair(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            failed = self.record(Path(temp))
            failed["status"] = "failed"
            major = copy.deepcopy(failed)
            major["status"] = "completed"
            major["source"]["number"] = 12
            major["review"]["interest_and_pacing"]["score"] = 0
            groups = r.group_problems([failed, major])
            failed["status"] = "completed"
            major["review"]["interest_and_pacing"]["score"] = 1
            self.assertEqual([len(groups[g]) for g in (-1, 0, 1)], [1, 1, 0])
            self.assertEqual(len({id(item) for group in groups.values() for item in group}), 2)

    def test_cache_restores_success_metadata_without_paid_calls(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp, patch.object(a.requests, "post") as post:
            root = Path(temp)
            record = self.record(root)
            candidate = b.Candidate.model_validate(record["candidate"])
            review = a.Review.model_validate(record["review"])
            review.causality.score = 2
            review.revision_instructions = []
            def response(model, identifier):
                return {"status": "completed", "id": identifier, "model": a.MODEL,
                    "usage": {"input_tokens": 100, "output_tokens": 200,
                              "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": model.model_dump_json()}]}]}
            post.return_value.ok = True
            post.return_value.json.side_effect = [response(candidate, "gen"), response(review, "review")]
            budget = b.Budget(root, 1)
            first = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key")
            self.assertTrue(first["resolved"])
            cost = budget.total()
            failed = {"source_id": 4, "status": "failed", "error": "기존 캐시와 설정이 다릅니다"}
            restored = r.restore_cached_results([failed], {4: "작은 이도 도울 수 있다."}, root, budget)
            self.assertEqual(restored[0]["status"], "completed")
            self.assertTrue(restored[0]["metadata_restored_from_cache"])
            self.assertEqual(post.call_count, 2)
            self.assertEqual(budget.total(), cost)

    def test_language_retry_uses_separate_cache_paths(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            record = self.record(root)
            candidate = b.Candidate.model_validate(record["candidate"])
            review = a.Review.model_validate(record["review"])
            budget = Mock()
            budget.stage.side_effect = [(candidate, {}), (review, {})]
            result = r.repair(record, "작은 이도 도울 수 있다.", root, budget, "test-key",
                              stage_prefix="post_language_retry", prompt="한국어로 작성")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(budget.stage.call_args_list[0].args[0].name, "post_language_retry.json")
            self.assertEqual(budget.stage.call_args_list[1].args[0].name, "post_language_retry_review.json")
            self.assertEqual(budget.stage.call_args_list[0].args[1], "한국어로 작성")


if __name__ == "__main__":
    unittest.main()
