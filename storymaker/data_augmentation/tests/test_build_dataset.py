import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
class DatasetTests(unittest.TestCase):
    def candidate(self):
        raw = json.loads((a.ROOT / "outputs/sft_20_scenes_v1/사자와_생쥐.json").read_text(encoding="utf-8"))
        return b.Candidate(moral=raw["input"]["moral"], output=raw["output"])

    def test_core_validation_and_wordless_preservation(self):
        candidate = self.candidate()
        b.validate(candidate)
        candidate.output.scenes[5].text = "말 없는 장면의 잘못된 글"
        with self.assertRaisesRegex(ValueError, "빈 문자열"):
            b.validate(candidate)
        candidate = self.candidate()
        candidate.output.scenes[0].character_ids = ["unknown"]
        with self.assertRaisesRegex(ValueError, "참조"):
            b.validate(candidate)

    def test_budget_blocks_before_paid_request(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp, patch.object(a, "api_stage") as api:
            root = Path(temp)
            budget = b.Budget(root, 0.00001)
            with self.assertRaises(b.BudgetExceeded):
                budget.stage(root / "work/0001/calls/generate.json", "prompt", {}, b.Candidate, 7000, "test-key")
            api.assert_not_called()
            self.assertEqual(budget.total(), 0)

    def test_unknown_charge_is_reserved_and_not_retried(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp, patch.object(a, "api_stage", side_effect=RuntimeError("timeout")) as api:
            root = Path(temp)
            budget = b.Budget(root, 1)
            path = root / "work/0001/calls/generate.json"
            with self.assertRaises(RuntimeError):
                budget.stage(path, "prompt", {}, b.Candidate, 7000, "test-key")
            self.assertGreater(budget.total(), 0)
            restored = b.Budget(root, 1)
            with self.assertRaisesRegex(RuntimeError, "재전송"):
                restored.stage(path, "prompt", {}, b.Candidate, 7000, "test-key")
            self.assertEqual(api.call_count, 1)

    def test_settlement_uses_actual_usage_and_caches(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            budget = b.Budget(root, 1)
            path = root / "work/0001/calls/generate.json"
            def stage(*args):
                a.save_json(path, {"response": {"usage": {"input_tokens": 100, "output_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}}})
                return self.candidate(), {}
            with patch.object(a, "api_stage", side_effect=stage):
                budget.stage(path, "prompt", {}, b.Candidate, 7000, "test-key")
                charged = budget.total()
                budget.stage(path, "prompt", {}, b.Candidate, 7000, "test-key")
            self.assertAlmostEqual(charged, 0.0012)
            self.assertEqual(budget.total(), charged)

    def test_missing_usage_does_not_release_reserved_cost(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            budget = b.Budget(root, 1)
            path = root / "work/0001/calls/generate.json"
            def stage(*args):
                a.save_json(path, {"response": {"usage": None}})
                return self.candidate(), {}
            with patch.object(a, "api_stage", side_effect=stage):
                budget.stage(path, "prompt", {}, b.Candidate, 7000, "test-key")
            self.assertGreater(budget.total(), 0)
            self.assertEqual(next(iter(budget.snapshot().values()))["status"], "unknown")
            self.assertEqual(b.Budget(root, 1).total(), budget.total())

    def test_export_preserves_splits_and_keeps_blocking_out_of_training(self):
        raw, labels, _ = b.sources()
        self.assertEqual(len(labels), 304)
        self.assertEqual(len(raw) - len(labels), 8)
        with tempfile.TemporaryDirectory(dir=a.ROOT) as temp:
            root = Path(temp)
            b.seed_existing(root, labels)
            bad_path = root / "work/0208/record.json"
            bad = json.loads(bad_path.read_text(encoding="utf-8"))
            bad["review"]["blocking_issues"] = ["검토 전용 표식"]
            a.save_json(bad_path, bad)
            summary = b.export(root, raw, labels, b.Budget(root, 38))
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["quality_excluded"], 1)
            self.assertEqual(summary["dataset_examples"], 1)
            rows = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["source_id"], 4)
            self.assertEqual(rows[0]["source_split"], labels[4])
            self.assertNotIn("검토 전용 표식", json.dumps(rows, ensure_ascii=False))
            self.assertEqual(json.loads(rows[0]["messages"][2]["content"]), self.candidate().output.model_dump())


if __name__ == "__main__":
    unittest.main()
