import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import build_dataset_v2 as v2
from storymaker.data_augmentation.code import revise_dataset_v3 as v
class FlowRevisionTests(unittest.TestCase):
    def setUp(self):
        row = json.loads(v2.STYLE_FILE.read_text(encoding="utf-8"))
        self.candidate = b.Candidate(moral=row["input"]["moral"], output=row["output"])
        self.shorter = self.candidate.model_copy(deep=True)
        self.shorter.output.scenes = self.shorter.output.scenes[:3]
        self.source = {"number": 4, "title": "사자와 생쥐", "story": "원작 우화"}

    def review(self, score=2):
        grade = {"score": 2, "scene_refs": [1], "reason": "실제 장면 근거"}
        fields = {k: copy.deepcopy(grade) for k in list(a.Review.model_fields)[:6]}
        fields["interest_and_pacing"]["score"] = score
        return a.Review.model_validate({**fields, "blocking_issues": [],
            "revision_instructions": [] if score == 2 else ["반복 장면을 합친다."]})

    def seed(self, root, review=None):
        record = {"source": self.source, "source_split": "train", "reference_moral": "교훈",
            "candidate": self.candidate.model_dump(), "review": (review or self.review(1)).model_dump(),
            "api_recommendation": "revise", "status": "completed", "human_review": "pending",
            "reused_existing": True, "calls": {}, "selected_attempt": 0, "correction_status": "pending"}
        a.save_json(root / "work/0004/record.json", record)

    def test_variable_count_still_checks_numbering_wordless_and_character_references(self):
        v.validate(self.shorter)
        with self.assertRaises(ValueError):
            b.validate(self.shorter)
        for change in ("number", "reference", "wordless", "empty"):
            candidate = self.shorter.model_copy(deep=True)
            if change == "number":
                candidate.output.scenes[1].scene_no = 8
            elif change == "reference":
                candidate.output.scenes[0].character_ids = ["unknown"]
            elif change == "wordless":
                candidate.output.scenes[0].text_mode = "wordless"
            else:
                candidate.output.scenes = []
            with self.assertRaises(ValueError):
                v.validate(candidate)

    def test_one_revision_and_review_when_resolved(self):
        budget = Mock()
        budget.stage.side_effect = [(self.shorter, {}), (self.review(), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            self.seed(v.ROOT)
            record = v.process(4, budget, "test-key")
        self.assertEqual(record["correction_status"], "completed")
        self.assertEqual(record["selected_attempt"], 1)
        self.assertEqual(len(record["candidate"]["output"]["scenes"]), 3)
        self.assertEqual(budget.stage.call_count, 2)
        payload = budget.stage.call_args_list[0].args[2]
        self.assertNotIn("target_scene_count", payload["conditions"])
        self.assertEqual(payload["source"]["story"], self.source["story"])

    def test_remaining_issue_gets_only_one_more_attempt_and_rejects_regression(self):
        budget = Mock()
        budget.stage.side_effect = [(self.shorter, {}), (self.review(1), {}),
                                    (self.candidate, {}), (self.review(0), {})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            self.seed(v.ROOT)
            record = v.process(4, budget, "test-key")
        self.assertEqual(budget.stage.call_count, 4)
        self.assertEqual(record["selected_attempt"], 1)
        self.assertEqual(record["candidate"], self.shorter.model_dump())

    def test_unknown_failure_keeps_previous_candidate_without_resend(self):
        budget = Mock()
        budget.stage.side_effect = RuntimeError("과금 미확인")
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            self.seed(v.ROOT)
            record = v.process(4, budget, "test-key")
        self.assertEqual(record["correction_status"], "failed")
        self.assertEqual(record["candidate"], self.candidate.model_dump())
        self.assertEqual(budget.stage.call_count, 1)

    def test_export_removes_fixed_count_from_input_and_keeps_cumulative_budget(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(v, "ROOT", Path(directory)):
            self.seed(v.ROOT, self.review())
            summary = v.export([self.source], {4: "train"}, b.Budget(v.ROOT, 78), 22)
            row = json.loads((v.ROOT / "train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(row["messages"][1]["content"]), {"moral": self.candidate.moral})
            self.assertEqual(row["messages"][0]["content"], v.SYSTEM_PROMPT)
            self.assertEqual(summary["cumulative_v2_v3_upper_usd"], 22)
            self.assertEqual(summary["corrections_targeted"], 1)


if __name__ == "__main__":
    unittest.main()
