import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import review_dataset_v4 as v
class FocusReviewTests(unittest.TestCase):
    def review(self, age=2, moral=2, integrity=2, hold=None):
        def grade(score):
            return {"score":score,"scene_refs":[1],"reason":"실제 본문에 근거한 판단"}
        return v.FocusReview(age_appropriateness=grade(age),moral_expression=grade(moral),
            adaptation_integrity=grade(integrity),sensitive_topics=[],publication_hold=hold or [],
            revision_instructions=[] if min(age,moral,integrity)==2 else ["해당 표현을 비노골적으로 고친다."])

    def record(self, number=144):
        old=v.load(v.BASE/"work"/f"{number:04d}"/"record.json")
        return {"source":old["source"],"source_split":old["source_split"],"reference_moral":old["reference_moral"],
            "candidate":old["candidate"],"baseline_candidate_digest":a.digest(old["candidate"]),
            "editorial_adaptations":[],"calls":{},"audit_status":"pending","human_review":"pending"}

    def test_three_preedits_preserve_baseline_and_remove_known_harmful_expressions(self):
        for n in (144,99,235):
            original=self.record(n)["candidate"]
            before=copy.deepcopy(original)
            revised,notes=v.preedit(n,b.Candidate.model_validate(original))
            v.v3.validate(revised)
            self.assertEqual(original,before)
            self.assertTrue(notes)
            prose=revised.model_dump_json()
            if n==144:
                self.assertNotIn("주머니가 터",prose)
                self.assertNotIn("당해도 싸",prose)
                self.assertIn("죽어 가",prose)
            elif n==99:
                self.assertNotIn("푸줏간",prose)
                self.assertNotIn("피를 흘",prose)
                self.assertIn("다치",prose)
            else:
                self.assertNotIn("절뚝",prose)
                self.assertNotIn("주름진 피부",prose)
                self.assertIn("무엇을 배웠",prose)

    def test_verdict_and_scene_reference_boundaries(self):
        self.assertEqual(v.verdict(self.review()),"pass")
        self.assertEqual(v.verdict(self.review(age=1)),"revise")
        self.assertEqual(v.verdict(self.review(moral=0)),"hold")
        self.assertEqual(v.verdict(self.review(hold=["핵심 편집 판단 필요"])),"hold")
        candidate=b.Candidate.model_validate(self.record()["candidate"])
        bad=self.review();bad.age_appropriateness.scene_refs=[999]
        with self.assertRaises(ValueError):
            v.validate_review(bad,candidate)

    def test_pass_costs_one_review_without_rewriting(self):
        budget=Mock();budget.stage.return_value=(self.review(),{})
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory,patch.object(v,"ROOT",Path(directory)):
            a.save_json(v.ROOT/"work/0144/record.json",self.record())
            result=v.process(144,budget,"test-key")
        self.assertEqual(budget.stage.call_count,1)
        self.assertEqual(result["focus_verdict"],"pass")

    def test_local_issue_gets_one_revision_and_recheck_but_not_regression(self):
        budget=Mock()
        candidate=b.Candidate.model_validate(self.record()["candidate"])
        budget.stage.side_effect=[(self.review(age=1),{}),(candidate,{}),(self.review(integrity=1),{})]
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory,patch.object(v,"ROOT",Path(directory)):
            a.save_json(v.ROOT/"work/0144/record.json",self.record())
            result=v.process(144,budget,"test-key")
        self.assertEqual(budget.stage.call_count,3)
        self.assertFalse(result["focus_revision_adopted"])
        self.assertEqual(result["focus_review"]["age_appropriateness"]["score"],1)

    def test_hold_and_unknown_requests_do_not_trigger_more_calls(self):
        for response in (self.review(moral=0),RuntimeError("과금 미확인")):
            budget=Mock()
            budget.stage.side_effect=response if isinstance(response,Exception) else [(response,{})]
            with tempfile.TemporaryDirectory(dir=a.ROOT) as directory,patch.object(v,"ROOT",Path(directory)):
                a.save_json(v.ROOT/"work/0144/record.json",self.record())
                result=v.process(144,budget,"test-key")
            self.assertEqual(budget.stage.call_count,1)
            self.assertIn(result.get("focus_verdict",result["audit_status"]),("hold","failed"))

    def test_export_filters_unreviewed_and_minor_issues_from_training(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory,patch.object(v,"ROOT",Path(directory)):
            good=self.record(144)
            good.update(audit_status="completed",focus_verdict="pass",focus_review=self.review().model_dump())
            bad=self.record(99)
            bad.update(audit_status="completed",focus_verdict="revise",focus_review=self.review(age=1).model_dump())
            a.save_json(v.ROOT/"work/0144/record.json",good)
            a.save_json(v.ROOT/"work/0099/record.json",bad)
            summary=v.export(b.Budget(v.ROOT,69),31)
            rows=v.load(v.ROOT/"dataset.json")
            self.assertEqual([r["source_id"] for r in rows],[144])
            self.assertEqual(summary["revise"],1)
            self.assertEqual(summary["cumulative_upper_usd"],31)
            for split in ("train","validation","test"):
                for line in (v.ROOT/f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    self.assertEqual(json.loads(line)["messages"][0]["content"],v.SYSTEM_PROMPT)


if __name__=="__main__":
    unittest.main()
