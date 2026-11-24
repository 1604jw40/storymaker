import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import refine
class AugmentationTests(unittest.TestCase):
    def test_transient_save_lock_retries_without_losing_old_or_new_data(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"old": true}', encoding="utf-8")
            replace = Path.replace
            attempts = []
            def briefly_locked(source, target):
                attempts.append(1)
                if len(attempts) < 3:
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
                    raise PermissionError("file sharing lock")
                return replace(source, target)
            with patch.object(Path, "replace", autospec=True, side_effect=briefly_locked), patch.object(a.time, "sleep"):
                a.save_json(path, {"new": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": True})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_persistent_save_lock_preserves_target_and_recovery_file(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("locked")) as replace, patch.object(a.time, "sleep"):
                with self.assertRaises(PermissionError):
                    a.save_json(path, {"new": True})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(json.loads(path.with_suffix(".json.tmp").read_text(encoding="utf-8")), {"new": True})

    def plan(self, scene_count=None):
        return a.Plan.model_validate({
            "explicit_moral": None, "target_moral": "함께 도울 수 있다.",
            "moral_evidence": "생쥐가 줄을 끊었다.", "source_concerns": [],
            "key_events": [{"event_id": 1, "summary": "생쥐가 구한다.", "evidence": "생쥐가 줄을 끊었다."}],
            "suitability": "proceed", "suitability_reason": "행동으로 표현할 수 있다.",
            "title": "도움", "characters": [{"character_id": "mouse", "source_name": "생쥐", "goal": "구하기", "traits": "용감함"}],
            "scenes": [{"scene_no": i, "character_ids": ["mouse"], "setting": "빈터",
                        "visual_focus": f"밧줄의 {i}번째 위치를 살피는 생쥐",
                        "reader_discovery": f"{i}번째 발견", "page_turn_role": "다음 행동을 기대한다.",
                        "text_mode": "read_aloud", "source_event_ids": [1]}
                       for i in range(1, (scene_count or a.CONDITIONS["target_scene_count"]) + 1)],
            "added_details": []})

    def test_source_evidence_and_scene_links(self):
        plan = self.plan()
        a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        plan.key_events[0].evidence = "없는 사건"
        with self.assertRaisesRegex(ValueError, "근거 인용"):
            a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        plan = self.plan()
        plan.scenes[0].source_event_ids = [99]
        with self.assertRaisesRegex(ValueError, "사건 참조"):
            a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})

    def test_story_cannot_drop_scenes(self):
        story = a.Story(title="도움", scenes=[a.WrittenScene(scene_no=1, text_mode="read_aloud", text="글", visual_note="그림")], adaptation_notes=[])
        with self.assertRaisesRegex(ValueError, "장면 번호"):
            a.validate_story(story, self.plan())

    def test_review_cannot_refer_to_nonexistent_scene(self):
        assessment = {"score": 2, "scene_refs": [99], "reason": "근거"}
        review = a.Review.model_validate({**{name: assessment for name in list(a.Review.model_fields)[:6]},
                                         "blocking_issues": [], "revision_instructions": []})
        story = a.Story(title="도움", scenes=[a.WrittenScene(scene_no=1, text_mode="read_aloud", text="글", visual_note="그림")], adaptation_notes=[])
        with self.assertRaisesRegex(ValueError, "장면 참조"):
            a.validate_review(review, story)

    def test_cache_prevents_duplicate_paid_call(self):
        result = {"title": "도움", "scenes": [{"scene_no": 1, "text_mode": "read_aloud", "text": "글", "visual_note": "그림"}], "adaptation_notes": []}
        raw = {"status": "completed", "id": "test", "model": a.MODEL,
               "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(result)}]}],
               "usage": {"input_tokens": 100, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 0}}}
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(a.requests, "post") as post:
            self.assertTrue(Path(directory).resolve().is_relative_to(a.ROOT.resolve()))
            post.return_value.ok = True
            post.return_value.json.return_value = raw
            path = Path(directory) / "call.json"
            first, _ = a.api_stage(path, "prompt", {}, a.Story, 1000, "test-key")
            second, _ = a.api_stage(path, "prompt", {}, a.Story, 1000, "test-key")
            self.assertEqual(first, second)
            self.assertEqual(post.call_count, 1)
            with self.assertRaises(ValueError):
                a.api_stage(path, "changed", {}, a.Story, 1000, "test-key")
            self.assertEqual(post.call_count, 1)
            raw["status"] = "incomplete"
            other = Path(directory) / "incomplete.json"
            with self.assertRaisesRegex(ValueError, "미완료"):
                a.api_stage(other, "prompt", {}, a.Story, 1000, "test-key")
            self.assertTrue(other.exists())

    def test_cost_includes_output_and_cached_input(self):
        cost = a.cost_estimates({"input_tokens": 1000, "output_tokens": 500,
                                 "input_tokens_details": {"cached_tokens": 200}})
        self.assertAlmostEqual(cost["standard_usd"], 0.00662)
        self.assertAlmostEqual(cost["cache_write_upper_usd"], 0.00702)
        written = a.cost_estimates({"input_tokens": 1000, "output_tokens": 500,
                                    "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 600}})
        self.assertAlmostEqual(written["standard_usd"], 0.00692)
        self.assertEqual(written["standard_usd"], written["cache_write_upper_usd"])
        with self.assertRaises(ValueError):
            a.cost_estimates({"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 30}})

    def test_timeout_is_logged_without_retry(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory, patch.object(a.requests, "post") as post:
            self.assertTrue(Path(directory).resolve().is_relative_to(a.ROOT.resolve()))
            post.side_effect = a.requests.Timeout("test timeout")
            path = Path(directory) / "calls" / "call.json"
            with self.assertRaisesRegex(RuntimeError, "과금 미확인"):
                a.api_stage(path, "prompt", {}, a.Story, 1000, "test-key")
            self.assertEqual(post.call_count, 1)
            attempts = json.loads((Path(directory) / "unconfirmed_calls.json").read_text(encoding="utf-8"))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["charge_status"], "unknown")

    def test_one_source_event_can_span_multiple_distinct_scenes(self):
        plan = self.plan()
        a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        self.assertEqual(len(plan.key_events), 1)
        plan.scenes[1].visual_focus = plan.scenes[0].visual_focus
        plan.scenes[1].reader_discovery = plan.scenes[0].reader_discovery
        with self.assertRaisesRegex(ValueError, "동일한 시각 장면"):
            a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})

    def test_scene_target_and_held_work(self):
        plan = self.plan()
        plan.scenes.pop()
        with self.assertRaisesRegex(ValueError, "목표 장면 수"):
            a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        plan.suitability = "hold"
        plan.scenes = []
        a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})

    def test_explicit_scene_target_does_not_change_default(self):
        plan = self.plan()
        expanded = []
        for i in range(1, 25):
            scene = plan.scenes[(i - 1) % len(plan.scenes)].model_copy(deep=True)
            scene.scene_no = i
            scene.visual_focus = f"{i}번째 시각적 순간"
            scene.reader_discovery = f"{i}번째 발견"
            expanded.append(scene)
        plan.scenes = expanded
        a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."}, scene_count=24)
        with self.assertRaisesRegex(ValueError, "목표 장면 수"):
            a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        self.assertEqual(a.CONDITIONS["target_scene_count"], 12)

    def test_wordless_scene_is_intentional_and_still_needs_visual_content(self):
        plan = self.plan()
        plan.scenes[0].text_mode = "wordless"
        plan.scenes[0].character_ids = []
        plan.scenes[0].visual_focus = "빈터에 놓인 끊어진 밧줄"
        plan.scenes[0].reader_discovery = "밧줄이 끊어졌다는 사실을 발견한다."
        a.validate_plan(plan, {"story": "생쥐가 줄을 끊었다."})
        story = a.Story(title="도움", scenes=[a.WrittenScene(
            scene_no=s.scene_no, text_mode=s.text_mode,
            text="" if s.text_mode == "wordless" else "생쥐가 살폈어요.",
            visual_note="빈터에 놓인 밧줄의 잘린 끝.") for s in plan.scenes], adaptation_notes=[])
        a.validate_story(story, plan)
        story.scenes[1].text = ""
        with self.assertRaisesRegex(ValueError, "낭독 장면"):
            a.validate_story(story, plan)
        story.scenes[1].text = "생쥐가 살폈어요."
        story.scenes[0].text = "글이 들어갔어요."
        with self.assertRaisesRegex(ValueError, "글 없는 장면"):
            a.validate_story(story, plan)
        story.scenes[0].text = ""
        story.scenes[0].visual_note = ""
        with self.assertRaisesRegex(ValueError, "빈 문자열"):
            a.validate_story(story, plan)

    def test_scene_report_renders_wordless_scene_and_escapes_source(self):
        plan = self.plan()
        plan.scenes[0].text_mode = "wordless"
        story = a.Story(title="도움", scenes=[a.WrittenScene(
            scene_no=s.scene_no, text_mode=s.text_mode,
            text="" if s.text_mode == "wordless" else "생쥐가 살폈어요.",
            visual_note="빈터의 밧줄.") for s in plan.scenes], adaptation_notes=[])
        record = {"source": {"number": 1, "title": "<script>", "story": "생쥐가 줄을 끊었다."},
                  "source_split": "train", "status": "completed",
                  "plan": plan.model_dump(), "story": story.model_dump()}
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(a.ROOT.resolve()))
            a.write_reports(Path(directory), [record])
            report = (Path(directory) / "review.html").read_text(encoding="utf-8")
            self.assertIn("글 없는 장면", report)
            self.assertIn("그림의 중심 순간", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            exported = json.loads((Path(directory) / "0001.json").read_text(encoding="utf-8"))
            self.assertEqual(exported, record)

    def test_revision_is_bounded_and_preserves_source(self):
        plan = self.plan(24)
        story = a.Story(title="도움", scenes=[a.WrittenScene(
            scene_no=s.scene_no, text_mode=s.text_mode, text="생쥐가 살폈어요.", visual_note="밧줄을 보는 생쥐.")
            for s in plan.scenes], adaptation_notes=[])
        grade = {"score": 2, "scene_refs": [1], "reason": "장면 근거"}
        review = a.Review.model_validate({**{name: grade for name in list(a.Review.model_fields)[:6]},
                                         "blocking_issues": [], "revision_instructions": []})
        record = {"source": {"number": 1, "story": "생쥐가 줄을 끊었다."},
                  "conditions": {**a.CONDITIONS, "target_scene_count": 24}, "plan": plan.model_dump(), "story": story.model_dump(),
                  "review": review.model_dump(), "api_recommendation": "revise",
                  "calls": {"review": {"response_id": "original-review"}}}
        original_source = dict(record["source"])
        shorter = story.model_copy(update={"scenes": story.scenes[:20]})
        revision = refine.RevisedScenes(scenes=plan.scenes[:20], added_details=[], story=shorter)
        with patch.object(a, "api_stage", side_effect=[(revision, {}), (review, {})]) as api:
            self.assertTrue(refine.revise_record(record, Path("unused-test-output"), "test-key", scene_count=20))
            self.assertFalse(refine.revise_record(record, Path("unused-test-output"), "test-key", scene_count=20))
            self.assertEqual(api.call_count, 2)
            self.assertEqual(api.call_args_list[0].args[2]["conditions"]["target_scene_count"], 20)
        self.assertEqual(record["source"], original_source)
        self.assertEqual(record["revision_passes"], 1)
        self.assertEqual(record["api_recommendation"], "pass")
        self.assertEqual(record["conditions"]["target_scene_count"], 20)
        self.assertEqual(record["revision_history"][0]["initial_scene_count"], 24)

    def test_revision_rejects_wrong_language(self):
        revision = refine.RevisedScenes(scenes=self.plan().scenes, added_details=[],
            story=a.Story(title="狮子和老鼠", scenes=[], adaptation_notes=[]))
        with self.assertRaisesRegex(ValueError, "한국어"):
            refine.validate_korean_revision(revision)


if __name__ == "__main__":
    unittest.main()
