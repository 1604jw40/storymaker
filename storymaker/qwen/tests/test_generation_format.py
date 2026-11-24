import unittest
import json
from pathlib import Path
from unittest.mock import patch

from storymaker.qwen.code import generate_story as generation
class GenerationFormatTests(unittest.TestCase):
    def test_empty_input_fails_before_gpu_access_and_scene_references_are_checked(self):
        with patch("storymaker.qwen.code.sft_v5_train.gpu_ready") as gpu:
            with self.assertRaises(ValueError):
                generation.generate(" ", Path("missing-adapter"))
            gpu.assert_not_called()
        story = {
            "title": "새 이야기",
            "characters": [
                {
                    "character_id": "mouse",
                    "name": "생쥐",
                    "goal": "친구를 도와요.",
                    "traits": "작고 끈기 있어요.",
                }
            ],
            "scenes": [
                {
                    "scene_no": 1,
                    "character_ids": ["mouse"],
                    "setting": "숲, 아침",
                    "text_mode": "wordless",
                    "text": "",
                    "visual_note": "생쥐가 바라봐요.",
                }
            ],
        }
        self.assertEqual(generation.parse_story(json.dumps(story))[1], [])

        scene = story["scenes"][0]
        for scene_no in (True, 1.0, "1"):
            with self.subTest(scene_no=scene_no):
                scene["scene_no"] = scene_no
                self.assertEqual(generation.parse_story(json.dumps(story))[1], ["invalid_scene_number_1"])
        scene["scene_no"] = 1
        for mode in ([], {}, None):
            with self.subTest(text_mode=mode):
                scene["text_mode"] = mode
                self.assertEqual(generation.parse_story(json.dumps(story))[1], ["invalid_text_mode_1"])
        scene["text_mode"] = "wordless"

        story["scenes"][0]["character_ids"] = ["unknown"]
        self.assertIn(
            "invalid_character_reference_1",
            generation.parse_story(json.dumps(story))[1],
        )

    def test_moral_only_input_and_json_failures(self):
        with self.assertRaises(ValueError):
            generation.messages_for(" ")
        messages = generation.messages_for("작은 도움도 소중해요.")
        self.assertEqual(messages[1]["content"], '{"moral": "작은 도움도 소중해요."}')
        self.assertEqual(generation.parse_story("not json")[1], ["invalid_json"])
        self.assertEqual(generation.parse_story("[]")[1], ["invalid_top_level_fields"])
        self.assertEqual(
            generation.parse_story('{"title":"a","characters":[],"scenes":[]}')[1],
            ["empty_scenes"],
        )


if __name__ == "__main__":
    unittest.main()
