import tempfile
import unittest
from pathlib import Path

from storymaker.data_augmentation.code import augment as a
from storymaker.data_augmentation.code import build_dataset as b
from storymaker.data_augmentation.code import repair_remaining_v5 as v
class RemainingSevenTests(unittest.TestCase):
    def test_edits_preserve_baseline_and_named_outcomes(self):
        required = {52:"죽고", 61:"아파요", 67:"형편이 달라질", 87:"잡아먹었어요",
                    153:"죽었어요", 245:"죽어 가던", 290:"목숨을 잃었어요"}
        for n in v.TARGETS:
            before = v.v4.load(v.BASE/"work"/f"{n:04d}"/"record.json")["candidate"]
            candidate = b.Candidate.model_validate(before)
            after, notes, highlights = v.edit(n, candidate)
            self.assertEqual(candidate.model_dump(), before)
            v.v4.v3.validate(after)
            self.assertIn(required[n], after.model_dump_json())
            self.assertTrue(notes and highlights)
            for h in highlights:
                if h["after_scene_no"]:
                    self.assertEqual(h["after_text"], after.output.scenes[h["after_scene_no"]-1].text)
            if n == 61:
                self.assertEqual(len(after.output.scenes), 4)
                self.assertNotIn("꽉 잡아 보렴", after.model_dump_json())
                self.assertIn("먼저 어른에게", after.model_dump_json())
                self.assertTrue(any("의도적으로" in x for x in notes))
            if n == 290:
                self.assertNotIn("너를 죽이겠다", after.model_dump_json())
        with self.assertRaises(ValueError):
            v.edit(1,candidate)

    def test_versioned_export_uses_only_requested_destination(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT) as directory:
            root = Path(directory)
            old = v.v4.load(v.BASE/"work/0001/record.json")
            old.update(baseline_candidate_digest=a.digest(old["candidate"]), calls={})
            a.save_json(root/"work/0001/record.json", old)
            summary = v.v4.export(b.Budget(root,61),39,root=root,dataset_version="v5")
            self.assertEqual(summary["training_candidates"],1)
            self.assertEqual(v.v4.load(root/"dataset.json")[0]["id"],"aesop_0001_scenes_sft_v5")


if __name__ == "__main__":
    unittest.main()
