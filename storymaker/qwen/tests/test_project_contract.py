import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker import project as p
from storymaker.qwen.code import sft_v5_train as training
class ProjectContractTests(unittest.TestCase):
    def test_notebook_outputs_match_original_tokens_and_data(self):
        provenance = p.read(p.ROOT / "data/source/provenance.json")
        config = p.read(p.RUN / "config.json")
        self.assertEqual(config["epochs_per_fold"], 10)
        self.assertEqual(
            config["dataset_sha256"], provenance["baseline_processed_sha256"]
        )
        digest = hashlib.sha256(
            (p.RUN / "encoded_development.json").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, provenance["baseline_encoded_sha256"])
        summary = p.read(p.ROOT / "outputs/eda/token_summary.json")
        self.assertEqual(summary["counts"]["v5"], 304)
        self.assertEqual(summary["counts"]["scenes"], 4267)
        self.assertEqual(summary["truncated_examples"], 0)
        for name in ["01_preprocessing_eda_tokenizer.ipynb", "02_loss_analysis.ipynb"]:
            notebook = p.read(p.ROOT / "notebooks" / name)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNotNone(cell["execution_count"])
                    self.assertFalse(
                        any(
                            output["output_type"] == "error"
                            for output in cell["outputs"]
                        )
                    )

    def test_run_directory_is_used_and_outside_outputs_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=p.RUN) as directory:
            with patch.object(p, "RUN", p.RUN), patch.object(training, "run") as run:
                with patch.object(sys, "argv", ["train", "--run-dir", directory]):
                    training.main()
                self.assertEqual(p.RUN, Path(directory).resolve())
                run.assert_called_once_with()
        with patch.object(training, "run") as run:
            with patch.object(sys, "argv", ["train", "--run-dir", str(p.ROOT.parent)]):
                with self.assertRaises(ValueError):
                    training.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
