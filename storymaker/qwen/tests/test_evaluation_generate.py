import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker.qwen.code import evaluation_generate as evaluation
from storymaker import project as p
class PairedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "outputs/run"
        self.output = self.run / "evaluation"
        self.context = {
            "config": {"python": "unused", "model_revision": "frozen-revision"},
            "adapter": self.run / "final_refit/adapter", "files": [],
            "encoded": [{"source_id": i, "input_ids": [1] * i} for i in range(1, 244)],
            "books": [{"source_id": i, "source_split": "development" if i <= 243 else "test",
                       "input": {"moral": f"moral {i}"}, "output": {"title": "Must not enter generation"}}
                      for i in range(1, 305)],
        }
        self.manifest = evaluation.manifest_for(self.context)

    def result_for(self, job):
        return {"moral": job["moral"], "model_variant": job["variant"],
                "adapter": str(self.context["adapter"]) if job["variant"] == "sft" else None,
                "decoding": evaluation.generation.DECODING, "seed": 42,
                "base_model_revision": "frozen-revision", "raw_text": "broken JSON",
                "format_problems": ["invalid_json"], "story": None}

    def test_training_must_finish_before_host_checks_or_generation(self):
        p.save(self.run / "status.json", {"state": "training"})
        with patch.object(evaluation.w, "processes") as host, patch.object(evaluation.generation, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "has not completed"):
                evaluation.completed_context(self.run)
            host.assert_not_called()
            generate.assert_not_called()

    def test_pilot_uses_development_and_test_jobs_contain_only_moral(self):
        pilot = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")
        test = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "test")
        self.assertEqual(len(pilot), 6)
        self.assertEqual(len(test), 122)
        self.assertTrue({j["source_id"] for j in pilot}.isdisjoint({j["source_id"] for j in test}))
        self.assertTrue(all(j["source_id"] <= 243 for j in pilot))
        self.assertTrue(all(j["source_id"] > 243 for j in test))
        self.assertTrue(all(j["moral"] == f"moral {j['source_id']}" for j in pilot + test))
        self.assertTrue(all("title" not in str(j) and "story" not in j for j in pilot + test))

    def test_test_is_blocked_before_pilot_and_frozen_protocol(self):
        with patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.subprocess, "run") as launch:
            with self.assertRaises(FileNotFoundError):
                evaluation.batch(self.run, self.output, "test")
            launch.assert_not_called()

    def test_bad_quality_outputs_are_preserved_without_regeneration(self):
        jobs = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")
        for job in jobs:
            p.save(job["output"], {"state": "completed", "job": job, "generation": self.result_for(job)})
        with patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.subprocess, "run") as launch:
            evaluation.batch(self.run, self.output, "pilot")
            launch.assert_not_called()
        self.assertEqual(p.read(self.output / "generation_status.json")["completed_outputs"], 6)
        self.assertEqual(p.read(jobs[0]["output"])["generation"]["raw_text"], "broken JSON")

    def test_cached_settings_mismatch_stops_instead_of_mixing(self):
        job = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")[0]
        result = self.result_for(job)
        result["seed"] = 43
        p.save(job["output"], {"state": "completed", "job": job, "generation": result})
        with patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.subprocess, "run") as launch:
            with self.assertRaisesRegex(AssertionError, "settings mismatch"):
                evaluation.batch(self.run, self.output, "pilot")
            launch.assert_not_called()

    def test_technical_failure_is_not_automatically_retried(self):
        job = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")[0]
        p.save(Path(job["output"]).with_suffix(".failure.json"), {"returncode": 1})
        with patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.subprocess, "run") as launch:
            with self.assertRaisesRegex(RuntimeError, "Unresolved technical"):
                evaluation.batch(self.run, self.output, "pilot")
            launch.assert_not_called()

    def test_job_cannot_change_moral_before_gpu_call(self):
        job = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")[0]
        p.save(self.output / "generation_manifest.json", self.manifest)
        job["moral"] = "different moral"
        path = self.output / "job.json"
        p.save(path, job)
        with patch.object(p, "ROOT", Path(self.temp.name)), patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.generation, "generate") as generate:
            with self.assertRaises(AssertionError):
                evaluation.worker(self.run, path)
            generate.assert_not_called()

    def test_worker_uses_base_or_final_adapter_and_keeps_failed_json(self):
        jobs = evaluation.jobs_for(self.run, self.output, self.context, self.manifest, "pilot")
        p.save(self.output / "generation_manifest.json", self.manifest)
        for job in jobs[:2]:
            path = self.output / "job.json"
            p.save(path, job)
            with patch.object(p, "ROOT", Path(self.temp.name)), patch.object(p, "RUN", self.run), patch.object(evaluation, "completed_context", return_value=self.context), patch.object(evaluation.generation, "generate", return_value=self.result_for(job)) as generate:
                evaluation.worker(self.run, path)
                generate.assert_called_once_with(job["moral"], self.context["adapter"] if job["variant"] == "sft" else None)
            self.assertEqual(p.read(job["output"])["generation"]["format_problems"], ["invalid_json"])


if __name__ == "__main__":
    unittest.main()
