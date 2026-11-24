import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState, create_trial

from storymaker import project as p
from storymaker.qwen.code import settings
from storymaker.qwen.code import sft_v5_train as training
class LRQueueTests(unittest.TestCase):
    def setUp(self):
        self.config = settings.hyperparameters()
        self.distribution = FloatDistribution(2e-5, 3e-4, log=True)

    def completed(self, rate, value=1.2):
        return create_trial(
            params={"learning_rate": rate},
            distributions={"learning_rate": self.distribution},
            value=value,
            user_attrs={"selected_epoch": 6},
        )

    def test_replacement_preserves_baseline_and_is_idempotent(self):
        study = optuna.create_study()
        study.add_trial(self.completed(8e-5))
        study.enqueue_trial({"learning_rate": 8.037944071180388e-5})
        duplicate = study.ask()
        duplicate.suggest_float("learning_rate", 2e-5, 3e-4, log=True)
        with self.assertRaises(ValueError):
            training.prepare_lr_trials(study, self.config)
        study.tell(duplicate, state=TrialState.PRUNED)
        training.prepare_lr_trials(study, self.config)
        training.prepare_lr_trials(study, self.config)
        self.assertEqual(len(study.trials), 4)
        self.assertEqual(study.trials[0].value, 1.2)
        for rate in (5e-5, 2e-4):
            trial = study.ask()
            self.assertEqual(
                trial.suggest_float("learning_rate", 2e-5, 3e-4, log=True), rate
            )
            study.tell(trial, 1.1)
        training.prepare_lr_trials(study, self.config)
        self.assertEqual(len(study.trials), 4)
        self.assertEqual(sum(t.state == TrialState.COMPLETE for t in study.trials), 3)

    def test_bad_candidate_lists_are_rejected_before_enqueue(self):
        for values in (
            [8e-5, 8e-5, 2e-4],
            [8e-5],
            [8e-5, 1e-7, 2e-4],
            [8e-5, float("nan"), 2e-4],
        ):
            config = {**self.config, "candidate_learning_rates": values}
            study = optuna.create_study()
            with self.assertRaises(ValueError):
                training.prepare_lr_trials(study, config)
            self.assertEqual(len(study.trials), 0)

    def test_completed_search_defers_refit_without_loading_model(self):
        config = {
            **self.config,
            "not_before": "1970-01-01T00:00:00+00:00",
            "model": "unused",
            "study_name": "cpu-test",
        }
        for defer in (True, False):
            config["defer_final_refit"] = defer
            with tempfile.TemporaryDirectory(dir=p.RUN) as directory:
                run = Path(directory)
                p.save(run / "config.json", config)
                p.save(run / "isolated_smoke/result.json", {"optimizer_steps": 2})
                p.save(
                    run / "final_refit/result.json", {"adapter": "cached-test-adapter"}
                )
                study = optuna.create_study(study_name="cpu-test")
                for rate in config["candidate_learning_rates"]:
                    study.add_trial(self.completed(rate))
                records = {1: {"input_ids": [1, 2]}}
                with patch.object(p, "RUN", run), patch.object(
                    training,
                    "inputs",
                    return_value=(copy.deepcopy(config), records, []),
                ), patch.object(
                    training, "gpu_ready", return_value={"ready": True}
                ), patch.object(
                    training.time, "sleep"
                ), patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=SimpleNamespace(pad_token_id=0),
                ), patch(
                    "transformers.AutoModelForImageTextToText.from_pretrained"
                ) as model, patch.object(
                    training, "worker_fit"
                ) as worker, patch("optuna.create_study", return_value=study):
                    training.run()
                model.assert_not_called()
                worker.assert_not_called()
                state = p.read(run / "status.json")
                self.assertEqual(
                    state["state"], "lr_search_complete" if defer else "complete"
                )
                self.assertEqual(p.read(run / "selection.json")["epoch"], 6)


if __name__ == "__main__":
    unittest.main()
