import copy
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna

from storymaker.qwen.code import hpo_sft as h
from storymaker import project as p
from storymaker.qwen.code import settings
from storymaker.qwen.code import sft_v5_train as training
from storymaker.qwen.code import watch_sft as w
def curve(value, epoch=5):
    return {"eval_loss_by_epoch": [value + abs(e - epoch) * .001 for e in range(1, 11)]}


class HPOTests(unittest.TestCase):
    def config(self):
        return {**settings.hyperparameters(), "phase": "hpo", "fixed_learning_rate": .0002,
                "hpo_candidates": copy.deepcopy(h.CANDIDATES), "baseline_hyperparameters": h.BASELINE,
                "study_name": "test_hpo", "selected_lr_trial": 3, "dataset_sha256": {},
                "model": "unused", "model_revision": "fixed", "python": "approved-python"}

    def test_only_approved_overrides_reach_training_arguments_without_mutating_config(self):
        config = self.config()
        before = copy.deepcopy(config)
        actual = training.fit_config(config, h.CANDIDATES[1]["hyperparameters"])
        args = settings.training_arguments(actual, Path("unused"), .0002, 10, -1, True)
        self.assertEqual(args["weight_decay"], .1)
        self.assertEqual(actual["lora"]["dropout"], .05)
        self.assertEqual(training.fit_config(config, h.CANDIDATES[0]["hyperparameters"])["lora"]["dropout"], .1)
        self.assertEqual(config, before)
        for invalid in ({"lora_dropout": .2, "weight_decay": .01}, {"learning_rate": .001},
                        {"lora_dropout": float("nan"), "weight_decay": .01}):
            with self.assertRaises(ValueError):
                training.fit_config(config, invalid)
        with self.assertRaises(ValueError):
            training.fit_config(settings.hyperparameters(), h.BASELINE)

    def test_common_epoch_is_not_average_of_independent_fold_minima(self):
        a = {"eval_loss_by_epoch": [1, 2] + [3] * 8}
        b = {"eval_loss_by_epoch": [2, 1] + [3] * 8}
        score = h.common_epoch([a, b])
        self.assertEqual((score["epoch"], score["mean_validation_ce"]), (1, 1.5))
        with self.assertRaises(ValueError):
            h.common_epoch([{"eval_loss_by_epoch": [1] * 9}])
        with self.assertRaises(ValueError):
            h.common_epoch([{"eval_loss_by_epoch": [float("nan")] * 10}])

    def test_screening_requires_improvement_and_chooses_only_best_candidate(self):
        self.assertIsNone(h.screen_choice(curve(1.2), {"a": curve(1.2), "b": curve(1.3)}))
        self.assertEqual(h.screen_choice(curve(1.2), {"a": curve(1.1), "b": curve(1.15)}), "a")

    def test_full_flow_reuses_baseline_confirms_only_promising_and_refits_all_development(self):
        for improvement, confirmation_value, expected in ((False, 1.1, "baseline"),
                                                          (True, 1.15, "dropout_010"),
                                                          (True, 1.25, "baseline")):
            with self.subTest(improvement=improvement, expected=expected), tempfile.TemporaryDirectory(dir=p.ROOT / "outputs") as directory:
                run = Path(directory)
                config = self.config()
                p.save(run / "config.json", config)
                p.save(run / "status.json", {"state": "prepared"})
                records = {str(i): {} for i in range(243)}
                ids = sorted(records)
                folds = []
                for n in range(5):
                    val = ids[n::5]
                    folds.append({"fold": n + 1, "train_ids": [i for i in ids if i not in val], "validation_ids": val})
                p.save(run / "folds.json", folds)
                p.save(run / "encoded_development.json", [])
                baseline = [curve(1.2)] * 5
                study = optuna.create_study(study_name=config["study_name"], storage="sqlite:///" + (run / "optuna.sqlite3").as_posix())
                for candidate in h.CANDIDATES:
                    study.enqueue_trial({"candidate": candidate["name"]})
                calls = []

                def worker(folder, train_ids, val_ids, rate, epochs, tag, max_steps, cfg, hyperparameters):
                    calls.append((tag, train_ids, val_ids, hyperparameters, epochs))
                    if tag.startswith("screen/"):
                        value = (1.1 if "dropout_010" in tag else 1.15) if improvement else 1.3
                    else:
                        value = confirmation_value
                    adapter = folder / "adapter"
                    adapter.mkdir(parents=True)
                    (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
                    result = {**curve(value), "source_ids": train_ids, "train_count": len(train_ids),
                              "validation_count": len(val_ids), "learning_rate": rate, "epochs": epochs,
                              "hyperparameters": hyperparameters, "run_config_sha256": h.digest(run / "config.json"),
                              "optimizer_steps": math.ceil(len(train_ids) / 4) * epochs,
                              "test_used_for_training": False, "adapter": str(adapter)}
                    if not val_ids:
                        result["eval_loss_by_epoch"] = []
                    p.save(folder / "result.json", result)
                    return result

                with patch.object(p, "RUN", run), patch.object(h, "context", return_value=(config, records, folds, baseline)), \
                     patch.object(training, "gpu_ready", return_value={"ready": True}), patch.object(h.time, "sleep"), \
                     patch.object(h, "resume_plan"), patch.object(training, "worker_fit", side_effect=worker), \
                     patch("optuna.load_study", return_value=study):
                    h.run()
                    count = len(calls)
                    h.run()
                    self.assertEqual(len(calls), count)
                selected = p.read(run / "selection.json")
                self.assertEqual(selected["candidate"], expected)
                self.assertFalse(selected["test_used_for_selection"])
                self.assertEqual(len(calls), 7 if improvement else 3)
                self.assertEqual(calls[-1][0], "final_refit")
                self.assertEqual(calls[-1][1], ids)
                self.assertEqual(calls[-1][2], [])
                self.assertEqual(calls[-1][3], selected["hyperparameters"])
                self.assertEqual(calls[-1][4], selected["epoch"])
                self.assertEqual(p.read(run / "status.json")["state"], "complete")
                study._storage._backend.engine.dispose()

    def test_resume_uses_same_hpo_trial_and_rejects_changed_active_settings(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT / "outputs") as directory:
            run = Path(directory)
            config = self.config()
            dbpath = run / "optuna.sqlite3"
            study = optuna.create_study(study_name=config["study_name"], storage="sqlite:///" + dbpath.as_posix())
            names = [c["name"] for c in h.CANDIDATES]
            for name in names:
                study.enqueue_trial({"candidate": name})
            trial = study.ask()
            trial.suggest_categorical("candidate", names)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            active = {"phase": "screen", "candidate": names[0], "fold": 1, "epochs": 10,
                      "stage": f"screen/{names[0]}/fold_1", "trial_number": 0,
                      "hyperparameters": h.CANDIDATES[0]["hyperparameters"]}
            p.save(run / "active_task.json", active)
            with patch.object(w, "check_checkpoint", return_value={"checkpoint": None, "initial_step": 0, "sha256": {}}), \
                 patch("optuna.load_study", return_value=study):
                plan = h.resume_plan(run, config)
                self.assertEqual(plan["trial_state"], "FAIL")
                w.recover_failed_trial(dbpath, plan)
                with sqlite3.connect(dbpath) as db:
                    self.assertEqual(db.execute("SELECT state FROM trials ORDER BY number").fetchall(), [("RUNNING",), ("WAITING",)])
                db.close()
                p.save(run / "active_task.json", {**active, "hyperparameters": h.BASELINE})
                with self.assertRaises(ValueError):
                    h.resume_plan(run, config)
            study._storage._backend.engine.dispose()

    def test_cached_result_with_wrong_parameters_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "RUN", Path(directory)):
            p.save(p.RUN / "config.json", self.config())
            with self.assertRaises(ValueError):
                h.validate_result({"hyperparameters": h.BASELINE}, self.config(), ["a"], ["b"], h.CANDIDATES[0]["hyperparameters"], 10)

    def test_grid_covers_four_combinations_without_enabling_combination_in_legacy_run(self):
        actual = {(h.BASELINE["lora_dropout"], h.BASELINE["weight_decay"])}
        actual.update((c["hyperparameters"]["lora_dropout"], c["hyperparameters"]["weight_decay"]) for c in h.GRID_CANDIDATES)
        self.assertEqual(actual, {(d, w) for d in (.05, .1) for w in (.01, .1)})
        config = self.config()
        with self.assertRaises(ValueError):
            training.fit_config(config, h.GRID_CANDIDATES[-1]["hyperparameters"])
        config.update(screening_design="2x2_grid", hpo_candidates=copy.deepcopy(h.GRID_CANDIDATES))
        self.assertEqual(len(h.approved_candidates(config)), 3)
        fit_config = training.fit_config(config, h.GRID_CANDIDATES[-1]["hyperparameters"])
        self.assertEqual((fit_config["lora"]["dropout"], fit_config["weight_decay"]), (.1, .1))

    def test_grid_resume_keeps_two_completed_imports_and_restores_only_combination_trial(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT / "outputs") as directory:
            run = Path(directory)
            config = self.config()
            config.update(screening_design="2x2_grid", hpo_candidates=copy.deepcopy(h.GRID_CANDIDATES))
            dbpath = run / "optuna.sqlite3"
            study = optuna.create_study(study_name=config["study_name"], storage="sqlite:///" + dbpath.as_posix())
            names = [c["name"] for c in h.GRID_CANDIDATES]
            for name in names:
                study.enqueue_trial({"candidate": name})
            for index in range(3):
                trial = study.ask()
                trial.suggest_categorical("candidate", names)
                if index < 2:
                    study.tell(trial, 1.3)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
            candidate = h.GRID_CANDIDATES[-1]
            p.save(run / "active_task.json", {"phase": "screen", "candidate": candidate["name"], "fold": 1,
                   "stage": f"screen/{candidate['name']}/fold_1", "trial_number": 2,
                   "epochs": 10, "hyperparameters": candidate["hyperparameters"]})
            with patch.object(w, "check_checkpoint", return_value={"checkpoint": None, "initial_step": 0, "sha256": {}}), \
                 patch("optuna.load_study", return_value=study):
                plan = h.resume_plan(run, config)
                self.assertEqual((plan["trial_number"], plan["trial_state"]), (2, "FAIL"))
                w.recover_failed_trial(dbpath, plan)
                with sqlite3.connect(dbpath) as db:
                    self.assertEqual(db.execute("SELECT state FROM trials ORDER BY number").fetchall(),
                                     [("COMPLETE",), ("COMPLETE",), ("RUNNING",)])
                db.close()
            study._storage._backend.engine.dispose()

    def test_grid_reuses_two_results_with_original_config_hash_and_confirms_combination(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT / "outputs") as directory:
            root = Path(directory)
            source, run = root / "source", root / "grid"
            previous, config = self.config(), self.config()
            config.update(screening_design="2x2_grid", hpo_candidates=copy.deepcopy(h.GRID_CANDIDATES), reused_screen_results={})
            p.save(source / "config.json", previous)
            ids = list(range(243))
            records = {i: {} for i in ids}
            folds = [{"train_ids": [i for i in ids if i % 5 != n], "validation_ids": ids[n::5]} for n in range(5)]

            def result(folder, train_ids, val_ids, params, value, config_path, tag, epochs=10):
                adapter = folder / "adapter"
                adapter.mkdir(parents=True, exist_ok=True)
                (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
                return {**curve(value), "tag": tag, "source_ids": train_ids, "train_count": len(train_ids),
                        "validation_count": len(val_ids), "learning_rate": .0002, "epochs": epochs,
                        "hyperparameters": params, "run_config_sha256": h.digest(config_path),
                        "optimizer_steps": math.ceil(len(train_ids) / 4) * epochs,
                        "test_used_for_training": False, "adapter": str(adapter)}

            for c in h.CANDIDATES:
                tag = f"screen/{c['name']}/fold_1"
                path = source / tag / "result.json"
                p.save(path, result(path.parent, folds[0]["train_ids"], folds[0]["validation_ids"], c["hyperparameters"], 1.3, source / "config.json", tag))
                config["reused_screen_results"][tag] = {"result": str(path), "config": str(source / "config.json")}
            p.save(run / "config.json", config)
            p.save(run / "status.json", {"state": "prepared"})
            p.save(run / "folds.json", folds)
            p.save(run / "encoded_development.json", [])
            study = optuna.create_study(study_name=config["study_name"], storage="sqlite:///" + (run / "optuna.sqlite3").as_posix())
            for c in h.GRID_CANDIDATES:
                study.enqueue_trial({"candidate": c["name"]})
            calls = []

            def worker(folder, train_ids, val_ids, rate, epochs, tag, max_steps, cfg, hyperparameters):
                calls.append(tag)
                value = result(folder, train_ids, val_ids, hyperparameters, 1.1, run / "config.json", tag, epochs)
                if not val_ids:
                    value["eval_loss_by_epoch"] = []
                p.save(folder / "result.json", value)
                return value

            with patch.object(p, "RUN", run), patch.object(h, "context", return_value=(config, records, folds, [curve(1.2)] * 5)), \
                 patch.object(training, "gpu_ready", return_value={"ready": True}), patch.object(h.time, "sleep"), \
                 patch.object(h, "resume_plan"), patch.object(training, "worker_fit", side_effect=worker), \
                 patch("optuna.load_study", return_value=study):
                h.run()
                h.run()
            self.assertEqual(calls, ["screen/dropout_010_weight_decay_010/fold_1"] +
                             [f"confirmation/dropout_010_weight_decay_010/fold_{i}" for i in range(2, 6)] + ["final_refit"])
            self.assertEqual(p.read(run / "selection.json")["candidate"], "dropout_010_weight_decay_010")
            self.assertEqual([t.state.name for t in study.trials], ["COMPLETE"] * 3)
            self.assertTrue(all("reused_from" in t.user_attrs for t in study.trials[:2]))
            for c in h.CANDIDATES:
                tag = f"screen/{c['name']}/fold_1"
                self.assertEqual(p.read(run / tag / "result.json"), p.read(source / tag / "result.json"))
                tampered = copy.deepcopy(p.read(run / tag / "result.json"))
                tampered["eval_loss_by_epoch"][0] -= .1
                with patch.object(p, "RUN", run), self.assertRaises(ValueError):
                    h.validate_result(tampered, config, folds[0]["train_ids"], folds[0]["validation_ids"], c["hyperparameters"], 10)
            study._storage._backend.engine.dispose()


if __name__ == "__main__":
    unittest.main()
