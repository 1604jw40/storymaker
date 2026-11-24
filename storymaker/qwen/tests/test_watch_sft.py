import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from storymaker import project as p
from storymaker.qwen.code import watch_sft as w
class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve()
        (self.run / "watchdog").mkdir()
        self.status = {"state": "training", "updated_at_utc": w.now(),
                       "stage": "trial_0002/fold_3", "optimizer_step": 60, "logs": {"learning_rate": 5e-5}}
        p.save(self.run / "status.json", self.status)
        p.save(self.run / "config.json", {"python": "approved-python"})
        p.save(self.run / "launcher.json", {"initial_started_at_utc": "original"})
        p.save(self.run / "watchdog/protected_sha256.json", {})
        (self.run / "stderr.log").write_text("")
        with closing(sqlite3.connect(self.run / "optuna.sqlite3")) as db, db:
            db.execute("CREATE TABLE trials(trial_id INTEGER, number INTEGER, state TEXT, datetime_complete TEXT)")
            db.executemany("INSERT INTO trials VALUES(?,?,?,?)",
                           [(1, 0, "COMPLETE", "done"), (2, 1, "PRUNED", "excluded"),
                            (3, 2, "RUNNING", None), (4, 3, "WAITING", None)])
        self.memory = {"stderr_offset": 0}
        self.plan = {"trial_id": 3, "trial_number": 2, "trial_state": "RUNNING",
                     "stage": "trial_0002/fold_3", "initial_step": 49,
                     "learning_rate": 5e-5, "checkpoint": "checkpoint-49", "sha256": {}}
        for name, value in (
            ("stop_reason", None), ("processes", []), ("lock_held", False),
            ("gpu_sample", {"temperature_c": 60, "free_mib": 15000}),
            ("readiness", {"ready": True}), ("resume_plan", self.plan),
        ):
            mocked = patch.object(w, name, return_value=value)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        self.launch = patch.object(w.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)).start()
        self.addCleanup(patch.stopall)

    def test_hold_completion_and_changed_run_override_recovery(self):
        pointer = {"run_dir": str(self.run)}
        self.assertEqual(w.gate({"enabled": False}, {}, {}, pointer, self.run), "disabled_by_operator")
        self.assertEqual(w.gate({"enabled": True}, {}, {"explicit_user_resume_required": True}, pointer, self.run), "user_hold")
        self.assertEqual(w.gate({"enabled": True}, {"state": "lr_search_complete"}, {}, pointer, self.run), "phase_complete")
        self.assertEqual(w.gate({"enabled": True}, {}, {}, {"run_dir": str(self.run / "other")}, self.run), "active_run_changed")

    def test_live_process_or_locked_run_never_restarts(self):
        for rows, held in (([{"ProcessId": 1}], False), ([], True)):
            self.processes.return_value, self.lock_held.return_value = rows, held
            w.tick(self.run, self.memory)
        self.launch.assert_not_called()
        self.readiness.assert_not_called()

    def test_confirmed_exit_restarts_once_then_requires_real_step_progress(self):
        w.tick(self.run, self.memory)
        self.launch.assert_not_called()
        w.tick(self.run, self.memory)
        self.launch.assert_not_called()
        w.tick(self.run, self.memory)
        self.launch.assert_called_once()
        self.assertTrue((Path(self.memory["pending_resume"]["receipt_dir"]) / "optuna_before.sqlite3").exists())
        self.assertEqual(p.read(self.run / "launcher.json")["initial_started_at_utc"], "original")
        self.processes.return_value = [{"ProcessId": 1234}]
        w.tick(self.run, self.memory)
        self.assertIn("pending_resume", self.memory)
        p.save(self.run / "status.json", {**self.status, "updated_at_utc": w.now(), "optimizer_step": 65})
        w.tick(self.run, self.memory)
        self.assertNotIn("pending_resume", self.memory)
        self.assertIn("verified_at_utc", self.memory["last_verified_resume"])
        self.launch.assert_called_once()

    def test_observe_only_cannot_launch(self):
        self.memory["missing_polls"] = 10
        w.tick(self.run, self.memory, observe_only=True)
        self.launch.assert_not_called()

    def test_old_result_does_not_prove_resume_until_new_stage_steps(self):
        for _ in range(3):
            w.tick(self.run, self.memory)
        p.save(self.run / self.plan["stage"] / "result.json", {"optimizer_steps": 490})
        self.processes.return_value = [{"ProcessId": 1234}]
        new_status = {**self.status, "updated_at_utc": w.now(), "stage": "next_fold", "optimizer_step": 0}
        p.save(self.run / "status.json", new_status)
        w.tick(self.run, self.memory)
        self.assertIn("pending_resume", self.memory)
        p.save(self.run / "status.json", {**new_status, "optimizer_step": 5, "learning_rate": 9e-5})
        w.tick(self.run, self.memory)
        self.assertIn("pending_resume", self.memory)
        p.save(self.run / "status.json", {**new_status, "optimizer_step": 5})
        w.tick(self.run, self.memory)
        self.assertIn("last_verified_resume", self.memory)

    def test_hpo_restart_uses_hpo_runner_and_never_lr_queue(self):
        p.save(self.run / "config.json", {"python": "approved-python", "phase": "hpo"})
        for _ in range(3):
            w.tick(self.run, self.memory)
        self.launch.assert_called_once()
        self.assertEqual(self.launch.call_args.args[0][3:5], ["-m", "storymaker.qwen.code.hpo_sft"])
        self.assertEqual(self.launch.call_args.kwargs["cwd"], p.ROOT.parent)

    def test_unknown_error_and_restart_loop_require_inspection(self):
        self.memory["missing_polls"] = 2
        p.save(self.run / "status.json", {**self.status, "state": "failed", "error": "invalid data"})
        w.tick(self.run, self.memory)
        self.assertEqual(self.memory["phase"], "needs_attention")
        self.launch.assert_not_called()
        self.assertTrue(w.can_retry({"state": "failed"}, "RuntimeError: CUDA out of memory"))
        p.save(self.run / "status.json", self.status)
        self.memory.pop("needs_attention")
        self.memory["restart_times"] = [w.now()] * 3
        w.tick(self.run, self.memory)
        self.assertIn("Three restarts", self.memory["needs_attention"])

    def test_changed_protected_file_and_busy_gpu_never_launch(self):
        self.memory["missing_polls"] = 2
        self.readiness.return_value = {"ready": False}
        w.tick(self.run, self.memory)
        self.assertEqual(self.memory["phase"], "waiting_gpu")
        p.save(self.run / "watchdog/protected_sha256.json", {str(self.run / "config.json"): "wrong"})
        w.tick(self.run, self.memory)
        self.assertIn("changed", self.memory["needs_attention"])
        self.launch.assert_not_called()

    def test_only_same_failed_trial_is_restored_and_excluded_trial_preserved(self):
        path = self.run / "optuna.sqlite3"
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE trials SET state='FAIL',datetime_complete='crash' WHERE trial_id=3")
        w.recover_failed_trial(path, {**self.plan, "trial_state": "FAIL"})
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT number,state,datetime_complete FROM trials ORDER BY number").fetchall(),
                             [(0, "COMPLETE", "done"), (1, "PRUNED", "excluded"), (2, "RUNNING", None), (3, "WAITING", None)])
        with self.assertRaises(ValueError):
            w.recover_failed_trial(path, {**self.plan, "trial_state": "FAIL"})

    def test_incomplete_newest_checkpoint_is_not_silently_overwritten(self):
        folder = self.run / "fold"
        (folder / "checkpoint-49").mkdir(parents=True)
        (folder / "checkpoint-98").mkdir()
        with self.assertRaisesRegex(ValueError, "checkpoint-98"):
            w.check_checkpoint(folder, 5e-5)


if __name__ == "__main__":
    unittest.main()
