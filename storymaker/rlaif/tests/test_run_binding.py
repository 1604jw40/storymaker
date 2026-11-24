"""Explicit child-run binding preserves the global pointer and project boundary."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker import project as p


def environment():
    source=Path(__file__).parents[1]/'code/rlaif_environment.py'
    spec=importlib.util.spec_from_file_location('isolated_run_binding',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RunBindingChecks(unittest.TestCase):
    def test_explicit_worker_run_does_not_change_the_global_active_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);active=root/'rlaif/results/active';readonly=root/'rlaif/results/readonly'
            pointer=root/'rlaif/configs/active_run.json';p.save(pointer,{'run_dir':str(active)})
            original=pointer.read_bytes()
            with patch.object(p,'ROOT',root),patch.dict('os.environ',{'STORYMAKER_RLAIF_RUN_DIR':str(readonly)}):
                self.assertEqual(environment().OUT,readonly.resolve())
            self.assertEqual(pointer.read_bytes(),original)

    def test_empty_override_keeps_the_existing_default_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);active=root/'rlaif/results/active'
            p.save(root/'rlaif/configs/active_run.json',{'run_dir':str(active)})
            with patch.object(p,'ROOT',root),patch.dict('os.environ',{'STORYMAKER_RLAIF_RUN_DIR':''}):
                self.assertEqual(environment().OUT,active.resolve())

    def test_explicit_binding_cannot_escape_the_project_results_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'project';outside=Path(directory)/'outside'
            with patch.object(p,'ROOT',root),patch.dict('os.environ',{'STORYMAKER_RLAIF_RUN_DIR':str(outside)}):
                with self.assertRaises(AssertionError):environment()


if __name__=='__main__':unittest.main()
