import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker.qwen.code.legacy import sft_v5_prepare as p
spec=importlib.util.spec_from_file_location('transition',p.ROOT/'tools/transition_sft_epoch10.py')
transition=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transition)


class TransitionTests(unittest.TestCase):
    def test_process_output_preserves_unicode_and_rejects_missing_output(self):
        rows=[{'ProcessId':10,'ParentProcessId':1,'CommandLine':str(p.ROOT/'storymaker.qwen.code.legacy.sft_v5_train.py')}]
        result=subprocess.CompletedProcess([],0,stdout=json.dumps(rows,ensure_ascii=False).encode('utf-8'))
        with patch.object(transition.subprocess,'run',return_value=result):
            self.assertEqual(transition.process_rows(),rows)
        result.stdout=None
        with patch.object(transition.subprocess,'run',return_value=result):
            with self.assertRaises(RuntimeError):transition.process_rows()

    def test_process_ownership_rejects_another_launch(self):
        script=str((p.ROOT/'storymaker.qwen.code.legacy.sft_v5_train.py').resolve())
        parent={'ProcessId':10,'ParentProcessId':1,'CommandLine':script}
        child={'ProcessId':20,'ParentProcessId':10,'CommandLine':script+' --fit-task x'}
        self.assertEqual(transition.owned_processes([parent,child],10),[parent,child])
        other={'ProcessId':30,'ParentProcessId':1,'CommandLine':script}
        with self.assertRaises(RuntimeError):transition.owned_processes([parent,child,other],10)
        self.assertEqual(transition.owned_processes([child],10,[parent]),[child])

    def test_checkpoint_copy_is_verified_and_never_overwrites_mismatch(self):
        with tempfile.TemporaryDirectory(dir=p.RUN) as directory:
            base=Path(directory)/'baseline';target=Path(directory)/'new'
            source=base/'fold/checkpoint-245/optimizer.pt';source.parent.mkdir(parents=True)
            source.write_bytes(b'optimizer-and-rng-fixture')
            destination=target/'fold/checkpoint-245/optimizer.pt'
            manifest={}
            with patch.object(transition,'BASE',base),patch.object(transition,'TARGET',target):
                transition.copy_verified(source,destination,manifest)
                self.assertEqual(source.read_bytes(),destination.read_bytes())
                transition.copy_verified(source,destination,manifest)
                destination.write_bytes(b'changed-by-another-run')
                with self.assertRaises(RuntimeError):transition.copy_verified(source,destination,manifest)
                self.assertEqual(destination.read_bytes(),b'changed-by-another-run')


if __name__=='__main__':unittest.main()
