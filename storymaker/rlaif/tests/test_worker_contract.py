"""Real package commands and imported API responses, without GPU or paid calls."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from storymaker import project as p
from storymaker.rlaif.code import rlaif_budget as budget


class WorkerChecks(unittest.TestCase):
    def test_real_phase_modules_accept_help_without_starting_a_job(self):
        for worker in ('rlaif_sequence.py', 'rlaif_train.py'):
            command = p.worker_command(worker, '--help', python=sys.executable)
            self.assertEqual(command[3], '-m')
            result = subprocess.run(command, cwd=p.ROOT.parent, capture_output=True,
                text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_budget_reuse_checks_the_preserved_source_and_exact_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'project/rlaif/results/current'
            old=root/'history'; destination=old/'reward'
            destination.mkdir(parents=True); run.mkdir(parents=True)
            old_ledger=old/'budget_ledger.json'; p.save(old_ledger, {'calls':[]})
            config={'shared_budget_ledger':str(run/'budget_ledger.json'),
                    'source_budget_ledger':str(old_ledger),'source_budget_root':str(old)}
            p.save(run/'config.json',config);p.save(run/'budget_ledger.json',{'calls':[]})
            p.save(run/'budget_import.json',{'source':str(old_ledger),
                'source_sha256':hashlib.sha256(old_ledger.read_bytes()).hexdigest()})
            body={'model':'unit-model','messages':[]}
            raw={'id':'unit-response','model':'unit-model','usage':{'prompt_tokens':4}}
            p.save(destination/'request.json',body)
            p.save(destination/'response.json',{'http_status':200,'raw_body':json.dumps(raw)})
            record={'state':'completed','request_sha256':'fixed','destination':str(destination),
                    'response_id':raw['id'],'usage':raw['usage']}
            with patch.object(p,'ROOT',root/'project'):
                self.assertEqual(budget.ledger_path(run),run/'budget_ledger.json')
                self.assertEqual(budget.completed_response(record,run/'budget_ledger.json',body,'fixed')[1],raw)
                with self.assertRaises(AssertionError):
                    budget.completed_response(record,run/'budget_ledger.json',dict(body,messages=['changed']),'fixed')
                p.save(old_ledger,{'calls':['altered']})
                with self.assertRaises(AssertionError):
                    budget.completed_response(record,run/'budget_ledger.json',body,'fixed')


if __name__=='__main__':unittest.main()
