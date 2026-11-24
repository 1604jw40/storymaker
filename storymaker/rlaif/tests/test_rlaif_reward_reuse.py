"""An inspected response can be reparsed in place without another paid request."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from storymaker import project as p
from storymaker.rlaif.code import rlaif_reward_api as api


class RewardReuse(unittest.TestCase):
    def test_run_allowance_extension_keeps_config_and_shared_limit(self):
        from storymaker.rlaif.code.rlaif_budget import run_allowance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = {'cycle_API_management_cap_usd': 10}; p.save(root/'config.json', config)
            original = (root/'config.json').read_bytes(); ledger = {'budget_usd': 60}
            self.assertEqual(run_allowance(root, config, ledger), 10)
            extension = {'previous_cap_usd': 10, 'cap_usd': '28.3522360', 'config_sha256': hashlib.sha256(original).hexdigest()}
            ledger['run_allowances'] = {str(root.resolve()): extension}
            self.assertEqual(str(run_allowance(root, config, ledger)), '28.3522360')
            for key, value in [('config_sha256', 'stale'), ('previous_cap_usd', 9), ('cap_usd', -1), ('cap_usd', 61), ('cap_usd', 'Infinity')]:
                bad = dict(extension, **{key: value}); ledger['run_allowances'][str(root.resolve())] = bad
                with self.assertRaises(AssertionError):
                    run_allowance(root, config, ledger)
            self.assertEqual((root/'config.json').read_bytes(), original)

    def test_same_or_other_response_file_is_reused_without_a_request(self):
        for same in (True,False):
            with self.subTest(same_file=same),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source.json';p.save(source,{'raw_text':'test'})
                destination=root/'result';destination.mkdir();origin=(destination if same else root)/'response.json'
                origin.write_bytes(b'{"preserved":true}')
                body={'model':'test'};signature=hashlib.sha256(json.dumps(body,ensure_ascii=False).encode()).hexdigest()
                ledger=root/'ledger.json';p.save(ledger,{'calls':[{'state':'completed','request_sha256':signature,'cost_upper_usd':'.01'}]})
                context={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'corpus_sha256':'corpus'}
                with patch.object(api,'OUT',root),patch.object(api,'budget_path',return_value=ledger),\
                     patch.object(api,'body_for',return_value=body),patch.object(api.rag,'sha',return_value='corpus'),\
                     patch.object(api,'completed_response',return_value=(origin,{'model':'test'})),\
                     patch.object(api,'parse',return_value={'selected_score':1}),\
                     patch.object(api.requests,'post',side_effect=AssertionError('Must not resend')):
                    result=api.score(source,destination,'unused',context)
                self.assertEqual(result['selected_score'],1)
                self.assertEqual((destination/'response.json').read_bytes(),b'{"preserved":true}')
                self.assertTrue((destination/'reward.json').is_file())


if __name__=='__main__':unittest.main()
