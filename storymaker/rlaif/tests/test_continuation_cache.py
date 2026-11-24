import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker import project as p
from storymaker.rlaif.code import rlaif_evaluate as evaluation
from storymaker.rlaif.code import rlaif_reward_pipeline as reward


class ContinuationReuse(unittest.TestCase):
    def test_committed_policy_reuse_rejects_changed_model_or_conditions(self):
        previous={'tag':'update_0004','settings':{'seed':42,'max_new_tokens':4096},
                  'system_prompt_sha256':'prompt','checkpoint_manifest_sha256':'checkpoint'}
        manifest=dict(previous,reused_policy_manifest_sha256='origin')
        prompt={'id':'9','moral':'서로 도와요.'}
        record={'identity':{'tag':'update_0004','id':'9','seed':42,'policy_sha256':'weights','manifest_sha256':'origin'},
                'moral':prompt['moral'],'prompt_ids':[1,2],'quality_resampled':False,
                'teacher_forced':{'ce':1.},'generated_token_ids':[3]}
        evaluation.verify_reused_output(record,previous,manifest,prompt,[1,2],'weights')
        for field,value in [('checkpoint_manifest_sha256','other'),('tag','update_0005'),('system_prompt_sha256','other')]:
            with self.subTest(field=field):
                bad=dict(manifest,**{field:value})
                with self.assertRaises(AssertionError):evaluation.verify_reused_output(record,previous,bad,prompt,[1,2],'weights')
        with self.assertRaises(AssertionError):evaluation.verify_reused_output(record,previous,manifest,prompt,[1,2],'other')
        with self.assertRaises(AssertionError):evaluation.verify_reused_original(record,previous,manifest,prompt,[1,2],'weights')

    def test_policy_copy_preserves_origin_and_rejects_byte_or_checkpoint_changes(self):
        for problem in (None,'bytes','checkpoint','missing_provenance'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                old=Path(directory)/'old';new=Path(directory)/'new'
                manifest={'tag':'update_0004','settings':{'seed':42},'system_prompt_sha256':'prompt',
                          'checkpoint_manifest_sha256':'model'}
                p.save(old/'manifest.json',manifest)
                p.save(new/'manifest.json',dict(manifest,checkpoint_manifest_sha256='changed' if problem=='checkpoint' else 'model',
                       reused_policy_manifest_sha256=evaluation.sha(old/'manifest.json')))
                item={'identity':{'tag':'update_0004','manifest_sha256':evaluation.sha(old/'manifest.json')},'quality_resampled':False,'raw_text':'원문'}
                for folder in (old,new):p.save(folder/'9.json',item)
                if problem!='missing_provenance':p.save(new/'provenance/9.json',{
                    'source':str(old/'9.json'),'sha256':evaluation.sha(old/'9.json'),'new_generation':False,'identity_preserved':True})
                if problem=='bytes':p.save(new/'9.json',dict(item,raw_text='변경'))
                if problem:
                    with self.assertRaises((AssertionError,FileNotFoundError)):evaluation.output_origin(new/'9.json')
                else:self.assertEqual(evaluation.output_origin(new/'9.json'),(old/'9.json',old/'manifest.json'))

    def test_protocol_replay_count_must_match_hashed_passed_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config={'reward_protocol_kind':reward.PROTOCOL,'api_model':'gpt-4.1-2025-04-14'}
            p.save(root/'config.json',config);proof=root/'proof.json'
            p.save(proof,{'passed':True,'saved_response_replays':81})
            frozen={'protocol_kind':reward.PROTOCOL,'model':config['api_model'],'config_sha256':evaluation.sha(root/'config.json'),
                    'runtime_code_sha256':{},'acceptance_sha256':{str(proof):evaluation.sha(proof)},'saved_response_replays':81,'criteria_count':11}
            with patch.object(reward.api,'OUT',root):
                p.save(root/'reward_protocol_frozen.json',frozen);reward.accepted_protocol(config)
                p.save(root/'reward_protocol_frozen.json',dict(frozen,saved_response_replays=82))
                with self.assertRaises(AssertionError):reward.accepted_protocol(config)
                p.save(root/'reward_protocol_frozen.json',frozen);p.save(proof,{'passed':False,'saved_response_replays':81})
                with self.assertRaises(AssertionError):reward.accepted_protocol(config)
