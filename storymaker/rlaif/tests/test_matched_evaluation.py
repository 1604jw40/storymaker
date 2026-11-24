"""Committed starting point and exact-condition reuse of original validation outputs."""
import copy
import unittest
from pathlib import Path
from storymaker import project as p
from storymaker.rlaif.code import rlaif_evaluate as evaluation


class MatchedChecks(unittest.TestCase):
    def test_only_retained_committed_policy_is_accepted(self):
        config=p.read(evaluation.OUT/'config.json')
        tag,checkpoint=evaluation.policy_context(Path(config['policy_checkpoint']))
        self.assertEqual(tag,Path(config['policy_checkpoint']).parent.name)
        self.assertEqual(checkpoint,Path(config['policy_checkpoint']).resolve())
        with self.assertRaises(AssertionError):evaluation.policy_context(evaluation.OUT/'environment/discarded/policy')

    def test_cached_original_requires_weight_prompt_seed_and_settings_identity(self):
        settings={'seed':42,'max_new_tokens':4096,'temperature':.7}
        previous={'tag':'original','settings':settings,'system_prompt_sha256':'prompt'}
        manifest=dict(previous,reused_original_manifest_sha256='source')
        prompt={'id':'9','moral':'선택에는 결과가 따른다.'}
        record={'identity':{'tag':'original','id':'9','seed':42,'policy_sha256':'weights','manifest_sha256':'source'},
            'moral':prompt['moral'],'prompt_ids':[1,2],'quality_resampled':False,'teacher_forced':{'ce':1.},'generated_token_ids':[3]}
        evaluation.verify_reused_original(record,previous,manifest,prompt,[1,2],'weights')
        for field,value in [('seed',43),('policy_sha256','changed'),('manifest_sha256','changed')]:
            bad=copy.deepcopy(record);bad['identity'][field]=value
            with self.assertRaises(AssertionError):evaluation.verify_reused_original(bad,previous,manifest,prompt,[1,2],'weights')
        with self.assertRaises(AssertionError):evaluation.verify_reused_original(record,previous,manifest,prompt,[1,99],'weights')
        bad=copy.deepcopy(manifest);bad['settings']['temperature']=.9
        with self.assertRaises(AssertionError):evaluation.verify_reused_original(record,previous,bad,prompt,[1,2],'weights')


if __name__=='__main__':unittest.main()
