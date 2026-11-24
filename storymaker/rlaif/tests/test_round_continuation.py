"""Follow committed model ancestry across rounds; no GPU or paid calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker import project as p
from storymaker.rlaif.code import rlaif_evaluate as evaluation
from storymaker.rlaif.code import rlaif_probability_probe as probability
from storymaker.rlaif.code.rlaif_environment import preserve_batch_config
from storymaker.rlaif.code.rlaif_training_report import sha


def runs_fixture(folder):
    runs=[folder/str(i) for i in range(3)]
    for index,run in enumerate(runs):
        config={'train_ids':[23,139],'validation_ids':[205],'excluded_test_ids':[304],
            'batch_ids':[23],'batch_index':0,'logical_batch_size':1,'max_new_tokens':4096}
        if index:
            previous=runs[index-1];parent_tag=f'update_{index:04d}'
            config.update(source_previous_run=str(previous),source_previous_config_sha256=sha(previous/'config.json'),
                policy_checkpoint=str(previous/'checkpoints'/parent_tag/'policy'),
                value_checkpoint=str(previous/'checkpoints'/parent_tag/'value'))
        p.save(run/'config.json',config)
        tag=f'update_{index+1:04d}'
        for role in ('policy','value'):p.save(run/'checkpoints'/tag/role/'manifest.json',{'role':role})
        p.save(run/'batch_0000/joint_update_verified.json',{
            'policy':str(run/'checkpoints'/tag/'policy'),'value':str(run/'checkpoints'/tag/'value'),
            'committed_policy_updates':1,'committed_value_updates':1})
    return runs


def rollout(run,number,tokens):
    path=run/'batch_0000/rollouts'/f'{number}.json'
    p.save(path,{'identity':{'source_id':number,'config_sha256':sha(run/'config.json')},
        'prompt_ids':[10,20],'generated_token_ids':[30]*tokens,'generated_tokens':tokens})
    return path


class ContinuationChecks(unittest.TestCase):
    def test_readonly_review_node_can_forward_exact_committed_ancestor(self):
        from storymaker.rlaif.code.rlaif_training_report import committed_batches
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory));review=Path(directory)/'review';child=Path(directory)/'child'
            for root,parent in ((review,runs[2]),(child,review)):
                config={**p.read(runs[2]/'config.json'),'source_previous_run':str(parent),
                    'source_previous_config_sha256':sha(parent/'config.json'),
                    'policy_checkpoint':str(runs[2]/'checkpoints/update_0003/policy'),
                    'value_checkpoint':str(runs[2]/'checkpoints/update_0003/value')}
                p.save(root/'config.json',config)
            self.assertEqual(set(committed_batches(child)),{'update_0001','update_0002','update_0003'})

    def test_forwarded_pair_must_be_committed_and_from_a_verified_ancestor(self):
        from storymaker.rlaif.code.rlaif_training_report import committed_batches
        for problem in ('outside','uncommitted','mismatched_roles'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                runs=runs_fixture(Path(directory));review=Path(directory)/'review';config=p.read(runs[2]/'config.json')
                config.update(source_previous_run=str(runs[2]),source_previous_config_sha256=sha(runs[2]/'config.json'),
                    policy_checkpoint=str(runs[2]/'checkpoints/update_0003/policy'),
                    value_checkpoint=str(runs[2]/'checkpoints/update_0003/value'))
                if problem=='outside':config['policy_checkpoint']=str(Path(directory)/'outside/update_0003/policy')
                elif problem=='uncommitted':config.update(policy_checkpoint=str(runs[2]/'checkpoints/update_9999/policy'),value_checkpoint=str(runs[2]/'checkpoints/update_9999/value'))
                else:config['value_checkpoint']=str(runs[1]/'checkpoints/update_0002/value')
                p.save(review/'config.json',config)
                with self.assertRaises(AssertionError):committed_batches(review)

    def test_original_output_keeps_its_exact_origin_through_two_reuse_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            folders=[Path(directory)/str(i) for i in range(3)]
            item=None
            for index,folder in enumerate(folders):
                manifest={'tag':'original','settings':{'seed':42},'system_prompt_sha256':'prompt',
                    'reused_original_manifest_sha256':sha(folders[index-1]/'manifest.json') if index else None}
                p.save(folder/'manifest.json',manifest)
                if item is None:item={'identity':{'id':'9','tag':'original','manifest_sha256':sha(folder/'manifest.json')},
                    'quality_resampled':False,'raw_text':'원본 출력'}
                p.save(folder/'9.json',item)
                if index:p.save(folder/'provenance/9.json',{'source':str(folders[index-1]/'9.json'),
                    'sha256':sha(folders[index-1]/'9.json'),'new_generation':False,'identity_preserved':True})
            origin,manifest=evaluation.original_origin(folders[2]/'9.json')
            self.assertEqual(origin,folders[0]/'9.json')
            self.assertEqual(manifest,folders[0]/'manifest.json')
            p.save(folders[0]/'9.json',dict(item,raw_text='변경한 출력'))
            with self.assertRaises(AssertionError):evaluation.original_origin(folders[2]/'9.json')

    def test_original_reuse_refuses_changed_settings_or_missing_byte_provenance(self):
        for problem in ('settings','missing_provenance','bad_hash'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                old=Path(directory)/'old';new=Path(directory)/'new'
                p.save(old/'manifest.json',{'tag':'original','settings':{'seed':42},'system_prompt_sha256':'prompt'})
                p.save(new/'manifest.json',{'tag':'original','settings':{'seed':43 if problem=='settings' else 42},
                    'system_prompt_sha256':'prompt','reused_original_manifest_sha256':sha(old/'manifest.json')})
                item={'identity':{'tag':'original','manifest_sha256':sha(old/'manifest.json')},'quality_resampled':False}
                for folder in (old,new):p.save(folder/'9.json',item)
                if problem!='missing_provenance':p.save(new/'provenance/9.json',{'source':str(old/'9.json'),
                    'new_generation':False,'identity_preserved':True,'sha256':'changed' if problem=='bad_hash' else sha(old/'9.json')})
                with self.assertRaises((AssertionError,FileNotFoundError)):evaluation.original_origin(new/'9.json')

    def test_cached_original_math_requires_exact_snapshot_and_unchanged_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'run/evaluation/validation/original'
            current=root/'current';current.mkdir()
            snippets={'rlaif_evaluate.py':'def teacher_forced(model):\n    return 1\n',
                'rlaif_train.py':'def load_model(config):\n    from storymaker.rlaif.code.rlaif_precision import enable\n    return config\n',
                'rlaif_precision.py':'constant = 1\n'}
            for name,text in snippets.items():
                (current/name).write_text(text,'utf-8')
                target=folder/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(text.replace('storymaker.rlaif.code.rlaif_precision','rlaif_precision'),'utf-8')
            p.save(folder/'manifest.json',{'code_sha256':{name:sha(folder/'code_snapshot'/name) for name in snippets}})
            with patch.object(p,'code_path',side_effect=lambda name:current/name):
                evaluation.verify_original_math(folder/'manifest.json')
                (current/'rlaif_evaluate.py').write_text('def teacher_forced(model):\n    return 2\n','utf-8')
                with self.assertRaises(AssertionError):evaluation.verify_original_math(folder/'manifest.json')
                (current/'rlaif_evaluate.py').write_text(snippets['rlaif_evaluate.py'],'utf-8')
                (folder/'code_snapshot/rlaif_evaluate.py').write_text('changed','utf-8')
                (current/'rlaif_evaluate.py').write_text('changed','utf-8')
                with self.assertRaises(AssertionError):evaluation.verify_original_math(folder/'manifest.json')

    def test_exact_batch_configuration_is_saved_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config={'batch_index':2,'batch_ids':list(range(8)),'logical_batch_size':64}
            p.save(root/'config.json',config)
            path=preserve_batch_config(root,config)
            self.assertEqual(path,root/'batch_0002/config.json')
            self.assertEqual(path.read_bytes(),(root/'config.json').read_bytes())
            self.assertEqual(preserve_batch_config(root,config),path)

    def test_changed_configuration_cannot_replace_a_saved_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p.save(root/'config.json',{'batch_index':0,'learning_rate':1e-5})
            path=preserve_batch_config(root,p.read(root/'config.json'));before=path.read_bytes()
            p.save(root/'config.json',{'batch_index':0,'learning_rate':1e-4})
            with self.assertRaises(AssertionError):preserve_batch_config(root,p.read(root/'config.json'))
            self.assertEqual(path.read_bytes(),before)

    def test_stale_or_damaged_configuration_is_preserved_for_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p.save(root/'config.json',{'batch_index':0})
            with self.assertRaises(AssertionError):preserve_batch_config(root,{'batch_index':1})
            self.assertFalse((root/'batch_0001').exists())
            path=root/'batch_0000/config.json';path.parent.mkdir();path.write_text('{broken','utf-8')
            with self.assertRaises(AssertionError):preserve_batch_config(root,{'batch_index':0})
            self.assertEqual(path.read_text('utf-8'),'{broken')

    def test_any_committed_ancestor_and_current_policy_are_selectable(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory))
            with patch.object(evaluation,'OUT',runs[2]):
                for index,run in enumerate(runs):
                    path=run/f'checkpoints/update_{index+1:04d}/policy'
                    self.assertEqual(evaluation.policy_context(path),(f'update_{index+1:04d}',path))
                self.assertEqual(evaluation.policy_context(None),('original',None))

    def test_fake_policy_and_tampered_parent_are_rejected(self):
        for problem in ('fake_policy','missing_manifest','parent_config','value_pair'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                runs=runs_fixture(Path(directory));path=runs[0]/'checkpoints/update_0001/policy'
                if problem=='fake_policy':path=Path(directory)/'uncommitted/checkpoints/update_0001/policy'
                elif problem=='missing_manifest':(path/'manifest.json').unlink()
                elif problem=='parent_config':p.save(runs[1]/'config.json',{'changed':True})
                else:
                    receipt=runs[2]/'batch_0000/joint_update_verified.json';value=p.read(receipt)
                    value['value']=str(runs[0]/'checkpoints/update_0001/value');p.save(receipt,value)
                with patch.object(evaluation,'OUT',runs[2]),self.assertRaises(AssertionError):
                    evaluation.policy_context(path)

    def test_probability_cases_use_train_rollouts_across_canonical_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory))
            first=rollout(runs[2],23,16);full=rollout(runs[0],139,4096)
            cases=probability.replay_cases(runs[2],p.read(runs[2]/'config.json'))
            self.assertEqual([row[3] for row in cases],[first,full])
            self.assertEqual([len(row[2]) for row in cases],[16,4096])
            self.assertEqual(cases[1][1],[10,20])

    def test_probability_readiness_can_precede_new_collection_without_fabricating_a_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory));first=rollout(runs[1],23,16);full=rollout(runs[0],139,4096)
            cases=probability.replay_cases(runs[2],p.read(runs[2]/'config.json'))
            self.assertEqual(cases[0][0],'preserved_first_training_rollout')
            self.assertEqual([row[3] for row in cases],[first,full])
            self.assertFalse((runs[2]/'batch_0000/rollouts').exists())

    def test_precollection_replay_refuses_an_uncommitted_configured_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory));rollout(runs[1],23,16);rollout(runs[0],139,4096)
            config=p.read(runs[2]/'config.json');config['policy_checkpoint']=str(runs[1]/'checkpoints/update_9999/policy')
            with self.assertRaises(AssertionError):probability.replay_cases(runs[2],config)

    def test_missing_or_held_out_or_changed_rollouts_do_not_pass(self):
        for problem in ('no_current','no_full','held_out','changed_config','wrong_count'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                runs=runs_fixture(Path(directory));config=p.read(runs[2]/'config.json')
                if problem!='no_current':rollout(runs[2],23,16)
                if problem!='no_full':
                    full=rollout(runs[0],139,4096)
                    value=p.read(full)
                    if problem=='held_out':value['identity']['source_id']=205
                    elif problem=='changed_config':value['identity']['config_sha256']='changed'
                    elif problem=='wrong_count':value['generated_tokens']=4095
                    p.save(full,value)
                with self.assertRaises(AssertionError):probability.replay_cases(runs[2],config)

    def test_batch_snapshot_is_verified_instead_of_overwritten_parent_config(self):
        with tempfile.TemporaryDirectory() as directory:
            runs=runs_fixture(Path(directory));full=rollout(runs[0],139,4096)
            p.save(runs[0]/'batch_0000/config.json',{'original_snapshot':True})
            value=p.read(full);value['identity']['config_sha256']=sha(runs[0]/'batch_0000/config.json');p.save(full,value)
            rollout(runs[2],23,16)
            self.assertEqual(probability.replay_cases(runs[2],p.read(runs[2]/'config.json'))[1][3],full)


if __name__=='__main__':unittest.main()
