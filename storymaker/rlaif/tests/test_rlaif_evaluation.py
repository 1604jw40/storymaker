"""Evaluation split/failure invariants, with no GPU model loading or API requests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymaker import project as p
from storymaker.rlaif.code import rlaif_evaluate as e
from storymaker.rlaif.code import rlaif_evaluation_queue as queue
from storymaker.rlaif.code.rl_pilot_monitor import memory_alerts
from storymaker.rlaif.code.rlaif_evaluation_queue import verify_generation


class EvaluationChecks(unittest.TestCase):
    def test_queue_waits_on_live_validation_and_uses_current_shared_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'rlaif/results/current';folder=run/'evaluation/queue'
            folder.mkdir(parents=True)
            shared=root/'rlaif/results/shared/budget_ledger.json'
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
            write(shared,{'calls':[]})
            write(run/'config.json',{'python':'unused','policy_checkpoint':'saved/policy','shared_budget_ledger':str(shared)})
            write(root/'rlaif/configs/active_run.json',{'run_dir':str(run)})
            write(run/'control.json',{'enabled':True});write(folder/'control.json',{'enabled':True})
            write(run/'evaluation/evaluation_plan.json',{})
            write(run/'evaluation/validation/update_0001/status.json',{'stage':'generating','completed':9})
            for name in ['rlaif_evaluation_queue.py','rlaif_evaluate.py','rlaif_sequence.py','rlaif_train.py',
                         'rlaif_storage.py','rlaif_environment.py','generate_story.py','sft_v5_train.py','rlaif_precision.py','project.py']:
                (root/name).write_text('fixture',encoding='utf-8')
            def stop(_):write(folder/'control.json',{'enabled':False})
            with patch.object(queue,'OUT',run),patch.object(queue,'GPU_LOCK',root/'rlaif/results/generation.lock'), \
                 patch.object(p,'ROOT',root),patch.object(p,'code_path',side_effect=lambda name:root/name), \
                 patch.object(queue,'policy_context',return_value=('update_0001',Path('saved/policy'))), \
                 patch.object(queue,'active_workers',return_value=[{'ProcessId':11,'CommandLine':'python -m storymaker.rlaif.code.rlaif_evaluate'}]), \
                 patch.object(queue,'gpu_ready',return_value=(False,{})), \
                 patch.object(queue.time,'sleep',side_effect=stop),patch.object(queue.subprocess,'Popen') as launch:
                queue.main(execute=True)
            self.assertTrue((root/'rlaif/results/generation.lock').exists())
            self.assertTrue(shared.with_name('reward_api.lock').exists())
            self.assertFalse((run/'reward_api.lock').exists())
            self.assertEqual(p.read(folder/'state.json')['stage'],'stopped_by_control')
            launch.assert_not_called()

    def test_partial_or_failed_validation_is_never_implicitly_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            self.assertTrue(queue.may_launch(folder,[]))
            p.save(folder/'status.json',{'stage':'generating','completed':1})
            self.assertFalse(queue.may_launch(folder,[{'ProcessId':11}]))
            with self.assertRaises(RuntimeError):queue.may_launch(folder,[])
            p.save(folder/'failure.json',{'error':'CUDA'})
            with self.assertRaises(AssertionError):queue.may_launch(folder,[{'ProcessId':11}])

    def test_host_query_detects_modules_and_scripts_and_fails_on_inspection_errors(self):
        with patch.object(queue.subprocess,'check_output',return_value='[]') as call:
            self.assertEqual(queue.active_workers(),[])
        command=call.call_args.args[0][-1]
        self.assertIn('-ErrorAction Stop',command)
        self.assertIn(r'(\.py|\s|$)',command)

    def test_receipt_checks_all_outputs_and_preserved_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'original';folder.mkdir()
            manifest={'tag':'original','split':'validation','config_sha256':'config','evaluation_plan_sha256':'plan',
                'prompt_ids':[str(i) for i in range(43)],'settings':{'seed':42,'max_new_tokens':4096},
                'reused_original_manifest_sha256':'old'}
            old=root/'source'
            p.save(old/'manifest.json',{'tag':'original','settings':manifest['settings'],'system_prompt_sha256':'prompt'})
            manifest['system_prompt_sha256']='prompt'
            manifest['reused_original_manifest_sha256']=queue.sha(old/'manifest.json')
            p.save(folder/'manifest.json',manifest);digest=queue.sha(folder/'manifest.json')
            receipt={'tag':'original','split':'validation','completed':43,'actual_training_updates_in_this_job':0,
                'quality_claim':None,'manifest_sha256':digest,'policy_sha256':'weights'}
            p.save(folder/'verified.json',receipt)
            for number in manifest['prompt_ids']:
                p.save(folder/(number+'.json'),{'identity':{'id':number,'tag':'original','seed':42,'policy_sha256':'weights',
                    'manifest_sha256':digest},'quality_resampled':False,'teacher_forced':{'ce':1},
                    'generated_tokens':1,'generated_token_ids':[3]})
            cached=p.read(folder/'0.json');cached['identity']['manifest_sha256']=queue.sha(old/'manifest.json')
            p.save(old/'0.json',cached);p.save(folder/'0.json',cached)
            p.save(folder/'provenance/0.json',{'new_generation':False,'identity_preserved':True,
                'source':str(old/'0.json'),'sha256':queue.sha(old/'0.json')})
            queue.verify_outputs(folder,'original','config','plan')
            cached['quality_resampled']=True;p.save(folder/'0.json',cached)
            with self.assertRaises(AssertionError):queue.verify_outputs(folder,'original','config','plan')
            (folder/'1.json').unlink()
            with self.assertRaises((AssertionError,FileNotFoundError)):queue.verify_outputs(folder,'original','config','plan')

    def test_validation_queue_refuses_partial_or_final_results(self):
        receipt={'completed':43,'tag':'original','split':'validation','actual_training_updates_in_this_job':0,'quality_claim':None}
        verify_generation(receipt,'original')
        with self.assertRaises(AssertionError):verify_generation(dict(receipt,completed=42),'original')
        with self.assertRaises(AssertionError):verify_generation(dict(receipt,split='final'),'original')
        with self.assertRaises(AssertionError):verify_generation(dict(receipt,quality_claim='passed'),'original')

    def test_validation_refuses_training_or_test_prompts(self):
        config=p.read(e.OUT/'config.json')
        plan=p.read(e.OUT/'evaluation/evaluation_plan.json')
        books={b['source_id']:b for b in p.read(p.PROCESSED/'dataset.json')}
        self.assertEqual(len(e.prompt_rows(config,plan,'validation',books)),43)
        contaminated=copy.deepcopy(config)
        contaminated['train_ids'].append(config['validation_ids'][0])
        with self.assertRaises(AssertionError):e.prompt_rows(contaminated,plan,'validation',books)
        contaminated=copy.deepcopy(config)
        contaminated['excluded_test_ids'].append(config['validation_ids'][0])
        with self.assertRaises(AssertionError):e.prompt_rows(contaminated,plan,'validation',books)

    def test_final_prompts_have_frozen_digest_and_preupdate_origin(self):
        config=p.read(e.OUT/'config.json');plan=p.read(e.OUT/'evaluation/evaluation_plan.json')
        self.assertEqual(len(e.prompt_rows(config,plan,'final',{})),32)
        with self.assertRaises(AssertionError):e.prompt_rows(config,dict(plan,final_prompt_sha256='changed'),'final',{})

    def test_readable_format_failure_can_be_scored_but_blank_or_truncated_cannot(self):
        story={'title':'제목','characters':[], 'scenes':[{'scene_no':1,'text':'친구가 먼저 손을 내밀었어요.'}]}
        metric=e.local_metrics(json.dumps(story,ensure_ascii=False),True)
        self.assertFalse(metric['strict_format']);self.assertTrue(metric['semantic_eligible'])
        self.assertFalse(e.local_metrics(json.dumps(story),False)['semantic_eligible'])
        story['scenes'][0]['text']='   '
        self.assertFalse(e.local_metrics(json.dumps(story),True)['semantic_eligible'])
        self.assertFalse(e.local_metrics('unfinished {',True)['semantic_eligible'])

    def test_repeated_long_scene_preserves_whole_output_as_a_failure_candidate(self):
        text='친구는 상자를 내려놓고 나뭇가지에 걸린 작은 풍선을 꺼내 주었어요.'
        story={'title':'제목','characters':[],'scenes':[{'scene_no':1,'text':text},{'scene_no':2,'text':text+'!'}]}
        metric=e.local_metrics(json.dumps(story,ensure_ascii=False),True)
        self.assertEqual(metric['normalized_exact_repeated_long_scenes'],1)
        self.assertTrue(metric['semantic_eligible'])

    def test_discarded_hardware_checkpoint_is_never_evaluated_as_RL(self):
        with self.assertRaises(AssertionError):e.policy_context(e.OUT/'environment/full_policy_sharded/discarded_checkpoint')

    def test_observer_alerts_are_review_only_and_missing_counter_is_explicit(self):
        self.assertEqual(memory_alerts({'ram':{'free_kib':4*1024**2},'process_gpu_memory':[],'disk_free_gib':100}),[])
        alerts=memory_alerts({'ram':None,'memory_inspection_error':'unavailable'})
        self.assertEqual(alerts,['memory_inspection_error_no_restart'])
        alerts=memory_alerts({'ram':{'free_kib':1024},'process_gpu_memory':[{'SharedUsage':3*1024**3}],'disk_free_gib':2})
        self.assertEqual(len(alerts),3)
        self.assertTrue(all(a.endswith('review_only') for a in alerts))


if __name__=='__main__':unittest.main()
