"""Reward-aligned validation metrics, train-only references and pre-request gates."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone,timedelta

import requests
import json
import csv
from PIL import Image
from storymaker import project as p
from storymaker.rlaif.code import rlaif_reward_api as reward
from storymaker.rlaif.code import rlaif_reward_evaluate as evaluation


def result(raw_reward,selected=3,eos=True):
    metric={'score':3,'scene_nos':[1],'evidence':'장면1의 실제 선택을 확인한다.'}
    judgment={'novelty':copy.deepcopy(metric),'completion':copy.deepcopy(metric),
        'content':{name:copy.deepcopy(metric) for name in reward.CRITERIA}}
    return {'result':{'normalized_reward':raw_reward,'selected_score':selected,'certified_decimal_places':6,'judgment':judgment},
        'local':{'eos':eos,'strict_format':True,'semantic_eligible':eos},'prefix_repetition':{'has_exact_repeat':False}}


class RewardEvaluationChecks(unittest.TestCase):
    def test_reward_comparison_keeps_the_declared_43_and_40_sensitivity(self):
        rows=[{'id':str(index),'models':{'original':result(0),'update_0001':result(1 if index<3 else -.5)}} for index in range(43)]
        primary=evaluation.summarize(rows,'update_0001','validation')
        secondary=evaluation.calibration_sensitivity(rows,'update_0001',['0','1','2'])
        self.assertEqual(primary['prompt_count'],43)
        self.assertEqual(secondary['remaining_denominator'],40)
        self.assertEqual(secondary['summary']['paired']['expected_reward']['mean_RL_minus_original'],-.5)
        self.assertEqual(secondary['excluded_technical_calibration_ids'],['0','1','2'])
        self.assertEqual(len(rows),43)

    def test_reward_sensitivity_rejects_missing_duplicate_or_unregistered_exclusions(self):
        rows=[{'id':str(index)} for index in range(43)]
        for excluded in (['0','1'],['0','0','1'],['0','1','999']):
            with self.subTest(excluded=excluded),self.assertRaises(AssertionError):
                evaluation.calibration_sensitivity(rows,'update_0001',excluded)

    def test_primary_reward_table_and_plot_keep_reference_CE_separate(self):
        rows=[{'id':'1','models':{'original':result(.5),'update_0001':result(.6)}}]
        for variant,value in [('original',9.),('update_0001',1.)]:
            rows[0]['models'][variant]['teacher_forced']={'ce':value,'supervised_tokens':10}
        summary=evaluation.summarize(rows,'update_0001','validation')
        self.assertAlmostEqual(summary['paired']['expected_reward']['mean_RL_minus_original'],.1)
        self.assertEqual(summary['models']['original']['auxiliary_reference_CE']['token_weighted_ce'],9.)
        for model in summary['models'].values():model['literal_overlap_candidate_count']=1
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);evaluation.write_summary(summary,folder)
            with (folder/'comparison.csv').open(encoding='utf-8-sig',newline='') as stream:table=list(csv.reader(stream))
            self.assertEqual(table[1][0],'expected_reward');self.assertEqual(table[-1][0],'auxiliary_reference_CE')
            self.assertEqual(dict((row[0],row[1:]) for row in table)['semantic_eligible_n'],['1','1'])
            self.assertEqual(table[-2][0],'literal_overlap_candidate_rate')
            with Image.open(folder/'comparison.png') as image:self.assertEqual(image.size,(2400,975))

    def test_report_metrics_are_reparsed_from_the_paid_response_and_not_a_mutable_cache(self):
        from storymaker.rlaif.tests.test_rag_reward import response_for
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);run=project/'rlaif/results/current';source=run/'source.json';destination=run/'call'
            story={'moral':'함께 도와요.','raw_text':'{"scenes":[{"scene_no":1,"text":"서로 도와 다리를 놓았어요."}]}','ended_with_eos':True}
            p.save(source,story);p.save(run/'config.json',{})
            context={'train_only':True,'references':[{'doc_id':'train_23_full','source_id':23,'text':'함께 힘을 모아 다리를 놓았어요.'}]}
            metric={'score':3,'scene_nos':[1],'evidence':'장면1의 선택이 결과와 연결된다.'}
            judgment={'event_order':[{'scene_no':1,'prior':[],'conflict':False}],
                'novelty':dict(metric,score=1),'completion':dict(metric),'content':{name:dict(metric) for name in reward.CRITERIA},
                'citation_ids':['e1'],'summary':'동일한 사건을 다시 썼다.','score':3}
            raw=response_for(judgment);raw.update(id='unit-response',usage={'prompt_tokens':100,'completion_tokens':100,'total_tokens':200})
            body=reward.body_for(story,context);signature=reward.hashlib.sha256(json.dumps(body,ensure_ascii=False).encode('utf-8')).hexdigest()
            p.save(destination/'request.json',body);p.save(destination/'response.json',{'http_status':200,'raw_body':json.dumps(raw,ensure_ascii=False)})
            cached=reward.parse(raw,story,context);cached.update(source_sha256=evaluation.sha(source),request_sha256=signature)
            p.save(destination/'reward.json',cached)
            p.save(run/'budget_ledger.json',{'calls':[{'state':'completed','request_sha256':signature,'response_id':raw['id'],
                'destination':str(destination),'usage':raw['usage'],'cost_upper_usd':'.001'}]})
            with patch.object(p,'ROOT',project),patch.object(evaluation,'OUT',run),patch.object(reward,'OUT',run):
                self.assertEqual(evaluation.verified_result(source,destination,context),cached)
                cached['judgment']['novelty']['score']=4;p.save(destination/'reward.json',cached)
                with self.assertRaises(AssertionError):evaluation.verified_result(source,destination,context)

    def test_held_out_story_is_not_added_as_its_own_reference(self):
        config={'train_ids':[1,2,3],'validation_ids':[9],'excluded_test_ids':[10]}
        docs=[{'kind':'example','source_id':number} for number in (2,1,3)]
        self.assertEqual(reward.reference_sources(config,{'identity':{'id':'9'}},docs,[0,1,2],False),[2,1,3])
        self.assertEqual(reward.reference_sources(config,{'identity':{'source_id':1}},docs,[0,1,2],True),[1,2,3])
        with self.assertRaises(AssertionError):reward.reference_sources(config,{'identity':{'source_id':9}},docs,[0,1,2],True)
        docs[0]['source_id']=9
        with self.assertRaises(AssertionError):reward.reference_sources(config,{'identity':{'id':'9'}},docs,[0,1,2],False)

    def test_expected_reward_and_all_nine_components_keep_failures_in_denominator(self):
        rows=[{'id':'1','models':{'original':result(.5),'update_0001':result(-1,1,False)}},
            {'id':'2','models':{'original':result(.5),'update_0001':result(.5)}}]
        summary=evaluation.summarize(rows,'update_0001','validation')
        model=summary['models']['update_0001']
        self.assertEqual(model['mean_expected_reward'],-.25)
        self.assertEqual(model['total_n'],2);self.assertEqual(model['eos_count'],1)
        self.assertEqual(set(model['diagnostics']),set(evaluation.DIAGNOSTICS))
        self.assertEqual(len(model['diagnostics']),9)
        self.assertTrue(all(value['denominator']==2 for value in model['diagnostics'].values()))
        self.assertEqual(summary['paired']['expected_reward']['mean_RL_minus_original'],-.75)
        self.assertEqual(summary['reward_wins_ties_losses'],{'win':0,'tie':1,'loss':1})

    def test_sampled_score_is_not_substituted_for_probability_expectation(self):
        rows=[{'id':'1','models':{'original':result(-.5,9),'update_0001':result(.2,3)}}]
        summary=evaluation.summarize(rows,'update_0001','validation')
        self.assertAlmostEqual(summary['paired']['expected_reward']['mean_RL_minus_original'],.7)
        self.assertEqual(summary['reward_wins_ties_losses']['win'],1)
        self.assertEqual(summary['models']['update_0001']['mean_selected_score'],3)

    def test_failed_outputs_do_not_inflate_eligible_content_and_keep_zero_utility(self):
        failed=result(-1,1,False)
        for item in failed['result']['judgment']['content'].values():item['score']=4
        rows=[{'id':'1','models':{'original':result(.5),'update_0001':failed}},
            {'id':'2','models':{'original':result(.5),'update_0001':result(.5)}}]
        summary=evaluation.summarize(rows,'update_0001','validation');model=summary['models']['update_0001']
        self.assertEqual(model['semantic_eligible_n'],1);self.assertEqual(model['semantic_ineligible_n'],1)
        self.assertEqual(model['eligible_content_diagnostics']['moral'],{'mean':3,'denominator':1})
        self.assertEqual(model['diagnostics']['moral']['mean'],3.5)
        self.assertEqual(model['failure_inclusive_content_utility'],.375)
        self.assertEqual(summary['paired_failure_inclusive_content_utility']['mean_RL_minus_original'],-.375)
        self.assertEqual(summary['paired_eligible_content']['moral']['denominator'],1)
        self.assertEqual(model['mean_expected_reward'],-.25)

    def test_no_eligible_content_is_NA_while_failure_utility_and_reward_remain(self):
        rows=[{'id':'1','models':{'original':result(-1,1,False),'update_0001':result(-1,1,False)}}]
        summary=evaluation.summarize(rows,'update_0001','validation')
        for model in summary['models'].values():
            self.assertEqual(model['semantic_eligible_n'],0)
            self.assertIsNone(model['eligible_content_diagnostics']['age_fit']['mean'])
            self.assertEqual(model['failure_inclusive_content_utility'],0)
        self.assertIsNone(summary['paired_eligible_content']['age_fit']['mean_RL_minus_original'])

    def test_readable_format_errors_are_still_content_eligible(self):
        record=result(.5);record['local']['strict_format']=False
        summary=evaluation.summarize([{'id':'1','models':{'original':record,'update_0001':copy.deepcopy(record)}}],'update_0001','validation')
        self.assertEqual(summary['models']['original']['semantic_eligible_n'],1)
        self.assertEqual(summary['models']['original']['strict_format_count'],0)

    def test_eligibility_requires_an_explicit_consistent_flag(self):
        for problem in ('missing','inconsistent'):
            row={'id':'1','models':{'original':result(.5),'update_0001':result(.5)}}
            if problem=='missing':row['models']['original']['local'].pop('semantic_eligible')
            else:row['models']['original']['local']['eos']=False
            with self.assertRaises((AssertionError,KeyError)):evaluation.summarize([row],'update_0001','validation')

    def test_duplicate_ids_or_uncertified_rewards_cannot_be_aggregated(self):
        row={'id':'1','models':{'original':result(.1),'update_0001':result(.2)}}
        with self.assertRaises(AssertionError):evaluation.summarize([row,row],'update_0001','validation')
        row['models']['original']['result']['certified_decimal_places']=0
        with self.assertRaises(AssertionError):evaluation.summarize([row],'update_0001','validation')

    def test_plan_sends_no_request_and_does_not_require_model_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p.save(root/'config.json',{})
            pairs=[(str(i),root/f'original/{i}.json',root/f'policy/{i}.json') for i in range(43)]
            with patch.object(evaluation,'OUT',root),patch.object(evaluation,'source_pairs',return_value=pairs), \
                 patch('sys.argv',['evaluate','--phase','validation']),patch.object(reward,'score') as score:
                evaluation.main()
            score.assert_not_called()
            plan=p.read(root/'evaluation/reward_metrics/validation_update_0001/plan.json')
            self.assertEqual(plan['maximum_story_requests'],86)
            self.assertEqual(len(plan['criteria']),9);self.assertFalse(plan['execute'])

    def test_failed_protocol_blocks_credential_read_and_paid_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p.save(root/'config.json',{});p.save(root/'control.json',{'enabled':True})
            with patch.object(evaluation,'OUT',root),patch.object(evaluation,'source_pairs',return_value=[]), \
                 patch.object(evaluation,'verify_generated_sources'),patch.object(evaluation,'accepted_protocol',side_effect=AssertionError('failed contrasts')), \
                 patch('sys.argv',['evaluate','--execute']),patch.object(evaluation.yaml,'safe_load') as key,patch.object(reward,'score') as score:
                with self.assertRaises(AssertionError):evaluation.main()
            key.assert_not_called();score.assert_not_called()

    def test_only_final_evaluation_can_use_the_reserved_two_dollars(self):
        for phase in (None,'validation','final'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'story.json'
                p.save(source,{'moral':'서로 도와요.','raw_text':'{"scenes":[]}','ended_with_eos':False})
                p.save(root/'config.json',{})
                p.save(root/'budget_ledger.json',{'budget_usd':10,'previous_RL_spend_usd':.2,
                    'calls':[{'state':'completed','cost_upper_usd':'7.8'}]})
                p.save(root/'api_pricing_check.json',{'model':'gpt-4.1-2025-04-14','input_usd_per_million':2,
                    'output_usd_per_million':8,'checked_date_kst':datetime.now(timezone(timedelta(hours=9))).date().isoformat()})
                context={'source_sha256':reward.hashlib.sha256(source.read_bytes()).hexdigest(),'corpus_sha256':'corpus'}
                body={'model':'gpt-4.1-2025-04-14','max_completion_tokens':3072}
                with patch.object(reward,'OUT',root),patch.object(reward.rag,'sha',return_value='corpus'), \
                     patch.object(reward,'body_for',return_value=body),patch.object(reward.requests,'post',side_effect=requests.Timeout) as post:
                    with self.assertRaises(RuntimeError if phase=='final' else AssertionError):
                        reward.score(source,root/'call','unit-key',context,evaluation_phase=phase)
                self.assertEqual(post.call_count,1 if phase=='final' else 0)
                if phase=='final':self.assertEqual(p.read(root/'budget_ledger.json')['calls'][-1]['state'],'charge_unknown')


if __name__=='__main__':unittest.main()
