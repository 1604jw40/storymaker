"""Whole-denominator and paired-statistic checks, including truncated readable prefixes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from storymaker.qwen.code.evaluation_api import CRITERIA
from storymaker.rlaif.code.rlaif_quality_api import pair_payload
from storymaker import project as p
from storymaker.rlaif.code.rlaif_quality_report import summarize, copy_candidates, calibration_sensitivity, historical_calibration, training_output_sources, sha


TAG='update_0001'


def generation(eos=True,valid_format=True,ce=1.,tokens=4):
    character={'character_id':'c','name':'친구','goal':'분실물을 돌려준다.','traits':'침착하다.' if valid_format else []}
    body='친구는 주운 상자를 내려놓고 나뭇가지에 걸린 작은 풍선을 꺼내 주었어요.'
    book={'title':'작은 도움','characters':[character],'scenes':[{'scene_no':1,'character_ids':['c'],
        'setting':'길','text_mode':'read_aloud','text':body,'visual_note':'친구가 풍선을 꺼내 준다.'}]}
    return {'moral':'서로 도와요.','raw_text':json.dumps(book,ensure_ascii=False),'ended_with_eos':eos,
        'teacher_forced':{'ce':ce,'supervised_tokens':tokens,'gold_prefix_eos_rank':1}}


def report_for(ids,outputs):
    rows=[]
    for index,number in enumerate(ids):
        before,after=outputs[number,'original'],outputs[number,TAG]
        _,mapping=pair_payload(before,after,index,TAG)
        judgment={'comparison_reason':'고정 작품 비교','human_review_needed':True,'human_review_reason':'사람 검토 전'}
        for label,variant in mapping.items():
            score=(3 if variant=='original' else 4) if outputs[number,variant]['ended_with_eos'] else None
            metric={'score':score,'scene_nos':[1] if score is not None else [],'evidence':'장면1의 선택을 확인한다.','weakness':'','minimal_fix':'','uncertainty':''}
            judgment[label]={criterion:copy.deepcopy(metric) for criterion in CRITERIA}
        if not before['ended_with_eos'] and not after['ended_with_eos']:
            judgment['preference']='not_evaluable'
        else:
            choice=TAG if after['ended_with_eos'] else 'original'
            judgment['preference']=next(label for label,variant in mapping.items() if variant==choice)
        rows.append({'id':number,'mapping':mapping,'result':{'judgment':judgment}})
    return {'identity':{'phase':'validation','tag':TAG},'all_responses_complete':True,'completed_pairs':len(ids),'pairs':rows}


class ReportChecks(unittest.TestCase):
    def test_old_calibration_is_audited_in_its_own_config_without_freezing_the_new_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)/'old';current=Path(directory)/'current'
            p.save(old/'config.json',{'retained':True})
            rows=[]
            for number in (205,143,9):
                source=old/'pilot'/f'{number}.json'
                p.save(source,{'state':'completed','job':{'split':'pilot','source_id':number}})
                rows.append({'source':str(source),'source_sha256':sha(source)})
            p.save(old/'reward_protocol_frozen.json',{'results':rows})
            ids=['205','143','9']+[str(i) for i in range(40) if i!=9]+['100']
            p.save(old/'evaluation/calibration_split_audit.json',{'config_sha256':sha(old/'config.json'),
                'reward_protocol_frozen_sha256':sha(old/'reward_protocol_frozen.json'),'actual_RL_updates_at_declaration':0,
                'primary_denominator':43,'primary_validation_ids':ids,'technical_calibration_ids':['205','143','9']})
            config={'source_previous_run':str(old),'source_previous_config_sha256':sha(old/'config.json')}
            path,calibration_ids=historical_calibration(config,ids)
            self.assertEqual(path,old/'evaluation/calibration_split_audit.json')
            self.assertEqual(calibration_ids,['205','143','9'])
            self.assertFalse((current/'reward_protocol_frozen.json').exists())
            middle=Path(directory)/'middle'
            p.save(middle/'config.json',config)
            deeper={'source_previous_run':str(middle),'source_previous_config_sha256':sha(middle/'config.json')}
            self.assertEqual(historical_calibration(deeper,ids)[0],path)
            p.save(old/'pilot/205.json',{'changed':True})
            with self.assertRaises(AssertionError):historical_calibration(config,ids)

    def test_copy_references_exclude_later_updates_and_all_held_out_IDs(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs=[Path(temporary)/str(i) for i in range(3)]
            for i,root in enumerate(runs):
                config={'train_ids':[23,24,25],'validation_ids':[200],'excluded_test_ids':[304]}
                if i:config.update(source_previous_run=str(runs[i-1]),source_previous_config_sha256=sha(runs[i-1]/'config.json'),
                    policy_checkpoint=str(runs[i-1]/f'checkpoints/update_{i:04d}/policy'),
                    value_checkpoint=str(runs[i-1]/f'checkpoints/update_{i:04d}/value'))
                p.save(root/'config.json',config);tag=f'update_{i+1:04d}'
                p.save(root/'batch_0000/joint_update_verified.json',{'policy':str(root/'checkpoints'/tag/'policy'),
                    'value':str(root/'checkpoints'/tag/'value'),'committed_policy_updates':1,'committed_value_updates':1})
                p.save(root/'batch_0000/rollouts/0.json',{'identity':{'source_id':23+i},'raw_text':'훈련 출력'})
            self.assertEqual(training_output_sources(runs[2],'update_0001'),[runs[0]/'batch_0000/rollouts/0.json'])
            self.assertEqual(len(training_output_sources(runs[2],'update_0003')),3)
            p.save(runs[2]/'batch_0000/rollouts/0.json',{'identity':{'source_id':200},'raw_text':'검증 출력'})
            self.assertEqual(len(training_output_sources(runs[2],'update_0001')),1)
            with self.assertRaises(AssertionError):training_output_sources(runs[2],'update_0003')

    def test_nonEOS_remains_in_whole_denominator_and_nullable_scores(self):
        outputs={('1','original'):generation(),('1',TAG):generation(eos=False)}
        summary=summarize(report_for(['1'],outputs),['1'],outputs,TAG)
        self.assertEqual(summary['models'][TAG]['total_n'],1)
        self.assertEqual(summary['models'][TAG]['criteria']['moral']['valid_n'],0)
        self.assertIsNone(summary['models'][TAG]['criteria']['moral']['mean_valid_only'])
        self.assertEqual(summary['models'][TAG]['quality_pass_count'],0)
        self.assertEqual(summary['paired_criteria']['moral']['jointly_valid_n'],0)
        self.assertEqual(summary['quality_pass_difference']['mean_RL_minus_original'],-1)
        self.assertEqual(summary['failure_aware_utility_difference']['mean_RL_minus_original'],-.75)

    def test_token_weighted_ce_and_paired_difference_use_matching_prompts(self):
        outputs={('1','original'):generation(ce=1,tokens=2),('2','original'):generation(ce=2,tokens=6),
                 ('1',TAG):generation(),('2',TAG):generation()}
        summary=summarize(report_for(['1','2'],outputs),['1','2'],outputs,TAG)
        self.assertEqual(summary['models']['original']['teacher_forced']['token_weighted_ce'],1.75)
        self.assertEqual(summary['paired_criteria']['moral']['mean_RL_minus_original'],1)
        self.assertEqual(summary['paired_criteria']['moral']['bootstrap_95_interval'],[1,1])
        with self.assertRaises(AssertionError):summarize(report_for(['1','2'],outputs),['1','2'],{key:value for key,value in outputs.items() if key!=('2',TAG)},TAG)

    def test_readable_format_failure_does_not_hide_content_or_pass_product_gate(self):
        outputs={('1','original'):generation(eos=False),('1',TAG):generation(valid_format=False)}
        summary=summarize(report_for(['1'],outputs),['1'],outputs,TAG)
        model=summary['models'][TAG]
        self.assertEqual(model['semantic_eligible_count'],1)
        self.assertEqual(model['criteria']['moral']['mean_valid_only'],4)
        self.assertEqual(model['quality_pass_count'],0)
        self.assertEqual(summary['failure_aware_utility_difference']['mean_RL_minus_original'],1)

    def test_literal_overlap_can_be_checked_in_complete_fields_of_truncated_JSON(self):
        text='친구는 주운 상자를 내려놓고 나뭇가지에 걸린 작은 풍선을 꺼내 주었어요.'*3
        reference={'id':'train1','raw_text':json.dumps({'scenes':[{'text':text}]},ensure_ascii=False)}
        prefix='{"scenes":[{"text":'+json.dumps(text,ensure_ascii=False)+'},{"text":"unfinished'
        result=copy_candidates([reference],{('new',TAG):{'raw_text':prefix}})
        self.assertEqual(result['checked_outputs'],1)
        self.assertEqual(result['matches'][0]['reference_id'],'train1')
        self.assertGreaterEqual(result['matches'][0]['normalized_characters'],40)

    def test_calibration_sensitivity_preserves_primary43_and_its_failures(self):
        ids = [str(i) for i in range(1,44)]
        outputs = {(number,variant): generation(eos=not(number=='1' and variant==TAG))
            for number in ids for variant in ('original',TAG)}
        report = report_for(ids,outputs)
        primary = summarize(report,ids,outputs,TAG)
        secondary = calibration_sensitivity(report,outputs,TAG,['1','2','3'])
        self.assertEqual(primary['models'][TAG]['quality_pass_count'],42)
        self.assertEqual(primary['prompt_count'],43)
        self.assertEqual(secondary['remaining_denominator'],40)
        self.assertEqual(secondary['summary']['models'][TAG]['quality_pass_count'],40)
        self.assertEqual(secondary['excluded_technical_calibration_ids'],['1','2','3'])
        self.assertEqual(len(report['pairs']),43)
        self.assertEqual(len(outputs),86)

    def test_sensitivity_cannot_drop_other_IDs_or_redefine_final_evaluation(self):
        ids = [str(i) for i in range(1,44)]
        outputs = {(number,variant): generation() for number in ids for variant in ('original',TAG)}
        report = report_for(ids,outputs)
        for excluded in [['1','2','99'],['1','2','2'],['1','2']]:
            with self.assertRaises(AssertionError):
                calibration_sensitivity(report,outputs,TAG,excluded)
        report['identity']['phase'] = 'final'
        with self.assertRaises(AssertionError):
            calibration_sensitivity(report,outputs,TAG,['1','2','3'])


if __name__=='__main__':unittest.main()
