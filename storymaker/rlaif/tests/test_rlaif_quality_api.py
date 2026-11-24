"""Quality schema and money boundaries; never reads a real key or sends a real API request."""
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import requests
from storymaker import project as p
from storymaker.rlaif.code import rlaif_quality_api as q
def story(text='친구가 먼저 손을 내밀었어요.', eos=True):
    return {'moral':'서로 도우면 어려움을 풀 수 있어요.','ended_with_eos':eos,
        'raw_text':json.dumps({'title':'도움','characters':[], 'scenes':[{'scene_no':1,'text':text}]},ensure_ascii=False)}


def judgment():
    metric={'score':3,'scene_nos':[1],'evidence':'장면1에서 도움을 청한다.','weakness':'','minimal_fix':'','uncertainty':''}
    return {'A':{c:copy.deepcopy(metric) for c in q.CRITERIA},'B':{c:copy.deepcopy(metric) for c in q.CRITERIA},
        'preference':'tie','comparison_reason':'비슷하다.','human_review_needed':True,'human_review_reason':'사람 확인 전이다.'}


class QualityChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.override=patch.object(q,'OUT',self.root);self.override.start()
        self.settings={'model':'gpt-4.1-2025-04-14','max_completion_tokens':8192,'budget_usd':10,'final_evaluation_reserve_usd':2}
        p.save(self.root/'config.json',{})
        p.save(self.root/'budget_ledger.json',{'budget_usd':10,'calls':[]})
        p.save(self.root/'api_pricing_check.json',{'model':self.settings['model'],'input_usd_per_million':2,'output_usd_per_million':8,
            'checked_date_kst':datetime.now(timezone(timedelta(hours=9))).date().isoformat()})
        self.payload,self.mapping=q.pair_payload(story(),story(),0,'update_0001')

    def tearDown(self):
        self.override.stop();self.directory.cleanup()

    def response(self,content=None):
        raw={'id':'unit-response','model':self.settings['model'],
            'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(judgment()) if content is None else content}}]}
        class Response:
            status_code=200
            text=json.dumps(raw)
            def json(self):return raw
        return Response()

    def test_calibration_paths_use_the_explicit_preserved_SFT_source(self):
        sft=self.root/'saved_SFT'
        p.save(self.root/'config.json',{'source_sft_run':str(sft)})
        pairs=q.source_pairs('calibration','update_0001')
        self.assertEqual(pairs[0][1],sft/'evaluation/generations/pilot/205/base.json')
        self.assertEqual(pairs[1][2],sft/'evaluation/generations/pilot/143/sft.json')
        self.assertEqual(pairs[0][2],self.root/'evaluation/quality/calibration_inputs/205_epoch3_nonEOS.json')

    def test_reused_original_requires_exact_old_bytes_manifest_and_provenance(self):
        ids=[str(i) for i in range(43)];settings={'seed':42}
        p.save(self.root/'evaluation/evaluation_plan.json',{'generation':settings})
        old=self.root/'old_original'
        p.save(old/'manifest.json',{'tag':'original','settings':settings,'system_prompt_sha256':'prompt'})
        pairs=[(i,self.root/f'evaluation/validation/original/{i}.json',
            self.root/f'evaluation/validation/update_0001/{i}.json') for i in ids]
        for variant in ('original','update_0001'):
            folder=self.root/'evaluation/validation'/variant
            manifest={'settings':settings,'system_prompt_sha256':'prompt','split':'validation','tag':variant,
                'config_sha256':q.sha(self.root/'config.json'),'prompt_ids':ids,
                'reused_original_manifest_sha256':q.sha(old/'manifest.json') if variant=='original' else None}
            p.save(folder/'manifest.json',manifest);digest=q.sha(folder/'manifest.json')
            p.save(folder/'verified.json',{'tag':variant,'completed':43,'manifest_sha256':digest,
                'policy_sha256':variant+'_weights','actual_training_updates_in_this_job':0,'quality_claim':None})
            for number in ids:
                p.save(folder/(number+'.json'),{'identity':{'id':number,'tag':variant,'seed':42,
                    'manifest_sha256':digest,'policy_sha256':variant+'_weights'},'quality_resampled':False})
        path=pairs[0][1];cached=p.read(path)
        cached['identity']['manifest_sha256']=q.sha(old/'manifest.json')
        p.save(old/'0.json',cached);p.save(path,cached)
        p.save(path.parent/'provenance/0.json',{'new_generation':False,'identity_preserved':True,
            'source':str(old/'0.json'),'sha256':q.sha(old/'0.json')})
        with patch.object(q.requests,'post') as post:
            q.verify_generated_sources('validation','update_0001',pairs)
            post.assert_not_called()
        cached['raw_text']='changed';p.save(path,cached)
        with self.assertRaises(AssertionError):q.verify_generated_sources('validation','update_0001',pairs)

    def test_blind_order_balance_and_model_identity_not_in_payload(self):
        odd,mapping=q.pair_payload(story(),story(),1,'update_0001')
        self.assertEqual(self.mapping,{'A':'original','B':'update_0001'})
        self.assertEqual(mapping,{'A':'update_0001','B':'original'})
        self.assertEqual(set(odd),{'moral','A','B'})
        self.assertNotIn('update_0001',json.dumps(odd))

    def test_score_na_and_scene_evidence_follow_source_not_api_opinion(self):
        q.validate(judgment(),self.payload)
        wrong=judgment();wrong['A']['moral']['scene_nos']=[99]
        with self.assertRaises(AssertionError):q.validate(wrong,self.payload)
        truncated=copy.deepcopy(self.payload);truncated['A']['ended_with_eos']=False
        with self.assertRaises(AssertionError):q.validate(judgment(),truncated)
        valid=judgment()
        for value in valid['A'].values():value.update(score=None,scene_nos=[])
        valid['preference']='B';q.validate(valid,truncated)
        valid['preference']='A'
        with self.assertRaises(AssertionError):q.validate(valid,truncated)

    def test_cached_response_is_used_without_second_paid_request(self):
        with patch.object(q.requests,'post',return_value=self.response()) as post:
            first=q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','validation')
            second=q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','validation')
        self.assertEqual(post.call_count,1)
        self.assertEqual(first,second)
        self.assertEqual(p.read(self.root/'budget_ledger.json')['calls'][0]['state'],'completed')
        self.assertEqual(first['cost_upper_usd'],'0.0018')

    def test_same_paid_body_reused_across_destinations_retains_global_cost(self):
        from storymaker.rlaif.code import rlaif_budget
        real_verify=rlaif_budget.completed_response
        def verify(*args):
            with patch.object(p,'ROOT',self.root):
                return real_verify(*args)
        with patch.object(q.requests,'post',return_value=self.response()) as post:
            first=q.call_pair(self.root/'outputs/first',self.payload,self.settings,'unit-test','validation')
            with patch.object(q,'completed_response',side_effect=verify):
                second=q.call_pair(self.root/'outputs/second',self.payload,self.settings,'unit-test','validation')
        self.assertEqual(post.call_count,1)
        self.assertEqual(second['judgment'],first['judgment'])
        self.assertEqual(second['cost_upper_usd'],'0')
        self.assertEqual(len(p.read(self.root/'budget_ledger.json')['calls']),1)
        self.assertEqual(second['original_paid_cost_upper_usd'],'0.0018')
        request=self.root/'outputs/first/request.json'
        wrong=p.read(request);wrong['body']['temperature']=1;p.save(request,wrong)
        with patch.object(q.requests,'post') as post,patch.object(q,'completed_response',side_effect=verify):
            with self.assertRaises(AssertionError):
                q.call_pair(self.root/'outputs/third',self.payload,self.settings,'unit-test','validation')
        post.assert_not_called()

    def test_budget_and_final_reserve_are_checked_before_a_request(self):
        p.save(self.root/'config.json',{})
        p.save(self.root/'budget_ledger.json',{'budget_usd':10,'calls':[{'request_sha256':'old','state':'completed','cost_upper_usd':'8'}]})
        with patch.object(q.requests,'post') as post:
            with self.assertRaises(AssertionError):q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','validation')
        post.assert_not_called()

    def test_timeout_keeps_reservation_and_never_reposts(self):
        with patch.object(q.requests,'post',side_effect=requests.Timeout('unit timeout')) as post:
            with self.assertRaises(RuntimeError):q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','validation')
            with self.assertRaises(AssertionError):q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','validation')
        self.assertEqual(post.call_count,1)
        self.assertEqual(p.read(self.root/'budget_ledger.json')['calls'][0]['state'],'charge_unknown')

    def test_invalid_judgment_preserves_known_cost_without_retry(self):
        with patch.object(q.requests,'post',return_value=self.response('not JSON')) as post:
            with self.assertRaises(json.JSONDecodeError):q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','calibration')
            with self.assertRaises(AssertionError):q.call_pair(self.root/'pair',self.payload,self.settings,'unit-test','calibration')
        self.assertEqual(post.call_count,1)
        self.assertTrue((self.root/'pair/response.json').exists())
        self.assertEqual(p.read(self.root/'budget_ledger.json')['calls'][0]['cost_upper_usd'],'0.0018')

    def test_usage_rejects_boolean_token_counts(self):
        with self.assertRaises(AssertionError):q.usage_cost({'prompt_tokens':True,'completion_tokens':1,'total_tokens':2})

    def test_preserved_nonEOS_calibration_cannot_repair_raw_text_or_provenance(self):
        pairs = []
        original = self.root/'preserved.json'
        p.save(original, {'job': {'source_id':205}, 'generation':story(eos=False)})
        for number in ['205','143','9']:
            before,after = self.root/(number+'_before.json'),self.root/(number+'_after.json')
            p.save(before, {'state':'completed','job':{'source_id':int(number)},'generation':story()})
            item = p.read(original) if number == '205' else {'job':{'source_id':int(number)},'generation':story()}
            if number == '205':
                item['preserved_source'] = {'file':str(original),'sha256':q.sha(original)}
            p.save(after, {'state':'completed',**item})
            pairs.append((number,before,after))
        q.verify_generated_sources('calibration','calibration_sft',pairs)
        target = pairs[0][2]
        correct = p.read(target)
        wrong = copy.deepcopy(correct);wrong['generation']['raw_text'] += 'repaired'
        p.save(target,wrong)
        with self.assertRaises(AssertionError):q.verify_generated_sources('calibration','calibration_sft',pairs)
        p.save(target,correct)
        p.save(original,{'modified':True})
        with self.assertRaises(AssertionError):q.verify_generated_sources('calibration','calibration_sft',pairs)

    def test_both_nonEOS_outputs_require_not_evaluable_and_empty_evidence_ids(self):
        payload,_ = q.pair_payload(story(eos=False),story(eos=False),0,'update_0001')
        value = judgment()
        for label in ['A','B']:
            for metric in value[label].values():metric.update(score=None,scene_nos=[])
        value['preference'] = 'not_evaluable';q.validate(value,payload)
        value['preference'] = 'tie'
        with self.assertRaises(AssertionError):q.validate(value,payload)

    def test_request_constrains_source_failure_without_changing_raw_outputs(self):
        payload,_ = q.pair_payload(story(),story(eos=False),0,'update_0001')
        original = copy.deepcopy(payload)
        schema = q.request_body(payload,self.settings)['response_format']['json_schema']['schema']
        self.assertEqual(payload,original)
        self.assertEqual(schema['$defs']['Metric_A']['properties']['score']['type'],'integer')
        self.assertEqual(schema['$defs']['Metric_A']['properties']['scene_nos']['items']['enum'],[1])
        self.assertEqual(schema['$defs']['Metric_B']['properties']['score'],{'type':'null'})
        self.assertEqual(schema['$defs']['Metric_B']['properties']['scene_nos']['maxItems'],0)
        self.assertNotIn('B',schema['properties']['preference']['enum'])
        self.assertNotIn('not_evaluable',schema['properties']['preference']['enum'])
        value=judgment()
        for metric in value['B'].values():metric.update(score=None,scene_nos=[])
        value['preference']='A';q.validate(value,payload)

    def test_request_both_failures_and_readable_comparison_have_fixed_eligibility(self):
        payload,_=q.pair_payload(story(eos=False),story(eos=False),0,'update_0001')
        schema=q.judgment_schema(payload)
        self.assertEqual(schema['properties']['preference']['enum'],['not_evaluable'])
        for label in ('A','B'):
            self.assertEqual(schema['$defs']['Metric_'+label]['properties']['score'],{'type':'null'})
        value=judgment();value['preference']='not_evaluable'
        with self.assertRaises(AssertionError):q.validate(value,self.payload)


if __name__=='__main__':unittest.main()
