"""The temporal stage uses the same accounting, exact reuse and response verification as rewards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from datetime import datetime,timezone,timedelta

from storymaker import project as p
import storymaker.rlaif.code as package

code=Path(__file__).parents[1]/'code'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
review=load('flow_review',code/'rlaif_temporal_review.py')
api=load('flow_reward_api',code/'rlaif_reward_api.py')


class TemporalRewardFlow(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        project=Path(temporary.name);self.run=project/'rlaif/results/current';self.source=self.run/'story.json'
        self.story={'raw_text':'{"scenes":[{"scene_no":1,"text":"친구들이 다리를 건넜어요."},{"scene_no":2,"text":"친구들이 처음 다리를 놓았어요."}]}','ended_with_eos':True,'moral':'함께 도와요.'}
        p.save(self.source,self.story);p.save(self.run/'config.json',{});p.save(self.run/'control.json',{'enabled':True})
        p.save(self.run/'budget_ledger.json',{'budget_usd':25,'calls':[]})
        p.save(self.run/'api_pricing_check.json',{'checked_date_kst':datetime.now(timezone(timedelta(hours=9))).date().isoformat(),
            'model':'gpt-4.1-2025-04-14','input_usd_per_million':2,'output_usd_per_million':8})
        corpus=project/'rlaif/rag/train200';corpus.mkdir(parents=True);(corpus/'corpus.jsonl').write_text('indexed causality',encoding='utf-8')
        criterion={'doc_id':'project_causality','source_id':None,'kind':'criterion','text':'선택과 사건과 결과의 실제 연결을 확인한다.'}
        digest=hashlib.sha256((corpus/'corpus.jsonl').read_bytes()).hexdigest()
        self.context={'train_only':True,'include_training_source':False,'source_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'corpus_sha256':digest,'references':[criterion]}
        self.value={'relation':'before','flashback_signal_quote':'','A_quote':'다리를 건넜어요.','B_quote':'다리를 놓았어요.','explanation':'Building the bridge must precede crossing.'}
        self.raw={'id':'local-response','model':'gpt-4.1-2025-04-14','choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.value)}}],
            'usage':{'prompt_tokens':100,'completion_tokens':100,'total_tokens':200}}
        for change in (patch.object(p,'ROOT',project),patch.object(api,'OUT',self.run),patch.object(api.rag,'FOLDER',corpus),
            patch.object(api.rag,'documents',return_value=([criterion],None,{'corpus_sha256':digest})),
            patch.dict(sys.modules,{'storymaker.rlaif.code.rlaif_temporal_review':review}),
            patch.object(package,'rlaif_temporal_review',review,create=True)):
            change.start();self.addCleanup(change.stop)

    def response(self,*args,**kwargs):
        body=kwargs['json'];self.assertFalse(body['logprobs']);self.assertEqual(body['model'],'gpt-4.1-2025-04-14')
        self.assertEqual(json.loads(body['messages'][1]['content'])['RAG'][0]['doc_id'],'project_causality')
        return SimpleNamespace(status_code=200,text=json.dumps(self.raw),json=lambda:copy.deepcopy(self.raw))

    def test_paid_local_review_reuse_and_tamper_detection(self):
        folder=self.run/'batch'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            context=api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            again=api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            self.assertEqual(context,again);post.assert_called_once()
            body=api.body_for(self.story,context);candidate=json.loads(body['messages'][-1]['content'])
            self.assertEqual(len(candidate['LOCAL_ERROR_EVIDENCE']),1)
            self.assertFalse(context['include_training_source'])
            saved=folder/'temporal/story/01_02/reward.json';value=p.read(saved);value['local_chronology']['relation']='after_or_same';p.save(saved,value)
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            post.assert_called_once()
        ledger=p.read(self.run/'budget_ledger.json');self.assertEqual(len(ledger['calls']),1);self.assertEqual(ledger['calls'][0]['state'],'completed')

    def test_unfinished_has_no_local_API_requests_and_keeps_score1_priority(self):
        self.story.update(ended_with_eos=False,raw_text='{"scenes":[');p.save(self.source,self.story)
        self.context['source_sha256']=hashlib.sha256(self.source.read_bytes()).hexdigest()
        with patch.object(api.requests,'post') as post:
            context=api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        post.assert_not_called();self.assertEqual(context['temporal_review']['transitions'],[])
        body=api.body_for(self.story,context);self.assertIn('UNFINISHED_SCORE_PRIORITY',json.loads(body['messages'][-1]['content']))

    def ungrounded_response(self):
        self.value['B_quote']='친구들이 다리를 놓았지요.'
        self.raw['choices'][0]['message']['content']=json.dumps(self.value)

    def test_ungrounded_paid_opinion_is_quarantined_and_not_resent(self):
        self.ungrounded_response();folder=self.run/'batch'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            context=api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            again=api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
        post.assert_called_once();self.assertEqual(context,again)
        target=folder/'temporal/story/01_02'
        self.assertFalse((target/'reward.json').exists())
        self.assertFalse(p.read(target/'unverified.json')['used_as_evidence'])
        candidate=json.loads(api.body_for(self.story,context)['messages'][-1]['content'])
        self.assertEqual(candidate['LOCAL_ERROR_EVIDENCE'],[])
        self.assertEqual(len(candidate['LOCAL_REVIEW_UNAVAILABLE']),1)
        self.assertNotIn('놓았지요',json.dumps(candidate,ensure_ascii=False))
        self.assertEqual(candidate['CANDIDATE_STORY_TO_RATE'],json.loads(self.story['raw_text']))
        ledger=p.read(self.run/'budget_ledger.json')
        self.assertEqual(len(ledger['calls']),1);self.assertEqual(ledger['calls'][0]['state'],'completed')

    def test_quarantine_receipt_tampering_is_a_hard_failure(self):
        self.ungrounded_response();folder=self.run/'batch'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            path=folder/'temporal/story/01_02/unverified.json';receipt=p.read(path)
            receipt['used_as_evidence']=True;p.save(path,receipt)
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
        post.assert_called_once()

    def test_unknown_charge_cannot_be_quarantined_or_resent(self):
        p.save(self.run/'budget_ledger.json',{'budget_usd':25,'calls':[{'state':'charge_unknown','cost_upper_usd':'0.01'}]})
        with patch.object(api.requests,'post') as post:
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        post.assert_not_called();self.assertFalse((self.run/'batch/temporal/story/01_02/unverified.json').exists())

    def test_wrong_model_remains_a_hard_failure(self):
        self.raw['model']='other'
        with patch.object(api.requests,'post',side_effect=self.response):
            with self.assertRaises(AssertionError) as error:api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        self.assertNotIsInstance(error.exception,review.UngroundedQuote)
        self.assertFalse((self.run/'batch/temporal/story/01_02/unverified.json').exists())

    def test_token_limited_local_response_is_quarantined_without_retransmission(self):
        self.raw['choices'][0]['finish_reason']='length'
        self.raw['choices'][0]['message']['content']='{"B_quote":"long repeated text'
        self.raw['usage'].update(completion_tokens=512,total_tokens=612)
        folder=self.run/'batch'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            context=api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            self.assertEqual(context,api.prepare_temporal_context(self.source,folder,'unit-key',self.context))
        post.assert_called_once()
        receipt=p.read(folder/'temporal/story/01_02/unverified.json')
        self.assertEqual(receipt['evidence_unavailable']['reason'],'local_response_token_limit')
        self.assertFalse(receipt['used_as_evidence'])
        self.assertFalse((folder/'temporal/story/01_02/reward.json').exists())
        candidate=json.loads(api.body_for(self.story,context)['messages'][-1]['content'])
        self.assertEqual(candidate['LOCAL_ERROR_EVIDENCE'],[])
        self.assertIn('No truncated opinion is evidence',candidate['LOCAL_REVIEW_WARNING'])
        self.assertEqual(candidate['CANDIDATE_STORY_TO_RATE'],json.loads(self.story['raw_text']))

    def test_token_limit_requires_matching_usage(self):
        self.raw['choices'][0]['finish_reason']='length'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        post.assert_called_once()
        self.assertFalse((self.run/'batch/temporal/story/01_02/unverified.json').exists())

    def test_other_finish_reasons_remain_hard_failures(self):
        self.raw['choices'][0]['finish_reason']='content_filter'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        post.assert_called_once()
        self.assertFalse((self.run/'batch/temporal/story/01_02/unverified.json').exists())

    def test_wrong_field_type_remains_a_hard_failure(self):
        self.value['B_quote']=123;self.raw['choices'][0]['message']['content']=json.dumps(self.value)
        with patch.object(api.requests,'post',side_effect=self.response):
            with self.assertRaises(AssertionError) as error:api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)
        self.assertNotIsInstance(error.exception,review.UngroundedQuote)

    def test_malformed_JSON_remains_a_hard_failure(self):
        self.raw['choices'][0]['message']['content']='{invalid'
        with patch.object(api.requests,'post',side_effect=self.response):
            with self.assertRaises(json.JSONDecodeError):api.prepare_temporal_context(self.source,self.run/'batch','unit-key',self.context)

    def test_previously_verified_response_cannot_be_downgraded_to_quarantine(self):
        folder=self.run/'batch'
        with patch.object(api.requests,'post',side_effect=self.response) as post:
            api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
            self.ungrounded_response();path=folder/'temporal/story/01_02/response.json'
            receipt=p.read(path);receipt['raw_body']=json.dumps(self.raw);p.save(path,receipt)
            with self.assertRaises(AssertionError):api.prepare_temporal_context(self.source,folder,'unit-key',self.context)
        post.assert_called_once();self.assertFalse((folder/'temporal/story/01_02/unverified.json').exists())


if __name__=='__main__':unittest.main()
