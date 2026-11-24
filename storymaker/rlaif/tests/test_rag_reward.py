"""RAG evidence, first score-token likelihoods, and pre-request money boundaries."""
import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest.mock import patch
from storymaker import project as p
from storymaker.rlaif.code import rlaif_reward_api as reward


def response_for(value):
    text=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    position=text.rindex('"score":')+len('"score":')
    prefix=text[:position].encode('utf-8');tail=text[position+len(str(value['score'])):].encode('utf-8')
    entries=[{'token':str(i),'logprob':-.1 if i==value['score'] else -8} for i in range(1,11)]
    tokens=[{'token':'partial-unicode','bytes':list(prefix[:47])}, {'token':'repeated-score-prefix','bytes':list(prefix[47:])},
        {'token':str(value['score']),'bytes':list(str(value['score']).encode()),'top_logprobs':entries},
        {'token':'partial-unicode','bytes':list(tail[:47])}, {'token':'remaining','bytes':list(tail[47:])}]
    return {'model':'gpt-4.1-2025-04-14','choices':[{'finish_reason':'stop',
        'message':{'content':text},'logprobs':{'content':tokens}}]}



class RagRewardChecks(unittest.TestCase):
    def setUp(self):
        self.story={'identity':{'source_id':23},'moral':'서로 도와요.','ended_with_eos':True,
            'raw_text':json.dumps({'scenes':[{'scene_no':1,'text':'서로 다리를 놓았어요.'}]},ensure_ascii=False)}
        self.context={'train_only':True,'references':[{'doc_id':'train_23_full','source_id':23,'text':'친구들이 서로 힘을 모아 다리를 놓았어요.'}]}
        metric={'score':3,'scene_nos':[1],'evidence':'장면1의 선택이 결과와 연결된다.'}
        self.judgment={'event_order':[{'scene_no':1,'prior':[],'conflict':False}],
            'novelty':dict(metric,score=1),'completion':dict(metric),
            'content':{name:dict(metric) for name in reward.CRITERIA},
            'citation_ids':['e1'],'summary':'같은 사건 구조를 다시 썼다.','score':3}

    def response(self,value):
        return response_for(value)

    def test_request_requires_RAG_and_score_after_evidence(self):
        body=reward.body_for(self.story,self.context)
        self.assertEqual(list(body['response_format']['json_schema']['schema']['properties'])[-1],'score')
        self.assertEqual(body['max_completion_tokens'],3072)
        self.assertIn('RAG',json.loads(body['messages'][1]['content']))
        self.assertEqual(json.loads(body['messages'][-1]['content'])['CANDIDATE_STORY_TO_RATE'],json.loads(self.story['raw_text']))
        with self.assertRaises(AssertionError):reward.body_for(self.story,dict(self.context,references=[]))

    def test_partial_JSON_has_no_permitted_scene_citations(self):
        body=reward.body_for(dict(self.story,raw_text='{"scenes":[',ended_with_eos=False),self.context)
        schema=body['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['novelty']['properties']['scene_nos']['maxItems'],0)
        self.assertEqual(json.loads(body['messages'][-1]['content'])['parsed_scene_numbers'],[])

    def test_score_offset_handles_repeated_numbers_and_utf8_byte_pieces(self):
        result=reward.parse(self.response(self.judgment),self.story,self.context)
        self.assertEqual(result['selected_score'],3)
        self.assertEqual(result['observed_score_count'],10)
        bad=copy.deepcopy(self.judgment);bad['citation_ids']=['invented']
        with self.assertRaises(AssertionError):reward.parse(self.response(bad),self.story,self.context)
        bad=copy.deepcopy(self.judgment);bad['content']['moral']['scene_nos']=[99]
        with self.assertRaises(AssertionError):reward.parse(self.response(bad),self.story,self.context)

    def test_unfinished_and_high_score_anchors_are_enforced(self):
        with self.assertRaises(AssertionError):reward.parse(self.response(self.judgment),dict(self.story,ended_with_eos=False),self.context)
        bad=copy.deepcopy(self.judgment);bad['score']=9
        with self.assertRaises(AssertionError):reward.parse(self.response(bad),self.story,self.context)

    def test_event_conflict_cannot_receive_high_causality(self):
        bad=copy.deepcopy(self.judgment);bad['event_order'][0]['prior']=[1]
        with self.assertRaises(AssertionError):reward.parse(self.response(bad),self.story,self.context)
        bad['content']['causality']['score']=1
        reward.parse(self.response(bad),self.story,self.context)

    def test_budget_reserves_output_tokens_and_final_evaluation_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'story.json';p.save(source,self.story)
            context=dict(self.context,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),corpus_sha256='corpus')
            p.save(root/'config.json',{})
            p.save(root/'budget_ledger.json',{'budget_usd':10,'calls':[{'state':'completed','cost_upper_usd':'7.99'}]})
            p.save(root/'api_pricing_check.json',{'checked_date_kst':datetime.now(timezone(timedelta(hours=9))).date().isoformat(),
                'model':'gpt-4.1-2025-04-14','input_usd_per_million':2,'output_usd_per_million':8})
            with patch.object(reward,'OUT',root),patch.object(reward.rag,'sha',return_value='corpus'),patch.object(reward.requests,'post') as post:
                with self.assertRaises(AssertionError):reward.score(source,root/'call','unit-key',context)
            post.assert_not_called()


class CalibrationChecks(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.project=Path(temporary.name);self.run=self.project/'rlaif/results/current'
        self.folder=self.run/'reward_calibration';self.corpus=self.project/'rlaif/rag/train200'
        self.corpus.mkdir(parents=True);(self.corpus/'corpus.jsonl').write_text('fixed corpus',encoding='utf-8')
        for target,name,value in [(p,'ROOT',self.project),(reward,'OUT',self.run),(reward.rag,'FOLDER',self.corpus)]:
            change=patch.object(target,name,value);change.start();self.addCleanup(change.stop)
        self.config={'train_ids':[23,139,191],'api_model':'gpt-4.1-2025-04-14','rag_required':True}
        p.save(self.run/'config.json',self.config);p.save(self.run/'control.json',{'enabled':True})
        self.ledger={'budget_usd':10,'calls':[]};self.judgments={};entries=[]
        for number in self.config['train_ids']:
            for variant in ['copy','renamed','distinct','cutoff','repetition','causal_break']:
                name=f'{number}_{variant}';source=self.folder/'cases'/(name+'.json')
                story={'identity':{'source_id':number},'moral':'서로 도와요.','ended_with_eos':variant!='cutoff',
                    'raw_text':json.dumps({'scenes':[{'scene_no':1,'text':'친구와 함께 작은 다리를 놓았어요.'}]},ensure_ascii=False)}
                if variant=='cutoff':story['raw_text']='{"scenes":['
                p.save(source,story)
                context={'train_only':True,'source_sha256':reward.rag.sha(source),'corpus_sha256':reward.rag.sha(self.corpus/'corpus.jsonl'),
                    'references':[{'doc_id':f'train_{number}_full','source_id':number,'text':'친구들이 서로 힘을 모아 다리를 놓았어요.'}]}
                p.save(self.folder/'rag'/source.name,context)
                scenes=[] if variant=='cutoff' else [1]
                metric={'score':4 if variant=='distinct' else 3,'scene_nos':scenes,'evidence':'실제 장면의 근거다.'}
                value={'event_order':[] if not scenes else [{'scene_no':1,'prior':[],'conflict':False}],
                    'novelty':copy.deepcopy(metric),'completion':copy.deepcopy(metric),
                    'content':{key:copy.deepcopy(metric) for key in reward.CRITERIA},'citation_ids':['e1'],
                    'summary':'통제된 절차 검사이며 사람 평가가 아니다.','score':9 if variant=='distinct' else 3}
                if variant in ('copy','renamed'):value['novelty']['score']=0
                if variant=='cutoff':value.update(score=1);value['completion']['score']=0
                if variant=='repetition':value['content']['scene_density']['score']=1
                if variant=='causal_break':value['content']['causality']['score']=1
                self.judgments[name]=value
                self.write_response(name,value)
                entries.append({'source_id':number,'variant':variant,'path':str(source),'sha256':reward.rag.sha(source)})
        p.save(self.folder/'expectations.json',{'registered_before_paid_calls':True,'training_sources':self.config['train_ids'],'cases':entries})
        p.save(self.folder/'current_plan.json',{'expectations_sha256':reward.rag.sha(self.folder/'expectations.json')})
        p.save(self.folder/'rag_status.json',{'stage':'complete','completed':18,'total':18})

    def write_response(self,name,value):
        source=self.folder/'cases'/(name+'.json');story=p.read(source);context=p.read(self.folder/'rag'/source.name)
        destination=self.folder/'current'/name;body=reward.body_for(story,context)
        signature=hashlib.sha256(json.dumps(body,ensure_ascii=False).encode('utf-8')).hexdigest()
        raw=response_for(value);raw.update(id=name,usage={'prompt_tokens':100,'completion_tokens':100,'total_tokens':200})
        p.save(destination/'request.json',body)
        p.save(destination/'response.json',{'http_status':200,'seconds':0.,'raw_body':json.dumps(raw,ensure_ascii=False)})
        parsed=reward.parse(raw,story,context)
        parsed.update(request_sha256=signature,source_sha256=reward.rag.sha(source),model=raw['model'])
        p.save(destination/'reward.json',parsed)
        record={'label':str(destination.relative_to(self.run)),'destination':str(destination),'request_sha256':signature,
            'state':'completed','cost_upper_usd':'.001','response_id':name,'usage':raw['usage']}
        self.ledger['calls']=[r for r in self.ledger['calls'] if r['response_id']!=name]+[record]
        p.save(self.run/'budget_ledger.json',self.ledger)

    def test_all18_grounded_responses_and_relative_contrasts_are_required(self):
        with patch.object(reward.requests,'post') as post:
            result=reward.check_calibration(self.config,self.folder)
        self.assertTrue(result['passed']);self.assertEqual(result['accepted_cases'],18)
        self.assertEqual(len(result['checks']),3);post.assert_not_called()
        self.assertFalse((self.run/'reward_protocol_frozen.json').exists())

    def test_schema_valid_causal_misjudgment_does_not_pass_ordinal_validation(self):
        wrong=copy.deepcopy(self.judgments['139_distinct'])
        self.write_response('139_causal_break',wrong)
        result=reward.check_calibration(self.config,self.folder)
        self.assertEqual(result['accepted_cases'],18);self.assertFalse(result['passed'])
        self.assertFalse(next(r for r in result['checks'] if r['source_id']==139)['causal_order_penalty'])

    def test_partial_receipts_reward_tampering_and_ledger_mismatch_cannot_freeze(self):
        for problem in ('missing','reward','ledger'):
            with self.subTest(problem=problem):
                path=self.folder/'current/23_distinct/reward.json';saved=p.read(path)
                original_ledger=copy.deepcopy(self.ledger)
                if problem=='missing':path.unlink()
                elif problem=='reward':
                    wrong=dict(saved,normalized_reward=0);p.save(path,wrong)
                else:
                    self.ledger['calls'][0]['response_id']='different';p.save(self.run/'budget_ledger.json',self.ledger)
                result=reward.check_calibration(self.config,self.folder)
                self.assertFalse(result['passed']);self.assertLess(result['accepted_cases'],18)
                p.save(path,saved);self.ledger=original_ledger;p.save(self.run/'budget_ledger.json',self.ledger)

    def test_changed_registered_sources_or_nontraining_IDs_fail_before_paid_work(self):
        with patch.object(reward.requests,'post') as post:
            with self.assertRaises(AssertionError):reward.registered_cases(dict(self.config,train_ids=[23,139]),self.folder)
            path=self.folder/'cases/23_distinct.json';story=p.read(path);story['moral']='changed';p.save(path,story)
            with self.assertRaises(AssertionError):reward.source_paths(self.config,True)
            post.assert_not_called()

    def test_main_freezes_only_after_all_contrasts_and_never_replaces_a_frozen_protocol(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if Path(path).as_posix().endswith('/secrets/api_key.yaml'):return 'openai_api_key: unit-key'
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read),patch.object(reward.requests,'post') as post:
            reward.main(calibrate=True)
            frozen=p.read(self.run/'reward_protocol_frozen.json')
            self.assertEqual(frozen['acceptance_sha256'],reward.rag.sha(self.folder/'acceptance.json'))
            self.assertTrue(p.read(self.folder/'acceptance.json')['passed'])
            self.assertEqual(p.read(self.folder/'api_status.json')['stage'],'complete')
            with self.assertRaises(AssertionError):reward.main(calibrate=True)
            post.assert_not_called()


if __name__=='__main__':unittest.main()
