import copy
import importlib.util
import json
from pathlib import Path
import unittest

source=Path(__file__).parents[1]/'code/rlaif_temporal_review.py'
spec=importlib.util.spec_from_file_location('temporal_test_runtime',source)
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


class TemporalReview(unittest.TestCase):
    def setUp(self):
        self.story={'raw_text':json.dumps({'scenes':[{'text':'친구들이 다리를 건넜어요.'},{'text':'친구들이 처음 다리를 놓았어요.'}]}),'ended_with_eos':True}
        self.context={'references':[],'temporal_review':{'transitions':[]}}
        self.value={'relation':'before','flashback_signal_quote':'','A_quote':'다리를 건넜어요.','B_quote':'다리를 놓았어요.','explanation':'The bridge must be built before crossing.'}
        self.item={'positions':[1,2],'local_chronology':self.value}

    def body(self):
        return {'logprobs':True,'top_logprobs':20,'max_completion_tokens':3072,
            'messages':[{'role':'system','content':'rubric'},{'role':'user','content':json.dumps({'CANDIDATE_STORY_TO_RATE':json.loads(self.story['raw_text'])})}]}

    def test_verified_backward_evidence_reaches_joint_request(self):
        self.context['temporal_review']['transitions']=[self.item]
        body=review.joint_body(self.story,self.context,self.body());candidate=json.loads(body['messages'][-1]['content'])
        self.assertEqual(candidate['LOCAL_ERROR_EVIDENCE'][0]['earlier_position'],1)
        self.assertTrue(body['logprobs']);self.assertEqual(body['top_logprobs'],20)

    def test_missing_or_duplicate_coverage_rejected(self):
        for items in ([],[self.item,self.item]):
            self.context['temporal_review']['transitions']=items
            with self.assertRaises(AssertionError):review.joint_body(self.story,self.context,self.body())

    def test_invented_quote_rejected(self):
        value=dict(self.value,A_quote='내일 다시 만나기로 했어요.')
        with self.assertRaises(AssertionError):review.validate_transition(self.story,(1,2),value)

    def test_ungrounded_citation_is_distinct_from_invalid_schema(self):
        with self.assertRaises(review.UngroundedQuote):
            review.validate_transition(self.story,(1,2),dict(self.value,B_quote='다리를 놓았지요.'))
        with self.assertRaises(AssertionError) as error:
            review.validate_transition(self.story,(1,2),dict(self.value,B_quote=123))
        self.assertNotIsInstance(error.exception,review.UngroundedQuote)

    def test_unavailable_opinion_is_disclosed_without_repairing_story(self):
        missing={'positions':[1,2],'evidence_unavailable':{'reason':'literal_citation_unverified','field':'B_quote'}}
        self.context['temporal_review']['transitions']=[missing]
        body=review.joint_body(self.story,self.context,self.body())
        candidate=json.loads(body['messages'][-1]['content'])
        self.assertEqual(candidate['LOCAL_ERROR_EVIDENCE'],[])
        self.assertEqual(candidate['LOCAL_REVIEW_UNAVAILABLE'],[missing])
        self.assertIn('not evidence of coherence',candidate['LOCAL_REVIEW_WARNING'])
        self.assertEqual(candidate['CANDIDATE_STORY_TO_RATE'],json.loads(self.story['raw_text']))
        self.assertTrue(body['logprobs']);self.assertEqual(body['top_logprobs'],20)

    def test_unavailable_metadata_cannot_smuggle_unverified_opinions(self):
        missing={'positions':[1,2],'evidence_unavailable':{'reason':'literal_citation_unverified','field':'B_quote'},'local_chronology':self.value}
        self.context['temporal_review']['transitions']=[missing]
        with self.assertRaises(AssertionError):review.joint_body(self.story,self.context,self.body())

    def test_omission_keeps_literal_fragments_in_original_order(self):
        for separator in (' ... ','…'):
            self.assertTrue(review.literal_quote('친구들이'+separator+'건넜어요.','친구들이 다리를 건넜어요.'))

    def test_omission_cannot_reorder_literal_fragments(self):
        self.assertFalse(review.literal_quote('건넜어요. ... 친구들이','친구들이 다리를 건넜어요.'))

    def test_omission_cannot_invent_a_fragment(self):
        self.assertFalse(review.literal_quote('친구들이 ... 날아갔어요.','친구들이 다리를 건넜어요.'))

    def test_omission_cannot_borrow_from_another_scene(self):
        with self.assertRaises(AssertionError):
            review.validate_transition(self.story,(1,2),dict(self.value,A_quote='친구들이 ... 다리를 놓았어요.'))

    def test_empty_or_short_omission_fragments_rejected(self):
        for quote in ('... 친구들이','친구들이 ...','친구 ... 건넜어요.'):
            self.assertFalse(review.literal_quote(quote,'친구들이 다리를 건넜어요.'))

    def test_omitted_dialogue_tail_keeps_literal_closing_quote(self):
        text="다이스는 말했습니다. '하지만 꽃을 놓쳐버린다면, 호수는 우리를 기쁘게 해줄까요?'"
        for marker in ('...', '…'):
            self.assertTrue(review.literal_quote("다이스는 말했습니다. '하지만 꽃을 놓쳐버린다면"+marker+"'",text))

    def test_omission_cannot_invent_closing_punctuation(self):
        self.assertFalse(review.literal_quote("친구들이 ...'",'친구들이 다리를 건넜어요.'))

    def test_short_words_still_cannot_pass_as_closing_punctuation(self):
        self.assertFalse(review.literal_quote('친구들이 ... 다리','친구들이 다리를 건넜어요.'))

    def test_closing_quote_does_not_allow_invented_prefix(self):
        self.assertFalse(review.literal_quote("친구들이 날았어요 ...'","친구들이 다리를 건넜어요.'"))

    def setting_story(self):
        story=copy.deepcopy(self.story)
        value=json.loads(story['raw_text'])
        value['scenes'][0]['setting']='정원, 오전 10시.'
        value['scenes'][1]['setting']='정원, 오전 11시. 그전날'
        story['raw_text']=json.dumps(value,ensure_ascii=False)
        return story

    def test_quotes_may_use_actual_same_scene_setting(self):
        review.validate_transition(self.setting_story(),(1,2),dict(self.value,A_quote='정원, 오전 10시.',B_quote='정원, 오전 11시.'))

    def test_quote_cannot_mix_text_and_setting_fragments(self):
        with self.assertRaises(AssertionError):
            review.validate_transition(self.setting_story(),(1,2),dict(self.value,A_quote='친구들이 ... 정원, 오전 10시.'))

    def test_quote_cannot_borrow_other_scene_setting(self):
        with self.assertRaises(AssertionError):
            review.validate_transition(self.setting_story(),(1,2),dict(self.value,A_quote='정원, 오전 11시.'))

    def test_invented_setting_quote_rejected(self):
        with self.assertRaises(AssertionError):
            review.validate_transition(self.setting_story(),(1,2),dict(self.value,A_quote='정원, 오전 12시.'))

    def test_narrator_flashback_cue_still_requires_actual_B_text(self):
        with self.assertRaises(AssertionError):
            review.validate_transition(self.setting_story(),(1,2),dict(self.value,flashback_signal_quote='그전날'))

    def test_editorial_dialogue_closure_keeps_exact_words(self):
        for opening,closing in (("'","'"),('“','”')):
            text='오이는 생각했어요. '+opening+'판단해야겠어. 그래서 더 듣겠어.'+closing
            quote='오이는 생각했어요. '+opening+'판단해야겠어.'+closing
            self.assertTrue(review.literal_quote(quote,text))

    def test_editorial_closure_requires_actual_source_closure(self):
        self.assertFalse(review.literal_quote("오이는 생각했어요. '판단해야겠어.'","오이는 생각했어요. '판단해야겠어. 그래서 더 듣겠어."))

    def test_editorial_closure_cannot_rephrase_words(self):
        self.assertFalse(review.literal_quote("오이는 생각했어요. '무조건 믿겠어.'","오이는 생각했어요. '판단해야겠어. 그래서 더 듣겠어.'"))

    def test_editorial_closure_requires_unclosed_source_dialogue(self):
        self.assertFalse(review.literal_quote("오이는 '친구야.''","오이는 '친구야.' 그리고 기다렸어요."))

    def test_unmarked_paraphrase_and_two_dots_rejected(self):
        for quote in ('친구들이 다리를 즐겁게 건넜어요.','친구들이 .. 건넜어요.'):
            self.assertFalse(review.literal_quote(quote,'친구들이 다리를 건넜어요.'))

    def test_narrator_signal_must_exist_in_B(self):
        with self.assertRaises(AssertionError):review.validate_transition(self.story,(1,2),dict(self.value,flashback_signal_quote='그전날'))

    def test_explicit_flashback_evidence_is_not_automatically_a_failure(self):
        self.story['raw_text']=json.dumps({'scenes':[{'text':'친구들이 다리를 건넜어요.'},{'text':'그전날 친구들이 다리를 놓았어요.'}]})
        self.context['temporal_review']['transitions']=[{'positions':[1,2],'local_chronology':dict(self.value,flashback_signal_quote='그전날')}]
        candidate=json.loads(review.joint_body(self.story,self.context,self.body())['messages'][-1]['content'])
        self.assertEqual(candidate['LOCAL_ERROR_EVIDENCE'],[])

    def test_nonadjacent_pair_rejected(self):
        with self.assertRaises(AssertionError):review.validate_transition(self.story,(1,3),self.value)

    def test_unfinished_and_wordless_skip_unavailable_transitions(self):
        self.story['ended_with_eos']=False;self.assertEqual(review.pair_positions(self.story,self.context),[])
        candidate=json.loads(review.joint_body(self.story,self.context,self.body())['messages'][-1]['content'])
        self.assertIn('MUST be1',candidate['UNFINISHED_SCORE_PRIORITY'])
        self.story.update(ended_with_eos=True,raw_text='{"scenes":[{"text":""},{"text":"친구들이 다리를 놓았어요."}]}')
        self.assertEqual(review.pair_positions(self.story,self.context),[])

    def test_whole_text_copy_is_literal_evidence_and_short_common_text_is_not(self):
        self.story['raw_text']=json.dumps({'scenes':[{'text':'서로의 이야기를 끝까지 듣고 함께 새로운 다리를 만들었어요.'*3}]})
        self.context['references']=[{'source_id':1,'doc_id':'train_1_full','text':self.story['raw_text']}]
        self.assertTrue(review.machine_facts(self.story,self.context)['exact_full_scene_text_matches'])
        self.story['raw_text']='{"scenes":[{"text":"함께 놀았어요."}]}'
        self.context['references'][0]['text']=self.story['raw_text']
        self.assertFalse(review.machine_facts(self.story,self.context)['exact_full_scene_text_matches'])

    def test_wrong_model_or_unfinished_response_rejected(self):
        for model,reason in (('other','stop'),('gpt-4.1-2025-04-14','length')):
            raw={'model':model,'choices':[{'finish_reason':reason,'message':{'content':json.dumps(self.value)}}]}
            with self.assertRaises(AssertionError):review.parse_transition(raw,self.story,(1,2))


if __name__=='__main__':unittest.main()
