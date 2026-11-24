import copy
import itertools
import json
import unittest

from storymaker.rlaif.code import rlaif_reward_pipeline as reward
from storymaker.rlaif.tests import test_reward_pipeline as fixtures


class DiagnosisOrder(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.RewardPipeline();fixture.setUp()
        book=copy.deepcopy(fixture.book)
        book['scenes']=[dict(book['scenes'][0],scene_no=i) for i in [4,2,9]]
        self.story=dict(fixture.story,raw_text=json.dumps(book,ensure_ascii=False))
        self.context=fixture.context
        self.value=copy.deepcopy(fixture.diagnosis['diagnostics'])
        for metric in [self.value['novelty'],self.value['completion'],*self.value['content'].values()]:metric['scene_nos']=[4]
        self.value['content']['causality']['score']=1
        self.value['event_order']=[{'scene_no':4,'prior':[2],'conflict':True},
                                  {'scene_no':2,'prior':[],'conflict':False},
                                  {'scene_no':9,'prior':[2],'conflict':False}]

    def decode(self,value):
        raw={'model':'gpt-4.1-2025-04-14','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}
        return reward.diag_decode(raw,self.story,self.context)

    def test_every_complete_permutation_keeps_all_judgments_and_source_order(self):
        expected=self.decode(self.value)
        for order in itertools.permutations(self.value['event_order']):
            value=copy.deepcopy(self.value);value['event_order']=list(order)
            with self.subTest(order=[r['scene_no'] for r in order]):
                self.assertEqual(self.decode(value),expected)
                self.assertEqual(reward.ceiling(self.story,self.decode(value)['diagnostics']),reward.ceiling(self.story,expected['diagnostics']))

    def test_missing_duplicate_unknown_and_noninteger_scene_ids_are_rejected(self):
        for kind in ['missing','duplicate','unknown','bool']:
            value=copy.deepcopy(self.value)
            if kind=='missing':value['event_order'].pop()
            elif kind=='duplicate':value['event_order'][1]['scene_no']=4
            elif kind=='unknown':value['event_order'][1]['scene_no']=7
            else:value['event_order'][1]['scene_no']=True
            with self.subTest(kind=kind),self.assertRaises(AssertionError):self.decode(value)
