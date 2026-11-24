"""Trust boundaries for the fixed-criteria split reward; no network/GPU calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from storymaker.rlaif.code import rlaif_reward_pipeline as reward
from storymaker.rlaif.code.rlaif_learning_signal import compare
from storymaker.rlaif.code.rlaif_evaluate import reference_record
from storymaker.rlaif.code import rlaif_cycle


class RewardPipeline(unittest.TestCase):
    def setUp(self):
        self.book = {'title': '도움', 'characters': [{'character_id': 'a', 'name': '보리', 'goal': '돕기', 'traits': '친절함'}],
                     'scenes': [{'scene_no': 1, 'character_ids': ['a'], 'setting': '숲', 'text_mode': 'read_aloud',
                                 'text': '보리는 길에 떨어진 물건을 보고 주인을 찾아 길을 걸어갔어요. 이웃에게 물어보고 작은 집의 문을 두드렸어요. 물건을 잃고 울던 친구는 문을 열어 고맙다고 인사했어요. 보리는 친구와 함께 물건을 제자리에 놓고 저녁이 되기 전에 집으로 돌아왔어요.', 'visual_note': '보리가 물건을 들고 걸어요.'}]}
        self.story = {'raw_text': json.dumps(self.book, ensure_ascii=False), 'ended_with_eos': True}
        self.context = {'references': [{'source_id': 1, 'doc_id': 'train_1_full', 'text': self.story['raw_text']}], 'train_only': True}
        metric = {'score': 4, 'scene_nos': [1], 'evidence': '해당 장면의 근거'}
        self.diagnosis = {'diagnostics': {'event_order': [{'scene_no': 1, 'prior': [], 'conflict': False}],
            'novelty': copy.deepcopy(metric), 'completion': copy.deepcopy(metric),
            'content': {key: copy.deepcopy(metric) for key in reward.api.CRITERIA}, 'citation_ids': ['e1'], 'summary': '근거'},
            'semantic_consistency_notes': []}

    def test_name_only_copy_and_literal_audit(self):
        renamed = dict(self.story, raw_text=self.story['raw_text'].replace('보리', '누리'))
        self.assertEqual(reward.name_matches(renamed, self.context)[0]['source_id'], 1)
        with self.assertRaises(AssertionError):
            reward.grounded({'issues': [{'kind': 'language', 'evidence_ids': ['c9999'], 'explanation': 'wrong ID'}]}, self.story)
        with self.assertRaises(AssertionError):
            reward.validate_audit({'issues': [{'kind': 'language', 'quotes': [{'scene_no': 1, 'field': 'text', 'quote': '없는 인용문입니다'}], 'explanation': 'bad'}]}, self.story)

    def test_unparsed_raw_opinions_are_quarantined_not_promoted(self):
        story = dict(self.story, raw_text=self.story['raw_text'][:-1])
        issue = lambda kind, key: {'kind': kind, 'evidence_ids': [key], 'explanation': 'provider opinion'}
        value = {'issues': [issue('format', 'c1'), issue('language', 'c1'), issue('repetition', 'c1')]}
        raw = {'model': 'gpt-4.1-2025-04-14', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        result = reward.audit_decode(raw, story, self.context)
        self.assertEqual(result['selected_evidence'], value)
        self.assertEqual([i['kind'] for i in result['source_audit']['issues']], ['format'])
        self.assertEqual([i['kind'] for i in result['source_audit']['unverified_issues']], ['language', 'repetition'])
        self.assertEqual(story['raw_text'], self.story['raw_text'][:-1])
        with self.assertRaises(reward.AuditFieldMismatch):
            reward.grounded({'issues': [issue('language', 'c1')]}, story)
        raw['choices'][0]['message']['content'] = json.dumps({'issues': [issue('language', 'c999999')]})
        with self.assertRaises(AssertionError):
            reward.audit_decode(raw, story, self.context)

    def test_parsed_text_audit_keeps_existing_result_shape(self):
        key = next(r['evidence_id'] for r in reward.catalog(self.story) if r['field'] == 'text')
        value = {'issues': [{'kind': 'language', 'evidence_ids': [key], 'explanation': 'provider opinion'}]}
        raw = {'model': 'gpt-4.1-2025-04-14', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        expected = {'source_audit': reward.grounded(value, self.story), 'selected_evidence': value}
        self.assertEqual(reward.audit_decode(raw, self.story, self.context), expected)

    def test_auxiliary_role_mismatch_is_not_reclassified_as_a_fact(self):
        catalog = reward.catalog(self.story)
        text = next(r['evidence_id'] for r in catalog if r['field'] == 'text')
        visual = next(r['evidence_id'] for r in catalog if r['field'] == 'visual_note')
        issues = [{'kind': 'causality', 'evidence_ids': [text, visual], 'explanation': 'wrong role'},
                  {'kind': 'text_visual', 'evidence_ids': [text], 'explanation': 'missing visual'}]
        value = {'issues': issues}
        raw = {'model': 'gpt-4.1-2025-04-14', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        decoded = reward.audit_decode(raw, self.story, self.context)
        self.assertEqual(decoded['source_audit']['issues'], [])
        self.assertEqual([i['kind'] for i in decoded['source_audit']['unverified_issues']], ['causality', 'text_visual'])
        self.assertEqual(decoded['selected_evidence'], value)
        for issue in issues:
            with self.assertRaises(reward.AuditFieldMismatch):
                reward.grounded({'issues': [issue]}, self.story)
        book = copy.deepcopy(self.book); second = copy.deepcopy(book['scenes'][0]); second['scene_no'] = 2; book['scenes'].append(second)
        story = dict(self.story, raw_text=json.dumps(book, ensure_ascii=False))
        keys = [next(r['evidence_id'] for r in reward.catalog(story) if r['field'] == 'text' and r['scene_no'] == i) for i in [1, 2]]
        valid = {'issues': [{'kind': 'causality', 'evidence_ids': keys, 'explanation': 'supported roles'}]}
        self.assertEqual(len(reward.grounded(valid, story)['issues']), 1)

    def test_numeric_request_uses_the_existing_JSON_guard_limit(self):
        diagnosis = self.diagnosis['diagnostics']
        bad = dict(self.story, raw_text=self.story['raw_text'][:-1])
        self.assertEqual(reward.json_ceiling(bad), 1)
        payload = json.loads(reward.attach({'messages': [{'content': '{}'}]}, bad, diagnosis)['messages'][-1]['content'])
        self.assertEqual(payload['FIXED_DIAGNOSTIC_SCORE_ANCHORS']['maximum_selected_score'], 1)
        self.assertIn('JSON syntax failure =>score1', payload['DIAGNOSTIC_ANCHOR_DEFINITION'])
        malformed_type = copy.deepcopy(self.book); malformed_type['characters'][0]['traits'] = ['array']
        bad_type = dict(self.story, raw_text=json.dumps(malformed_type, ensure_ascii=False))
        self.assertEqual(reward.json_ceiling(bad_type), 4)
        self.assertEqual(reward.ceiling(bad_type, diagnosis)['maximum_selected_score'], 4)
        self.assertEqual(reward.json_ceiling(dict(self.story, raw_text='null')), 4)
        self.assertEqual(reward.ceiling(dict(bad, ended_with_eos=False), diagnosis)['reasons'], ['actual EOS=false'])

    def test_corrected_numeric_request_keeps_the_inspected_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory); old = {'score_rule': 'old'}; new = {'score_rule': 'corrected'}
            reward.p.save(dest/'numeric/request.json', old)
            reward.p.save(dest/'numeric/response.json', {'preserved': True})
            original = (dest/'numeric/request.json').read_bytes()
            self.assertEqual(reward.numeric_destination(dest, old), dest/'numeric')
            target = reward.numeric_destination(dest, new)
            self.assertNotEqual(target, dest/'numeric')
            self.assertTrue(target.is_relative_to(dest/'numeric'))
            self.assertEqual(target, reward.numeric_destination(dest, new))
            self.assertEqual((dest/'numeric/request.json').read_bytes(), original)
            self.assertEqual(reward.p.read(dest/'numeric/response.json'), {'preserved': True})

    def test_fixed_eleven_criteria_keep_story_refs(self):
        docs = [{'doc_id': str(i), 'kind': 'criterion', 'source_id': None, 'text': '기준 설명문입니다'} for i in range(11)]
        context = reward.fixed_context(self.context, docs)
        self.assertEqual(context['references'][0], self.context['references'][0])
        self.assertEqual(len(context['references']), 12)
        with self.assertRaises(AssertionError):
            reward.fixed_context(self.context, docs[:-1])

    def test_inconsistent_diagnosis_and_unfinished_high_score_stop(self):
        bad = dict(self.diagnosis, semantic_consistency_notes=['inconsistent'])
        with self.assertRaises(AssertionError):
            reward.combine(self.story, self.context, bad, {'selected_score': 8})
        with self.assertRaises(AssertionError):
            reward.combine(dict(self.story, ended_with_eos=False), self.context, self.diagnosis, {'selected_score': 8})

    def test_mandatory_type_error_keeps_readable_diagnosis_but_no_high_score(self):
        bad = copy.deepcopy(self.book)
        bad['characters'][0]['traits'] = ['친절함']
        story = dict(self.story, raw_text=json.dumps(bad, ensure_ascii=False))
        with self.assertRaises(AssertionError):
            reward.combine(story, self.context, self.diagnosis, {'selected_score': 8})
        self.assertEqual(reward.combine(story, self.context, self.diagnosis, {'selected_score': 4})['selected_score'], 4)

    def test_valid_json_null_is_a_type_failure_not_a_syntax_failure(self):
        story = dict(self.story, raw_text='null')
        diagnosis = copy.deepcopy(self.diagnosis)
        diagnosis['diagnostics']['event_order'] = []
        for metric in [diagnosis['diagnostics']['novelty'], diagnosis['diagnostics']['completion'], *diagnosis['diagnostics']['content'].values()]:
            metric['scene_nos'] = []
        self.assertEqual(reward.combine(story, self.context, diagnosis, {'selected_score': 4})['selected_score'], 4)

    def test_fresh_critic_guard_uses_projected_policy_baseline(self):
        self.assertEqual(compare(torch.tensor([-.5, -.5]), -.5), {'raw': 0., 'projected': 0., 'zero': .5})
        self.assertGreater(compare(torch.tensor([.5, .5]), -.5)['projected'], .5)
        self.assertEqual(compare(torch.tensor([-2., -2.]), -1)['projected'], 0)
        with self.assertRaises(AssertionError):
            compare(torch.tensor([float('nan')]), .2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_revised_prompt_preserves_every_gold_label(self):
        old = {'input_ids': [1, 2, 20, 21, 99, 10], 'labels': [-100, -100, 20, 21, 99, 10], 'prompt_tokens': 2}
        new = reference_record(old, [3, 4, 5])
        self.assertEqual(new['input_ids'], [3, 4, 5, 20, 21, 99, 10])
        self.assertEqual(new['labels'], [-100, -100, -100, 20, 21, 99, 10])
        self.assertEqual(reference_record(old, [1, 2]), old)
        bad = copy.deepcopy(old)
        bad['labels'][-1] = 777
        with self.assertRaises(AssertionError):
            reference_record(bad, [3, 4, 5])

    def test_controlled_resume_checks_preserved_sources_and_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root/'pipeline'; folder.mkdir()
            prior = root/'old_state.json'
            reward.p.save(prior, {'stage': 'needs_attention', 'message': 'train_reward failed; preserve logs, no automatic restart'})
            reward.p.save(folder/'cycle_state.json', {'stage': 'needs_attention'})
            sources = []
            for i in range(16):
                source = root/'rollouts'/f'{i}.json'; reward.p.save(source, {'id': i})
                sources.append({'source': str(source), 'source_sha256': reward.api.rag.sha(source)})
            collection = root/'environment/collection_verified.json'; rag = root/'environment/train_RAG_verified.json'
            reward.p.save(collection, {'passed': True, 'N': 16, 'config_sha256': 'cfg', 'rows': sources})
            reward.p.save(rag, {'passed': True, 'contexts': 16, 'source_sha256': {r['source']: r['source_sha256'] for r in sources}})
            authorization = {'passed': True, 'resume_from': 'train_reward', 'config_sha256': 'cfg', 'runtime_code_sha256': {},
                'previous_state_file': str(prior), 'previous_state_sha256': reward.api.rag.sha(prior),
                'evidence_sha256': {str(f): reward.api.rag.sha(f) for f in [collection, rag]}}
            reward.p.save(folder/'resume_authorization.json', authorization)
            self.assertEqual(rlaif_cycle.resume_inputs(folder, 'cfg', {}), authorization)
            with self.assertRaises(AssertionError):
                rlaif_cycle.resume_inputs(folder, 'wrong-config', {})
            with self.assertRaises(AssertionError):
                rlaif_cycle.resume_inputs(folder, 'cfg', {'changed-code': 'x'})
            reward.p.save(folder/'cycle_state.json', {'stage': 'running_child'})
            with self.assertRaises(AssertionError):
                rlaif_cycle.resume_inputs(folder, 'cfg', {})
            reward.p.save(folder/'cycle_state.json', {'stage': 'needs_attention'})
            reward.p.save(Path(sources[0]['source']), {'id': 'changed'})
            with self.assertRaises(AssertionError):
                rlaif_cycle.resume_inputs(folder, 'cfg', {})

    def test_explicit_dispatch_and_cycle_has_previous_comparison(self):
        context = {'reward_protocol_kind': reward.PROTOCOL}
        with patch.object(reward, 'score', return_value={'r': .2}) as call:
            self.assertEqual(reward.api.score(Path('unused'), Path('unused'), 'unused', context), {'r': .2})
            self.assertEqual(call.call_count, 1)
        with patch.object(reward, 'verified_score', return_value={'r': .2}) as call:
            self.assertEqual(reward.api.verified_score(Path('unused'), Path('unused'), context), {'r': .2})
            self.assertEqual(call.call_count, 1)
        tasks = rlaif_cycle.commands({'update_index': 4, 'policy_checkpoint': 'C:/source/update_0003/policy'})
        self.assertEqual(len(tasks), 13)
        self.assertEqual(tasks[0][0], 'collection')
        self.assertIn('validation_previous', [r[0] for r in tasks])
        self.assertEqual(tasks[-1], ('current_reward', 'rlaif_reward_evaluate.py', ['--policy-tag', 'update_0004', '--execute']))


if __name__ == '__main__':
    unittest.main()
