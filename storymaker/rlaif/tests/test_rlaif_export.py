"""Real tiny-Qwen CPU roundtrip and fail-closed final-selection checks; no production export."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForImageTextToText, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from storymaker import project as p
from storymaker.rlaif.code import rlaif_export as export
from storymaker.rlaif.code.rlaif_export import save_and_verify, sha, verify_selection


def tiny():
    config = Qwen3_5Config(tie_word_embeddings=True,
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16, layer_types=['full_attention'],
            tie_word_embeddings=True, eos_token_id=2, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            out_hidden_size=32, num_position_embeddings=64))
    return Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()


def selected_fixture(root, source=None):
    source = root if source is None else source
    p.save(source/'config.json', {'learning_rate':1e-5,'logical_batch_size':8})
    if source != root:
        p.save(root/'config.json', {'source_previous_run':str(source),'source_previous_config_sha256':sha(source/'config.json'),
            'policy_checkpoint':str(source/'checkpoints/update_0001/policy'),
            'value_checkpoint':str(source/'checkpoints/update_0001/value'),'logical_batch_size':64})
    checkpoint = source/'checkpoints/update_0001/policy'
    batch = source/'batch_0000'
    roles={role:source/'checkpoints/update_0001'/role for role in ('policy','value')}
    p.save(batch/'joint_update_verified.json',{'policy':str(roles['policy']),'value':str(roles['value']),
        'committed_policy_updates':1,'committed_value_updates':1,'trajectories':8})
    for role,path in roles.items():
        p.save(path/'manifest.json', {'parameters': [], 'changed_tensors':1,'fixture_only': True})
        p.save(batch/f'update_{role}_verified.json',{'phase':role,'config_sha256':sha(source/'config.json'),
            'checkpoint':str(path),'trajectories':8,'optimizer_steps':1,'saved_and_readback_verified':True,'changed_tensors':1})
    prompts = root/'evaluation/final_holdout/prompts.json'
    p.save(prompts, {'prompts': [{'id': 'F'+str(i)} for i in range(32)], 'fixture_only': True})
    p.save(root/'evaluation/evaluation_plan.json', {'final_prompt_file': str(prompts),
        'final_prompt_sha256': sha(prompts)})
    validation = root/'evaluation/reward_metrics/validation_update_0001/summary.json'
    p.save(validation, {'phase': 'validation', 'policy_tag': 'update_0001', 'prompt_count': 43})
    p.save(root/'evaluation/reward_metrics/final_update_0001/summary.json',
        {'phase': 'final', 'policy_tag': 'update_0001', 'prompt_count': 32})
    p.save(root/'evaluation/final_policy_selection.json', {'chosen_tag': 'update_0001',
        'chosen_policy_manifest_sha256': sha(checkpoint/'manifest.json'),
        'validation_summary_sha256': sha(validation), 'selection_used_final_outputs': False,
        'final_prompt_sha256': sha(prompts)})
    return checkpoint


class ExportChecks(unittest.TestCase):
    def test_old_seven_only_report_cannot_replace_the_primary_reward_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);selected_fixture(root)
            for phase in ('validation','final'):
                source=root/f'evaluation/reward_metrics/{phase}_update_0001/summary.json'
                p.save(root/f'evaluation/quality/{phase}_update_0001/summary.json',p.read(source))
                source.unlink()
            with self.assertRaises(FileNotFoundError):verify_selection(root)

    def test_retained_checkpoint_uses_its_actual_training_config_not_pending64_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'current';old=Path(temporary)/'old'
            checkpoint=selected_fixture(root,old)
            actual,identity=verify_selection(root)
            self.assertEqual(actual,checkpoint.resolve())
            self.assertEqual(Path(identity['training_config_path']),old/'config.json')
            self.assertEqual(p.read(identity['training_config_path'])['logical_batch_size'],8)
            self.assertEqual(p.read(root/'config.json')['logical_batch_size'],64)

    def test_changed_round_config_or_uncommitted_value_cannot_be_exported(self):
        for problem in ('config','value'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);selected_fixture(root)
                if problem=='config':p.save(root/'config.json',{'learning_rate':999})
                else:
                    path=root/'batch_0000/update_value_verified.json'
                    receipt=p.read(path);receipt['saved_and_readback_verified']=False;p.save(path,receipt)
                with self.assertRaises(AssertionError):verify_selection(root)

    def test_round_snapshot_preserves_training_provenance_after_next_config_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=selected_fixture(root)
            snapshot=root/'batch_0000/config.json'
            p.save(snapshot,p.read(root/'config.json'))
            p.save(root/'config.json',{'logical_batch_size':64,'learning_rate':9e-6})
            actual,identity=verify_selection(root)
            self.assertEqual(actual,checkpoint.resolve())
            self.assertEqual(Path(identity['training_config_path']),snapshot)
            self.assertEqual(p.read(snapshot)['logical_batch_size'],8)

    def test_tiny_qwen_standard_save_reload_preserves_tied_weights_and_logits(self):
        torch.manual_seed(42)
        model = tiny()
        model.generation_config.temperature = .7
        model.generation_config.top_p = .9
        model.generation_config.do_sample = True
        with torch.no_grad():
            model.model.language_model.embed_tokens.weight[3, 4] += .125
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'export'
            receipt = save_and_verify(model, folder)
            self.assertTrue(receipt['all_saved_weights_readback_equal_and_finite'])
            restored = AutoModelForImageTextToText.from_pretrained(folder, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').eval()
            self.assertIs(restored.lm_head.weight, restored.model.language_model.embed_tokens.weight)
            self.assertEqual(restored.generation_config.temperature, .7)
            self.assertEqual(restored.generation_config.top_p, .9)
            self.assertTrue(restored.generation_config.do_sample)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            with torch.no_grad():
                ids = torch.tensor([[3, 4, 5, 2]])
                self.assertTrue(torch.equal(model(input_ids=ids, use_cache=False).logits,
                    restored(input_ids=ids, use_cache=False).logits))
            self.assertFalse(torch.cuda.is_initialized())

    def test_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'export'
            folder.mkdir();(folder/'keep.txt').write_text('preserve', encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'Never overwrite'):
                save_and_verify(tiny(), folder)
            self.assertEqual((folder/'keep.txt').read_text('utf-8'), 'preserve')

    def test_nonfinite_export_is_not_verified(self):
        model = tiny()
        with torch.no_grad():
            model.model.language_model.embed_tokens.weight[3, 4] = float('nan')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'export'
            with self.assertRaises(AssertionError):
                save_and_verify(model, folder)
            self.assertFalse((folder/'export_verified.json').exists())

    def test_selected_checkpoint_and_reports_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = selected_fixture(root)
            self.assertEqual(verify_selection(root)[0], checkpoint.resolve())
            p.save(checkpoint/'manifest.json', {'modified': True})
            with self.assertRaises(AssertionError):
                verify_selection(root)

    def test_final_outputs_cannot_be_used_for_selection_or_old_test_as_final(self):
        for problem in ['selection', 'count']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);selected_fixture(root)
                path = root/('evaluation/final_policy_selection.json' if problem == 'selection'
                    else 'evaluation/reward_metrics/final_update_0001/summary.json')
                item = p.read(path)
                item['selection_used_final_outputs' if problem == 'selection' else 'prompt_count'] = True if problem == 'selection' else 61
                p.save(path, item)
                with self.assertRaises(AssertionError):
                    verify_selection(root)

    def test_live_budget_writer_or_unknown_charge_blocks_model_loading(self):
        for problem in ['GPU_busy', 'API_busy', 'charge_unknown']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary)
                root = project/'rlaif/results/current'
                source = project/'source'
                source.mkdir();(source/'weights.safetensors').write_bytes(b'fixture')
                p.save(source/'model.safetensors.index.json', {'weight_map': {'x': 'weights.safetensors'}})
                p.save(root/'control.json', {'enabled': True})
                p.save(root/'config.json', {'model': str(source)})
                p.save(root/'budget_ledger.json', {'calls': [{'state': 'charge_unknown'}]})
                effects = [OSError('GPU busy')] if problem == 'GPU_busy' else [None,OSError('API busy')] if problem=='API_busy' else [None,None]
                with patch.object(export, 'OUT', root), patch.object(p, 'ROOT', project), \
                        patch.object(export,'GPU_LOCK',project/'rlaif/results/generation.lock'), \
                        patch.object(export, 'verify_selection', return_value=(root/'checkpoint', {})), \
                        patch.object(export.shutil, 'disk_usage') as disk, \
                        patch.object(export.msvcrt, 'locking', side_effect=effects), \
                        patch('sys.argv', ['storymaker.rlaif.code.rlaif_export.py', '--execute']), \
                        patch.object(AutoModelForImageTextToText, 'from_pretrained') as load, \
                        patch.object(export.subprocess, 'check_output') as inspect_memory:
                    disk.return_value.free = 10**12
                    with self.assertRaises((OSError, AssertionError)):
                        export.main()
                    load.assert_not_called()
                    inspect_memory.assert_not_called()
                self.assertFalse((root/'final_model').exists())
                self.assertTrue((project/'rlaif/results/generation.lock').exists())
                self.assertFalse((project/'outputs/generation.lock').exists())


if __name__ == '__main__':
    unittest.main()
