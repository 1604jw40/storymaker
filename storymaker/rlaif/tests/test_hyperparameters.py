"""Actual batch size, generation settings and resumed optimizer configuration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from storymaker.rlaif.code import rlaif_train as training
from storymaker.rlaif.code import rlaif_generate as generation
from storymaker.rlaif.code.rlaif_storage import GradientBuffer, optimizer_step, read_parameter


class HyperparameterChecks(unittest.TestCase):
    def records(self, root, count):
        batch = root / 'batch_0000'
        (batch / 'rollouts').mkdir(parents=True)
        (batch / 'rewards').mkdir()
        config = {'logical_batch_size': 64, 'batch_ids': list(range(1, count + 1)),
                  'train_ids': list(range(1, 201)), 'validation_ids': [201],
                  'excluded_test_ids': [202], 'policy_version': 'update_0001', 'max_new_tokens': 4096}
        (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        signature = training.file_digest(root / 'config.json')
        for index, source_id in enumerate(config['batch_ids']):
            key = f'{index:03}_{source_id}'
            path = batch / 'rollouts' / (key + '.json')
            path.write_text(json.dumps({'identity': {'source_id': source_id, 'policy_version': 'update_0001',
                'config_sha256': signature}, 'generated_token_ids': [1, 2]}), encoding='utf-8')
            folder = batch / 'rewards' / key
            folder.mkdir()
            (folder / 'reward.json').write_text(json.dumps({'source_sha256': training.file_digest(path),
                'certified_decimal_places': 6, 'normalized_reward': .5}), encoding='utf-8')
        return batch, config

    def test_actual_64_and_last_8_are_accepted(self):
        for count in [64, 8]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                batch, config = self.records(root, count)
                with patch.object(training, 'OUT', root):
                    self.assertEqual(len(training.records_for(batch, config)), count)

    def test_generation_uses_selected_values_and_preserves_pad_zero(self):
        config = {'max_new_tokens': 1234, 'temperature': .7, 'top_p': 1., 'top_k': 0, 'repetition_penalty': 1.}
        options = generation.generation_options(config, SimpleNamespace(eos_token_id=2, pad_token_id=0))
        self.assertEqual(options['max_new_tokens'], 1234)
        self.assertEqual(options['temperature'], .7)
        self.assertEqual(options['pad_token_id'], 0)
        self.assertTrue(options['do_sample'])
        with self.assertRaises(AssertionError):
            generation.generation_options(dict(config, top_p=.9), SimpleNamespace(eos_token_id=2, pad_token_id=0))

    def test_resumed_optimizer_honours_new_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter = torch.nn.Parameter(torch.tensor([[.2, -.7], [.3, .4]]))
            gradients = []
            try:
                for step, rate in [(1, 1e-3), (2, 1e-4)]:
                    buffer = GradientBuffer(root / f'gradient{step}', [('weight', parameter)])
                    gradients.append(buffer)
                    buffer.install()
                    parameter.sum().backward()
                    buffer.finish()
                    manifest = optimizer_step([('weight', parameter)], buffer, root / f'checkpoint{step}', rate,
                        previous=root / 'checkpoint1' if step == 2 else None)
                    _, state = read_parameter(root / f'checkpoint{step}', manifest['parameters'][0])
                    self.assertEqual(state['param_groups'][0]['lr'], rate)
                    self.assertEqual(int(state['state'][0]['step']), step)
                    buffer.close()
            finally:
                for buffer in gradients:
                    buffer.close()


if __name__ == '__main__':
    unittest.main()
