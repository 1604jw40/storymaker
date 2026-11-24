"""Small CPU fixtures for full-entry change counts, roundoff loss and corrupt-checkpoint refusal."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from storymaker import project as p
from storymaker.rlaif.code.rlaif_storage import digest_tensor, save_parameter
from storymaker.rlaif.code.rlaif_update_audit import audit_checkpoint, compare_parameter
from storymaker.rlaif.code.rlaif_train import model_digest


def fixture(root, with_head=False):
    model, checkpoint = root/'source', root/'checkpoint'
    model.mkdir();checkpoint.mkdir()
    original = torch.ones(2, 3, dtype=torch.bfloat16)
    key = 'model.language_model.test.weight'
    save_file({key: original}, model/'model.safetensors')
    p.save(model/'model.safetensors.index.json', {'weight_map': {key: 'model.safetensors'}})
    updated = original.float()
    updated[0, 0] += .0001
    updated[0, 1] += .01
    updated[1, 0] -= .01
    values = [('test.weight', updated, torch.bfloat16)]
    head = None
    if with_head:
        head = root/'initial_head.pt'
        torch.save({'bias': torch.ones(1)}, head)
        values.append(('value_head.bias', torch.tensor([1.000001]), torch.float32))
    records = []
    for name, value, dtype in values:
        parameter = torch.nn.Parameter(value.clone())
        optimizer = torch.optim.Adafactor([parameter], lr=1e-5, foreach=False)
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        records.append(save_parameter(checkpoint, name, parameter, optimizer, digest_tensor(parameter.to(dtype))))
    # A fixture cut crosses a matrix row; all entries must still match the original layout.
    metadata_path = checkpoint/records[0]['file']
    metadata = torch.load(metadata_path, weights_only=True)
    chunks = []
    for index, part in enumerate([updated.flatten()[:2], updated.flatten()[2:]]):
        path = checkpoint/f'fixture_part_{index}.pt'
        torch.save(part.clone(), path)
        chunks.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    metadata['chunks'] = chunks
    torch.save(metadata, metadata_path)
    records[0]['metadata_sha256'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    torch.save({'fixture_only': True}, checkpoint/'rng.pt')
    p.save(checkpoint/'manifest.json', {'parameters': records, 'changed_tensors': len(values), 'previous': None,
        'rng_sha256': hashlib.sha256((checkpoint/'rng.pt').read_bytes()).hexdigest(), 'fixture_only': True})
    return model, checkpoint, head


class UpdateAuditChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_small_fp32_changes_lost_to_bf16_are_counted_separately(self):
        original = torch.ones(2, 3, dtype=torch.bfloat16)
        updated = original.float();updated[0, 0] += .0001;updated[1, 2] += .01
        result = compare_parameter(updated, original, original.shape, torch.bfloat16)
        self.assertEqual(result['elements'], 6)
        self.assertEqual(result['fp32_changed_elements'], 2)
        self.assertEqual(result['stored_changed_elements'], 1)
        expected = float((updated.double()-original.double()).square().mean().sqrt())
        self.assertAlmostEqual(result['fp32_delta_rms'], expected)
        self.assertEqual(result['stored_sha256'], digest_tensor(updated.to(torch.bfloat16)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_checkpoint_rechunk_and_fp32_value_head_are_verified_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, checkpoint, head = fixture(root, with_head=True)
            before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob('*') if path.is_file()}
            result = audit_checkpoint(checkpoint, model, head)
            self.assertEqual(result['totals']['elements'], 7)
            self.assertEqual(result['totals']['fp32_changed_elements'], 4)
            self.assertEqual(result['totals']['stored_changed_elements'], 3)
            self.assertEqual(result['new_optimizer_steps'], 0)
            self.assertEqual(result['initial_model_sha256'], model_digest([
                ('test.weight', torch.ones(2, 3, dtype=torch.bfloat16)), ('value_head.bias', torch.ones(1))]))
            self.assertEqual(result['paid_API_calls'], 0)
            row = next(x for x in result['parameters'] if x['name']=='value_head.bias')
            self.assertEqual(row['stored_dtype'], 'torch.float32')
            self.assertEqual(row['stored_changed_elements'], 1)
            self.assertAlmostEqual(row['adafactor_rho'], 1e-5)
            self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in root.rglob('*') if path.is_file()})
            self.assertFalse(torch.cuda.is_initialized())

    def test_corrupt_parameter_chunk_and_missing_trainable_name_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model, checkpoint, _ = fixture(Path(directory))
            path = checkpoint/'fixture_part_0.pt'
            raw = path.read_bytes();path.write_bytes(raw+b'corrupt')
            with self.assertRaises(AssertionError):audit_checkpoint(checkpoint, model)
            with self.assertRaises(AssertionError):
                compare_parameter(torch.full((1,), torch.finfo(torch.float32).max),
                    torch.zeros(1, dtype=torch.bfloat16), [1], torch.bfloat16)
            path.write_bytes(raw)
            manifest = p.read(checkpoint/'manifest.json');manifest['parameters'] = []
            p.save(checkpoint/'manifest.json', manifest)
            with self.assertRaises(AssertionError):audit_checkpoint(checkpoint, model)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
