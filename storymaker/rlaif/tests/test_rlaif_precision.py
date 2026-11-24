"""CPU regression for production precision, tiled vocabulary and bounded gradients; no API/GPU calls."""
import unittest
import torch
from torch.nn import functional as F
from storymaker.rlaif.code.rlaif_precision import enable, projection, MODE

def model_check():
    import copy
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(2)
    torch.manual_seed(61)
    config = Qwen3_5Config(tie_word_embeddings=True,
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=['linear_attention', 'full_attention'], tie_word_embeddings=True, eos_token_id=2, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            out_hidden_size=32, num_position_embeddings=64))
    model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    reference = copy.deepcopy(model).float()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    legacy=copy.deepcopy(model)
    with torch.no_grad():
        legacy_before=legacy(input_ids=torch.tensor([[3,4,5,6,7,8,9,2]]),use_cache=False).logits.clone()
    names={name:id(value) for name,value in model.named_parameters()}
    enable(model,MODE)
    assert names=={name:id(value) for name,value in model.named_parameters()}
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
    actual = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    expected = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.dtype == torch.float32
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    with torch.no_grad():
        own = model(input_ids=ids[:, :3], use_cache=True)
        ref = reference(input_ids=ids[:, :3], use_cache=True)
        for position in range(3, ids.shape[1]):
            own = model(input_ids=ids[:, position:position+1], past_key_values=own.past_key_values, use_cache=True)
            ref = reference(input_ids=ids[:, position:position+1], past_key_values=ref.past_key_values, use_cache=True)
            torch.testing.assert_close(own.logits, ref.logits, rtol=1e-6, atol=1e-6)
    F.cross_entropy(actual[0, :-1], ids[0, 1:]).backward()
    F.cross_entropy(expected[0, :-1], ids[0, 1:]).backward()
    grads, refs = [], []
    for name,param in model.named_parameters():
        ref=dict(reference.named_parameters())[name]
        assert param.dtype == torch.bfloat16
        assert (param.grad is None) == (ref.grad is None)
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            grads.append(param.grad.float().flatten()); refs.append(ref.grad.flatten())
    cosine = float(F.cosine_similarity(torch.cat(grads), torch.cat(refs), dim=0))
    assert cosine > .99998 and not torch.cuda.is_initialized()
    with torch.no_grad():
        assert torch.equal(legacy_before,legacy(input_ids=ids,use_cache=False).logits)
        assert enable(legacy,None) is legacy
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        loss=model(input_ids=ids,use_cache=False).logits.square().mean()
    loss.backward()
    assert not torch.cuda.is_initialized()
    return dict(tiny_mixed_attention_Qwen_FP32_logits_match=True,
        tiny_cached_decode_matches_native_FP32=True,
        source_weights_unchanged=True, BF16_parameter_gradients_finite=True,
        gradient_cosine_vs_native_FP32=cosine, cuda_initialized=False,
        scope='Synthetic tiny-model CPU check, not4B memory or actual RL validation')


class PrecisionChecks(unittest.TestCase):
    def test_mixed_qwen_cached_native_fp32_and_legacy_bf16(self):
        model_check()

    def test_bounded_mixed_gradient_accumulation_matches_joint_cpu_optimizer(self):
        import tempfile
        from pathlib import Path
        from storymaker.rlaif.code.rlaif_storage import GradientBuffer
        torch.manual_seed(5)
        parameters=[('a',torch.nn.Parameter(torch.randn(7,9,dtype=torch.bfloat16))),
                    ('b',torch.nn.Parameter(torch.randn(9)))]
        references=[torch.nn.Parameter(value.detach().float().clone()) for _,value in parameters]
        with tempfile.TemporaryDirectory() as directory:
            buffer=GradientBuffer(Path(directory),parameters);buffer.install()
            (parameters[0][1].sum()*2+parameters[1][1].sum()*3).backward()
            (parameters[0][1].sum()*5+parameters[1][1].sum()*7).backward()
            self.assertAlmostEqual(buffer.finish(),(63*49+9*100)**.5)
            joint=torch.optim.Adafactor(references,lr=1e-3,foreach=False)
            for value,constant in zip(references,[7,10]):value.grad=torch.full_like(value,constant)
            joint.step();joint.zero_grad(set_to_none=True)
            for (name,source),reference in zip(parameters,references):
                master=torch.nn.Parameter(source.detach().float().clone());master.grad=buffer.view(name)
                optimizer=torch.optim.Adafactor([master],lr=1e-3,foreach=False)
                optimizer.step();optimizer.zero_grad(set_to_none=True);buffer.release(name)
                self.assertTrue(torch.equal(master,reference))
            self.assertFalse(buffer.maps);self.assertEqual(buffer.peak_mapped_bytes,63*4)
            buffer.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_tiled_projection_preserves_tail_forward_and_gradients(self):
        torch.manual_seed(3)
        h=torch.randn(2,4,requires_grad=True)
        w=torch.randn(65537,4,dtype=torch.bfloat16,requires_grad=True)
        reference=w.detach().float().requires_grad_()
        actual=projection(h,w);expected=F.linear(h,reference)
        self.assertTrue(torch.equal(actual,expected))
        actual.square().mean().backward();expected.square().mean().backward()
        self.assertTrue(torch.isfinite(w.grad).all())
        self.assertGreater(float(F.cosine_similarity(w.grad.float().flatten(),reference.grad.flatten(),dim=0)),.99998)
        self.assertTrue(torch.equal(projection(h.detach().to(torch.bfloat16),w.detach()),F.linear(h.detach().to(torch.bfloat16),w.detach()).float()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
