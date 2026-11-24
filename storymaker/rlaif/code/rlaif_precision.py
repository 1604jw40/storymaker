"""FP32 operations on unchanged BF16 language weights for the direct-RLAIF runner."""
import types

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

MODE = 'fp32_operations_bf16_weights'


def projection(hidden, weight):
    if hidden.dtype == weight.dtype:
        return F.linear(hidden, weight).float()
    assert hidden.dtype == torch.float32 and weight.dtype == torch.bfloat16
    # Tile vocabulary rows, preserving every class and every input token.
    return torch.cat([F.linear(hidden, part.float()) for part in weight.split(32768)], dim=-1)


def enable(model, mode):
    if mode is None:
        return model
    assert mode == MODE and not getattr(model, '_rlaif_precision_mode', None)
    assert all(value.dtype == torch.bfloat16 for value in model.model.language_model.parameters())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def linear(module, value):
        return F.linear(value.float(), module.weight.float(), None if module.bias is None else module.bias.float())

    for module in model.model.language_model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(linear, module)
    assert model.lm_head.bias is None
    model.lm_head.forward = types.MethodType(lambda module, value: projection(value, module.weight), model.lm_head)
    model.model.language_model.embed_tokens.register_forward_hook(lambda module, args, value: value.float())
    if not getattr(qwen.causal_conv1d_fn, '_rlaif_fp32_ops', False):
        old_conv, old_update = qwen.causal_conv1d_fn, qwen.causal_conv1d_update

        def conv(value, weight, bias=None, **kwargs):
            if value.dtype != torch.float32:
                return old_conv(value, weight, bias, **kwargs)
            return old_conv(value, weight.float(), None if bias is None else bias.float(), **kwargs)

        def update(value, state, weight, bias=None, activation=None):
            if value.dtype != torch.float32:
                return old_update(value, state, weight, bias, activation)
            assert state.dtype == torch.float32
            return old_update(value, state, weight.float(), None if bias is None else bias.float(), activation)

        conv._rlaif_fp32_ops = True
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = conv, update
    # HF generation resolves model.device from the first registered parameter.
    visual = model.model.visual
    del model.model.visual
    model.model.add_module('visual', visual)
    assert model.device == model.model.language_model.embed_tokens.weight.device
    model._rlaif_precision_mode = mode
    return model
