import math
from storymaker.qwen.code.pre_rl_ce import aggregate


def test_token_weighted_ce_and_eos_are_separate():
    rows = [{'variant': 'base', 'split': 'test', 'ce': 1., 'supervised_tokens': 1,
             'eos_nll': 0., 'eos_rank': 1},
            {'variant': 'base', 'split': 'test', 'ce': 3., 'supervised_tokens': 3,
             'eos_nll': 2., 'eos_rank': 2}]
    group = aggregate(rows)['base/test']
    assert group['token_weighted_ce'] == 2.5
    assert group['story_mean_ce'] == 2.
    assert group['conditional_perplexity'] == math.exp(2.5)
    assert group['gold_prefix_eos_top1_count'] == 1
    assert group['gold_prefix_eos_mean_nll'] == 1.
