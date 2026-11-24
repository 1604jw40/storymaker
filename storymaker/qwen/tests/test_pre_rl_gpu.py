from storymaker.qwen.code.pre_rl_gpu import ready


def test_hot_busy_or_unknown_gpu_is_not_ready():
    assert ready({'ready': True}, 65)
    assert not ready({'ready': True}, 80)
    assert not ready({'ready': False}, 40)
    assert not ready({'ready': True}, float('nan'))
