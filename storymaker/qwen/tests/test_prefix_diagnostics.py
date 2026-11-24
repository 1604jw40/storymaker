from storymaker.qwen.code.prefix_diagnostics import complete_text_fields, repeated


def test_escaped_strings_and_truncated_tail_are_not_invented():
    raw = '{"scenes":[{"text":"이름은 \\"별이\\"예요."},{"text":"아직 끝나지'
    assert complete_text_fields(raw) == ['이름은 "별이"예요.']
    phrase = '같은 행동을 계속 반복하는 장면으로 새로운 정보가 없어요.'
    assert repeated([phrase, phrase])['extra_exact_repeated_texts'] == 1
    assert not repeated(['짧음', '짧음'])['has_exact_repeat']
