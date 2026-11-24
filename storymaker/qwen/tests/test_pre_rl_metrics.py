from storymaker.qwen.code.pre_rl_metrics import words, rouge_l


def test_korean_words_are_not_removed_and_lcs_is_f1():
    assert words('한국어 이야기, 꽃!') == ['한국어', '이야기', '꽃']
    assert rouge_l('한국어 이야기', '한국어 이야기') == 1.
    assert rouge_l('한국어 이야기', '다른 문장') == 0.
    assert rouge_l('가 나', '가 다') == .5
    assert rouge_l('', '가') == 0.
