from storymaker.qwen.code.pre_rl_benchmark import choices_for


def test_official_kobest_prompt_and_choice_order():
    prefix, options = choices_for('copa', {'premise': '비가 왔다.', 'question': '결과',
                         'alternative_1': '땅이 젖었다.', 'alternative_2': '해가 떴다.'})
    assert prefix == '비가 왔다.  그래서'
    assert options == ['땅이 젖었다.', '해가 떴다.']
    prefix, options = choices_for('hellaswag', {'context': '아침이다.',
                         **{f'ending_{i}': str(i) for i in range(1, 5)}})
    assert prefix == '문장: 아침이다.'
    assert options == ['1', '2', '3', '4']
