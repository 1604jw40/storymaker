"""Audit a committed batch without treating trajectory order as optimizer-step progress."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT
from storymaker.rlaif.code.rlaif_sequence import save


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(config, policy, value, stories, rewards, baseline_means=None):
    ids = config.get('batch_ids', config.get('initial_batch_ids'))
    assert type(config['logical_batch_size']) is int and 0 < len(ids) == len(set(ids)) <= config['logical_batch_size']
    assert set(ids) <= set(config['train_ids'])
    assert not set(ids) & (set(config['validation_ids']) | set(config['excluded_test_ids']))
    assert set(stories) == set(rewards) == set(ids)
    for role, receipt in [('policy', policy), ('value', value)]:
        assert receipt['phase'] == role and receipt['trajectories'] == len(ids)
        assert receipt['optimizer_steps'] == 1 and receipt['saved_and_readback_verified']
        assert receipt['changed_tensors'] > 0 and math.isfinite(receipt['gradient_norm']) and receipt['gradient_norm'] > 0
        assert [row['source_id'] for row in receipt['metrics']] == ids
    rows = []
    beta = config['kl_beta']
    assert 0 <= beta <= 1
    if 'value_baseline_bounds' in config:
        assert config['value_baseline_bounds'] == [-1, 1]
        assert baseline_means is not None and set(baseline_means) == set(ids)
        assert all(math.isfinite(v) and -1 <= v <= 1 for v in baseline_means.values())
        assert policy['value_baseline_bounds'] == value['value_baseline_bounds'] == [-1, 1]
    else:
        assert baseline_means is None
    for source_id, one, two in zip(ids, policy['metrics'], value['metrics']):
        story, reward = stories[source_id], rewards[source_id]
        tokens = len(story['generated_token_ids'])
        assert type(story['ended_with_eos']) is bool
        assert story['identity']['source_id'] == source_id
        assert story['identity']['policy_version'] == config.get('policy_version', 'initial_original_0000')
        assert 0 < tokens <= config['max_new_tokens']
        assert reward['certified_decimal_places'] == 6 and -1 <= reward['normalized_reward'] <= 1
        for metric in (one, two):
            assert metric['tokens'] == tokens and metric['raw_reward'] == reward['normalized_reward']
            assert all(math.isfinite(metric[key]) for key in ['raw_reward', 'regularized_return',
                'sample_sequence_logratio', 'mean_old_value', 'loss', 'logprob_or_value_max_difference'])
            assert 0 <= metric['logprob_or_value_max_difference'] <= .03
            expected = (1-beta)*metric['raw_reward']-beta*metric['sample_sequence_logratio']
            assert math.isclose(metric['regularized_return'], expected, rel_tol=1e-5, abs_tol=1e-5)
        assert two['loss'] >= 0
        for key in ['raw_reward', 'regularized_return', 'sample_sequence_logratio', 'mean_old_value']:
            assert math.isclose(one[key], two[key], rel_tol=1e-6, abs_tol=1e-6), 'Policy/value used different frozen targets'
        baseline = (1-beta)*(baseline_means[source_id] if baseline_means is not None else one['mean_old_value'])
        rows.append({'source_id': source_id, 'tokens': tokens, 'ended_with_eos': story['ended_with_eos'],
            'API_reward': one['raw_reward'], 'regularized_return': one['regularized_return'],
            'sample_sequence_logratio': one['sample_sequence_logratio'],
            'mean_old_value': one['mean_old_value'],
            'mean_policy_baseline': baseline,
            'mean_token_advantage': one['regularized_return']-baseline,
            'policy_loss_contribution': one['loss'], 'value_MSE_contribution': two['loss']})
    total_tokens = sum(row['tokens'] for row in rows)
    return {'scope': 'One committed RL batch: all trajectories sampled before one optimizer update per role.',
        'policy_version_before_update': config.get('policy_version', 'initial_original_0000'),
        'value_baseline_bounds': config.get('value_baseline_bounds'),
        'policy_optimizer_steps': 1, 'value_optimizer_steps': 1,
        'trajectories': len(rows), 'response_tokens': total_tokens,
        'EOS_count': sum(row['ended_with_eos'] for row in rows),
        'mean_API_reward': statistics.mean(row['API_reward'] for row in rows),
        'policy_batch_loss': sum(row['policy_loss_contribution'] for row in rows),
        'value_batch_loss': sum(row['value_MSE_contribution'] for row in rows),
        'value_token_MSE_diagnostic': sum(row['value_MSE_contribution'] for row in rows)*len(rows)/total_tokens,
        'mean_sample_sequence_logratio': statistics.mean(row['sample_sequence_logratio'] for row in rows),
        'gradient_norm': {'policy': policy['gradient_norm'], 'value': value['gradient_norm']},
        'rows': rows,
        'limitations': ['Trajectory order is gradient accumulation order, not optimizer-step progress or a learning curve.',
            'Policy REINFORCE loss can be negative; its decrease alone is not better generation or lower CE.',
            'Value loss sums tokens and divides by batch size; token MSE is a reporting diagnostic, not a changed training loss.',
            'The sampled logratio is not an exact KL estimate and can be negative; initial policy/reference should match.',
            'Rewards and EOS here describe pre-update training outputs. Improvement requires matched post-update validation.',
            'Checkpoint readback is a training-stage receipt; this CPU report does not independently reload every weight.']}


def write_report(summary, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    rows = summary['rows']
    save(folder/'summary.json', summary)
    with (folder/'trajectories.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    columns = [('API_reward', '학습 전 출력의 API 보상 / [-1, 1]'),
        ('policy_loss_contribution', '정책 REINFORCE 손실 기여분 / CE 아님'),
        ('value_MSE_contribution', '가치 예측 MSE 기여분 / 토큰합 ÷ 배치크기'),
        ('sample_sequence_logratio', '고정 참조 대비 표본 경로 logratio / 정확한 KL 아님')]
    x = list(range(1, len(rows)+1))
    for ax, (key, title) in zip(axes.flat, columns):
        ax.plot(x, [row[key] for row in rows], linewidth=.8, color='#335767')
        failed = [(i, row[key]) for i, row in zip(x, rows) if not row['ended_with_eos']]
        if failed:
            ax.scatter(*zip(*failed), color='#D94D2B', s=18, label='미종료 출력')
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('업데이트 전 출력의 누적 처리 순서 (optimizer step 아님)', fontsize=9)
        ax.grid(alpha=.15)
    fig.suptitle(f"실제 {len(rows)}개 배치 · 정책 1회 + 가치 1회 / 처리 순서는 학습 곡선이 아님", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(folder/'training_diagnostics.png', dpi=150)
    plt.close(fig)
    lines = ['# 실제 direct-RLAIF 배치 진단', '', summary['scope'], '',
        f"정책 {summary['policy_optimizer_steps']}회 / 가치 {summary['value_optimizer_steps']}회, 학습 전 출력 {summary['trajectories']}개.",
        f"훈련 출력 EOS {summary['EOS_count']}/{summary['trajectories']}, 평균 API 보상 {summary['mean_API_reward']:.6f}.", '',
        '그림의 가로축은 한 번의 업데이트를 위해 모은 출력의 처리 순서다. CE 또는 업데이트별 수렴 곡선으로 읽지 않는다.', '',
        *['- '+line for line in summary['limitations']]]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_lineage(root):
    """Verify config-source links; these may describe data reuse rather than model ancestry."""
    sources=[];seen=set();source=Path(root).resolve();expected=None
    while True:
        assert source not in seen, 'Cycle in the preserved run lineage'
        seen.add(source)
        if expected is not None:
            assert sha(source/'config.json') == expected, 'A preserved predecessor configuration changed'
        sources.append(source)
        config=p.read(source/'config.json')
        if not config.get('source_previous_run'):
            return sources
        expected=config['source_previous_config_sha256']
        source=Path(config['source_previous_run']).resolve()


def committed_batches(root):
    batches={};links=[];sources=run_lineage(root)
    for index,source in enumerate(sources):
        for batch in source.glob('batch_*'):
            path = batch/'joint_update_verified.json'
            if not path.exists():
                path = batch/'first_update_verified.json'
            if path.exists():
                receipt = p.read(path)
                tag=Path(receipt['policy']).parent.name
                assert tag not in batches, 'Duplicate policy tag across preserved runs'
                assert all(type(receipt[k]) is int and receipt[k]==1 for k in ('committed_policy_updates','committed_value_updates'))
                for role in ('policy','value'):
                    checkpoint=Path(receipt[role]).resolve()
                    assert checkpoint.name==role and checkpoint.parent.name==tag
                    assert checkpoint.is_relative_to((source/'checkpoints').resolve())
                batches[tag]=batch
        config=p.read(source/'config.json')
        if not config.get('policy_checkpoint'):
            assert not config.get('value_checkpoint')
            break
        assert index+1<len(sources), 'A numerical predecessor has no preserved source configuration'
        # Source links may include readonly review runs; exact committed pairs are checked below.
        ancestors=[(ancestor/'checkpoints').resolve() for ancestor in sources[index+1:]]
        pair={role:Path(config[role+'_checkpoint']).resolve() for role in ('policy','value')}
        if config.get('value_source_fork'):
            fork=config['value_source_fork'];proof_path=Path(fork['receipt'])
            assert sha(proof_path)==fork['receipt_sha256']
            proof=p.read(proof_path)
            assert proof_path.parent.resolve()==pair['value']
            assert sha(pair['value']/'manifest.json')==proof['manifest_sha256']
            assert proof['unchanged_language_parameters']==426 and proof['replaced_head_parameters']==2
            assert proof['source_RNG_bytes_preserved'] and proof['new_full_value_updates']==0
            pair['value']=Path(proof['source']).resolve()
            assert sha(pair['value']/'manifest.json')==proof['source_manifest_sha256']
        for role,path in pair.items():
            assert path.name==role and any(path.is_relative_to(parent) for parent in ancestors)
        assert pair['policy'].parent.name==pair['value'].parent.name
        links.append(pair)
    for pair in links:
        tag=pair['policy'].parent.name
        assert tag in batches, 'The configured predecessor is not a committed policy/value pair'
        batch=batches[tag];path=batch/'joint_update_verified.json'
        if not path.exists():path=batch/'first_update_verified.json'
        assert all(Path(p.read(path)[role]).resolve()==pair[role] for role in ('policy','value'))
    return batches


def committed_batch(root, tag):
    batches=committed_batches(root)
    assert tag in batches, 'Exactly one committed policy/value batch is required'
    return batches[tag]


def main(batch=None, output=None):
    root = Path(batch).parent if batch is not None else OUT
    config_path = root/'config.json'
    config = p.read(config_path) if config_path.exists() else {}
    batch = Path(batch) if batch is not None else root/f"batch_{config.get('batch_index', 0):04d}"
    first_path = batch/'joint_update_verified.json'
    if not first_path.exists():
        first_path = batch/'first_update_verified.json'
    assert first_path.exists(), 'No committed RL update: do not fabricate a training graph'
    if (batch/'config.json').exists():
        config_path = batch/'config.json'
        config = p.read(config_path)
    first = p.read(first_path)
    assert first['committed_policy_updates'] == first['committed_value_updates'] == 1
    policy_path, value_path = [batch/f'update_{role}_verified.json' for role in ('policy', 'value')]
    policy, value = p.read(policy_path), p.read(value_path)
    sources = [first_path, config_path, policy_path, value_path]
    for role, receipt in [('policy', policy), ('value', value)]:
        assert receipt['config_sha256'] == sha(config_path)
        assert receipt['checkpoint'] == first[role]
        manifest = Path(receipt['checkpoint'])/'manifest.json'
        assert manifest.parent.name == role and manifest.parent.parent.name == Path(first['policy']).parent.name
        assert p.read(manifest)['changed_tensors'] == receipt['changed_tensors']
        sources.append(manifest)
    stories, rewards, baseline_means = {}, {}, {} if 'value_baseline_bounds' in config else None
    if baseline_means is not None:
        from storymaker.rlaif.code.rlaif_train import prepared_tensor
        import torch
        prepared = batch/'prepared/value/verified.json'
        prepared_receipt = p.read(prepared)
        assert prepared_receipt['config_sha256'] == sha(config_path) and prepared_receipt['source_verified']
        assert prepared_receipt['phase'] == 'value' and prepared_receipt['count'] == config['logical_batch_size']
        sources.append(prepared)
    for path in sorted((batch/'rollouts').glob('*.json')):
        story = p.read(path)
        number = story['identity']['source_id']
        assert number not in stories
        assert story['identity']['config_sha256'] == sha(config_path)
        reward_path = batch/'rewards'/path.stem/'reward.json'
        reward = p.read(reward_path)
        assert reward['source_sha256'] == sha(path)
        stories[number], rewards[number] = story, reward
        sources.extend([path, reward_path])
        if baseline_means is not None:
            values = prepared_tensor(batch, 'value', path.stem, sha(path), prepared_receipt['model_state_sha256'])
            assert values.shape == (len(story['generated_token_ids']),) and torch.isfinite(values).all()
            assert math.isclose(float(values.mean()), policy['metrics'][config['batch_ids'].index(number)]['mean_old_value'], abs_tol=1e-6)
            baseline_means[number] = float(values.clamp(*config['value_baseline_bounds']).mean())
            sources.append(batch/'prepared/value'/(path.stem+'.pt'))
    summary = summarize(config, policy, value, stories, rewards, baseline_means)
    summary['source_sha256'] = {str(path): sha(path) for path in sources}
    summary['report_code_sha256'] = sha(Path(__file__))
    destination = Path(output) if output is not None else batch/'training_report'
    write_report(summary, destination)
    print(json.dumps({'report': str(destination), 'GPU_models_loaded': 0, 'API_calls': 0}))
    return destination


if __name__ == '__main__':
    main()
