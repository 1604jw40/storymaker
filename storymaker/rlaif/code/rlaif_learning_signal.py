"""Inspect frozen fresh returns and state predictions before an authorized update."""
import math
from datetime import datetime, timezone

import torch
from storymaker import project as p
from storymaker.rlaif.code.rlaif_train import records_for, prepared_tensor, file_digest
from storymaker.rlaif.code.rlaif_loss import regularized_return, reward_baseline


def compare(predictions, reward):
    assert predictions.ndim == 1 and len(predictions) and torch.isfinite(predictions).all()
    assert math.isfinite(reward) and -1 <= reward <= 1
    return {'raw': float((predictions.double()-reward).square().sum()),
            'projected': float((predictions.clamp(-1, 1).double()-reward).square().sum()),
            'zero': reward**2*len(predictions)}


def inspect(batch, config):
    assert not torch.cuda.is_initialized(), 'CPU inspection must not load another GPU model'
    receipts = {role: p.read(batch/'prepared'/role/'verified.json') for role in ('policy', 'reference', 'value')}
    ids = config['batch_ids']
    config_sha = file_digest(batch.parent/'config.json')
    assert all(r['count'] == len(ids) and r['config_sha256'] == config_sha and r['source_verified'] for r in receipts.values())
    rows = []
    squares = dict(raw=0., projected=0., zero=0.)
    sources = {}
    for key, story, result in records_for(batch, config):
        source = batch/'rollouts'/(key+'.json')
        values = {role: prepared_tensor(batch, role, key, file_digest(source), receipts[role]['model_state_sha256']) for role in receipts}
        assert all(v.shape == (story['generated_tokens'],) and torch.isfinite(v).all() for v in values.values())
        reward = result['normalized_reward']
        terminal = float(regularized_return(reward, values['policy'], values['reference'], config['kl_beta']))
        baseline = reward_baseline(values['value'], config['kl_beta'], config['value_baseline_bounds'])
        assert math.isfinite(terminal)
        rows.append({'source_id': story['identity']['source_id'], 'tokens': story['generated_tokens'],
                     'EOS': story['ended_with_eos'], 'reward': reward, 'regularized_return': terminal,
                     'mean_value': float(values['value'].mean()), 'mean_token_advantage': terminal-float(baseline.mean())})
        for name, value in compare(values['value'], reward).items():
            squares[name] += value
        for path in [source, batch/'rewards'/key/'reward.json', *[batch/'prepared'/role/(key+'.pt') for role in receipts]]:
            sources[str(path)] = file_digest(path)
    count = sum(r['tokens'] for r in rows)
    mse = {k: v/count for k, v in squares.items()}
    report = {'at_utc': datetime.now(timezone.utc).isoformat(), 'passed': mse['projected'] <= mse['zero'],
              'token_MSE': mse, 'trajectories': len(rows), 'tokens': count, 'rows': rows,
              'nonEOS_positive_mean_advantage': sum(not r['EOS'] and r['mean_token_advantage'] > 0 for r in rows),
              'source_sha256': sources, 'config_sha256': config_sha,
              'new_optimizer_updates': 0, 'CUDA_initialized': False,
              'scope': 'Fresh training returns only; do not refit/reselect the baseline on them before their policy gradient. Projected MSE no worse than zero is an engineering guard, not a paper theorem or validation-quality result. Advantage includes sampled KL and can differ from reward sign.'}
    p.save(batch/'learning_signal_before_update.json', report)
    return report
