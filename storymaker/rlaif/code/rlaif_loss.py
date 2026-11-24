"""Lee et al. A.3/E: sequence REINFORCE with a learned baseline and KL control."""
import torch


def regularized_return(reward, policy_logps, reference_logps, beta):
    assert 0 <= beta <= 1
    assert policy_logps.shape == reference_logps.shape and policy_logps.ndim == 1
    # E_pi[sum log(pi/ref)] is the trajectory KL; do not add a second KL loss.
    return (1 - beta) * reward - beta * (policy_logps - reference_logps).sum()


def policy_loss(chosen_logps, terminal_return, old_values, batch_size):
    assert chosen_logps.shape == old_values.shape and batch_size > 0
    advantage = (terminal_return - old_values).detach()
    return -(chosen_logps * advantage).sum() / batch_size


def reward_baseline(values, beta, bounds):
    assert list(bounds) == [-1, 1] and 0 <= beta < 1 and torch.isfinite(values).all()
    # Projection onto the known raw-reward range is state-only and cannot increase squared prediction error.
    return (1-beta)*values.detach().clamp(*bounds)


def value_loss(values, terminal_return, batch_size):
    assert values.ndim == 1 and batch_size > 0
    return (values - torch.as_tensor(terminal_return).detach().to(values)).square().sum() / batch_size
