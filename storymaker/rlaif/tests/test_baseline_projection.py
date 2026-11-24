"""Known reward-support projection and unbiased state-baseline gradients."""
import unittest
import torch
from storymaker.rlaif.code.rlaif_loss import regularized_return,policy_loss,reward_baseline
from storymaker.rlaif.code.rlaif_environment import validate_settings


class ProjectionChecks(unittest.TestCase):
    def test_projection_improves_bounded_target_MSE_and_penalty_signal(self):
        values=torch.tensor([-3.,-1.2,.4,2.],requires_grad=True)
        bounded=reward_baseline(values,0.,[-1,1])
        self.assertFalse(bounded.requires_grad)
        for reward in (-1.,-.4,.6,1.):
            self.assertTrue(torch.all((bounded-reward).square()<=(values-reward).square()))
        self.assertTrue(torch.all(-1-bounded<=0))

    def test_state_only_projection_preserves_the_exact_REINFORCE_gradient(self):
        logits=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
        lp=logits.log_softmax(0);prob=lp.exp()
        ref=torch.tensor([.4,.6],dtype=torch.float64).log()
        rewards=torch.tensor([.8,-1.],dtype=torch.float64);beta=.05
        exact=-((1-beta)*(prob*rewards).sum()-beta*(prob*(lp-ref)).sum())
        expected=torch.autograd.grad(exact,logits,retain_graph=True)[0]
        for estimate in (-4.,3.):
            baseline=reward_baseline(torch.tensor([estimate],requires_grad=True),beta,[-1,1])
            surrogate=sum(prob[i].detach()*policy_loss(lp[i:i+1],regularized_return(rewards[i],lp[i:i+1],ref[i:i+1],beta),baseline,1) for i in range(2))
            observed=torch.autograd.grad(surrogate,logits,retain_graph=True)[0]
            self.assertTrue(torch.allclose(observed,expected,atol=1e-12))

    def test_unknown_reward_support_is_rejected(self):
        with self.assertRaises(AssertionError):reward_baseline(torch.tensor([0.]),.05,[-2,2])

    def test_configuration_requires_the_explicit_reward_support(self):
        config={'batch_ids':[1],'logical_batch_size':64,'train_ids':[1],'validation_ids':[2],
            'excluded_test_ids':[3],'max_new_tokens':4096,'learning_rate':1e-5,'temperature':.9,
            'kl_beta':.05,'top_p':1,'top_k':0,'rag_required':True,'value_baseline_bounds':[-1,1]}
        self.assertEqual(validate_settings(config),[1])
        with self.assertRaises(AssertionError):validate_settings(dict(config,value_baseline_bounds=[-2,2]))


if __name__=='__main__':unittest.main()
