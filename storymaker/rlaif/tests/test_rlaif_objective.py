"""Paper reward precision, policy/value roles and streamed optimizer invariants."""
import io
import unittest
import tempfile
import subprocess
import sys
from pathlib import Path
import torch
from storymaker.rlaif.code.rlaif_reward_api import expected_reward
from storymaker.rlaif.code.rlaif_loss import regularized_return, policy_loss, value_loss
from storymaker.rlaif.code.rlaif_storage import GradientBuffer, optimizer_step, restore_parameters
from storymaker.rlaif.code.rlaif_sequence import read_shared, validate_receipt, lock_free


class RLAIFChecks(unittest.TestCase):
    def test_global_mutex_uses_byte0_even_when_file_has_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'generation.lock';path.write_bytes(b'0')
            code="import msvcrt,sys; f=open(sys.argv[1],'a+b'); f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1); print('held',flush=True); sys.stdin.readline(); f.close()"
            child=subprocess.Popen([sys.executable,'-c',code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                self.assertEqual(child.stdout.readline().strip(),'held')
                self.assertFalse(lock_free(path))
            finally:
                child.communicate('\n',timeout=10)
            self.assertEqual(child.returncode,0)
            self.assertTrue(lock_free(path))

    def test_status_reader_closes_before_next_file_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'state.json';replacement=root/'state.tmp'
            path.write_text('{"step":1}',encoding='utf-8');replacement.write_text('{"step":2}',encoding='utf-8')
            self.assertEqual(read_shared(path),{'step':1})
            replacement.replace(path)
            self.assertEqual(read_shared(path),{'step':2})

    def test_pipeline_rejects_wrong_configuration_or_role(self):
        valid={'phase':'policy','count':128,'source_verified':True,'config_sha256':'fixed'}
        validate_receipt(valid,'prepare-policy','fixed',128)
        with self.assertRaises(AssertionError):validate_receipt(valid,'prepare-policy','changed',128)
        with self.assertRaises(AssertionError):validate_receipt(valid,'prepare-value','fixed',128)
        with self.assertRaises(AssertionError):validate_receipt(dict(valid,count=127),'prepare-policy','fixed',128)
        for count in (64,8):
            validate_receipt(dict(valid,count=count),'prepare-policy','fixed',count)
            with self.assertRaises(AssertionError):validate_receipt(valid,'prepare-policy','fixed',count)
    def test_omitted_tiny_mass_has_certified_reward_not_assumed_zero(self):
        entries=[{'token':'1','logprob':-.001}]+[{'token':'x'+str(i),'logprob':-60-i} for i in range(19)]
        result=expected_reward(entries)
        self.assertEqual(result['normalized_reward'],-1.)
        self.assertEqual(result['missing_scores'],list(range(2,11)))
        self.assertGreater(float(result['each_missing_unnormalized_weight_upper']),0)

    def test_material_missing_probability_is_rejected(self):
        with self.assertRaises(AssertionError):
            expected_reward([{'token':'1','logprob':-.1},{'token':'x','logprob':-.2}])

    def test_all_ten_uniform_and_constant_logit_shift(self):
        entries=[{'token':str(i),'logprob':-5.} for i in range(1,11)]
        self.assertEqual(expected_reward(entries)['normalized_reward'],0.)
        entries=[dict(x,logprob=x['logprob']-10) for x in entries]
        self.assertEqual(expected_reward(entries)['normalized_reward'],0.)

    def test_streamed_optimizer_matches_joint_and_restores_state(self):
        torch.manual_seed(42)
        initial=[torch.randn(7,9),torch.randn(9)]
        joint=[torch.nn.Parameter(x.clone()) for x in initial]
        streamed=[torch.nn.Parameter(x.clone()) for x in initial]
        one=torch.optim.Adafactor(joint,lr=1e-5,foreach=False)
        many=[torch.optim.Adafactor([x],lr=1e-5,foreach=False) for x in streamed]
        for step in range(2):
            for a,b in zip(joint,streamed):
                grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
            one.step()
            for optimizer in many:optimizer.step()
        for a,b in zip(joint,streamed):self.assertTrue(torch.equal(a,b))
        buffer=io.BytesIO();torch.save(many[0].state_dict(),buffer);buffer.seek(0)
        restored=torch.optim.Adafactor([torch.nn.Parameter(streamed[0].detach().clone())],lr=1e-5,foreach=False)
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertEqual(int(restored.state_dict()['state'][0]['step']),2)

    def test_policy_advantage_does_not_backpropagate_through_value(self):
        logits=torch.tensor([0.,0.],requires_grad=True);value=torch.tensor(.3,requires_grad=True)
        loss=-torch.log_softmax(logits,0)[0]*(torch.tensor(.8)-value).detach()
        loss.backward();self.assertIsNone(value.grad);self.assertLess(float(logits.grad[0]),0)

    def test_paper_reward_coefficient_and_sequence_kl(self):
        lp=torch.tensor([-.3,-.5]);ref=torch.tensor([-.4,-.6])
        result=regularized_return(.8,lp,ref,.05)
        self.assertAlmostEqual(float(result),.95*.8-.05*.2,places=6)
        self.assertAlmostEqual(float(regularized_return(.8,lp,lp,.05)),.76,places=6)

    def test_score_function_kl_gradient_matches_exact_objective(self):
        logits=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
        lp=torch.log_softmax(logits,0);prob=lp.exp();ref=torch.tensor([.4,.6],dtype=torch.float64).log()
        rewards=torch.tensor([.8,-.2],dtype=torch.float64);beta=.05
        objective=-((1-beta)*(prob*rewards).sum()-beta*(prob*(lp-ref)).sum())
        exact=torch.autograd.grad(objective,logits,retain_graph=True)[0]
        surrogate=0
        for i in range(2):
            terminal=regularized_return(rewards[i],lp[i:i+1],ref[i:i+1],beta)
            surrogate=surrogate+prob[i].detach()*policy_loss(lp[i:i+1],terminal,torch.tensor([.95*.3]),1)
        estimated=torch.autograd.grad(surrogate,logits)[0]
        self.assertTrue(torch.allclose(exact,estimated,atol=1e-12))

    def test_loss_batch_average_preserves_token_sum_and_value_detach(self):
        lp=torch.tensor([-.3,-.6],requires_grad=True);old=torch.tensor([.2,.1],requires_grad=True)
        loss=policy_loss(lp,torch.tensor(.8),old,2)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertTrue(torch.allclose(lp.grad,torch.tensor([-.3,-.35])))
        value=torch.tensor([.1,.2],requires_grad=True)
        self.assertAlmostEqual(float(value_loss(value,.5,2).detach()),.125,places=6)

    def test_variable_length_EOS_paths_and_state_baseline_match_exact_gradient(self):
        logits=torch.tensor([[.2,-.3],[.1,.5]],dtype=torch.float64,requires_grad=True)
        values=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
        lp=logits.log_softmax(-1)
        reference=torch.tensor([[.4,.6],[.7,.3]],dtype=torch.float64).log()
        # EOS now, EOS on the second token, or unfinished at the two-token limit.
        paths=[([(0,0)],.1), ([(0,1),(1,0)],.9), ([(0,1),(1,1)],-1.)]
        exact_loss=0;surrogate=0
        for path,reward in paths:
            chosen=torch.stack([lp[state,action] for state,action in path])
            frozen=torch.stack([reference[state,action] for state,action in path])
            probability=chosen.sum().exp()
            utility=regularized_return(reward,chosen,frozen,.05)
            exact_loss=exact_loss-probability*utility
            baseline=.95*values[[state for state,_ in path]]
            surrogate=surrogate+probability.detach()*policy_loss(chosen,utility,baseline,1)
        exact=torch.autograd.grad(exact_loss,logits,retain_graph=True)[0]
        observed,value_grad=torch.autograd.grad(surrogate,(logits,values),allow_unused=True)
        self.assertTrue(torch.allclose(exact,observed,atol=1e-12))
        self.assertIsNone(value_grad)

    def test_fp32_mmap_accumulates_and_checkpoint_restores_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parameter=torch.nn.Parameter(torch.tensor([[.2,-.7],[.3,.4]]))
            original=parameter.detach().clone()
            gradient=GradientBuffer(root/'working',[('weight',parameter)]);gradient.install()
            (parameter*2).sum().backward();(parameter*3).sum().backward()
            norm=gradient.finish()
            self.assertTrue(torch.equal(gradient.view('weight'),torch.full_like(parameter,5)))
            self.assertAlmostEqual(norm,10.)
            manifest=optimizer_step([('weight',parameter)],gradient,root/'checkpoint',1e-3)
            updated=parameter.detach().clone();self.assertFalse(torch.equal(original,updated))
            with torch.no_grad():parameter.zero_()
            restore_parameters([('weight',parameter)],root/'checkpoint')
            self.assertTrue(torch.equal(parameter,updated))
            second=GradientBuffer(root/'working2',[('weight',parameter)]);second.install()
            parameter.sum().backward();second.finish()
            optimizer_step([('weight',parameter)],second,root/'checkpoint2',1e-3,previous=root/'checkpoint')
            state=torch.load(root/'checkpoint2'/manifest['parameters'][0]['file'],weights_only=True)
            self.assertEqual(int(state['optimizer']['state'][0]['step']),2)
            # Close the test mmap explicitly before Windows removes the temporary fixture.
            gradient.close();second.close()


if __name__=='__main__':unittest.main()
