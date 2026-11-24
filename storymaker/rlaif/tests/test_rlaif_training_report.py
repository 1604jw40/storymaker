"""CPU-only diagnostics: retain failures, reject mixed targets and never invent an update."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from storymaker import project as p
from storymaker.rlaif.code import rlaif_training_report as report
def fixture():
    ids = list(range(128))
    config = {'initial_batch_ids': ids, 'logical_batch_size': 128, 'train_ids': list(range(200)),
        'validation_ids': list(range(200, 243)), 'excluded_test_ids': list(range(243, 304)),
        'kl_beta': .05, 'max_new_tokens': 4096}
    stories = {number: {'identity': {'source_id': number, 'policy_version': 'initial_original_0000'},
        'generated_token_ids': [1, 2], 'ended_with_eos': number != 127} for number in ids}
    rewards = {number: {'certified_decimal_places': 6, 'normalized_reward': -.6} for number in ids}
    metrics = [{'source_id': number, 'tokens': 2, 'raw_reward': -.6, 'regularized_return': -.57,
        'sample_sequence_logratio': 0., 'mean_old_value': .1, 'loss': -.01,
        'logprob_or_value_max_difference': .001} for number in ids]
    policy = {'phase': 'policy', 'trajectories': 128, 'optimizer_steps': 1,
        'saved_and_readback_verified': True, 'changed_tensors': 315, 'gradient_norm': .2, 'metrics': metrics}
    value = copy.deepcopy(policy)
    value['phase'] = 'value'
    for row in value['metrics']:
        row['loss'] = .7**2*2/128
    return config, policy, value, stories, rewards


class TrainingReportChecks(unittest.TestCase):
    def test_actual64_and_final8_have_actual_divisors_and_policy_identity(self):
        for count in (64, 8):
            config,policy,value,stories,rewards=fixture()
            ids=list(range(count));config.update(batch_ids=ids,logical_batch_size=64,policy_version='update_0001')
            config.pop('initial_batch_ids')
            stories={i:stories[i] for i in ids};rewards={i:rewards[i] for i in ids}
            for story in stories.values():story['identity']['policy_version']='update_0001'
            for receipt in (policy,value):
                receipt['metrics']=receipt['metrics'][:count];receipt['trajectories']=count
            for row in value['metrics']:row['loss']=.49*2/count
            summary=report.summarize(config,policy,value,stories,rewards)
            self.assertEqual(summary['trajectories'],count)
            self.assertAlmostEqual(summary['value_token_MSE_diagnostic'],.49)
            self.assertEqual(summary['policy_version_before_update'],'update_0001')
            stories[0]['identity']['policy_version']='initial_original_0000'
            with self.assertRaises(AssertionError):report.summarize(config,policy,value,stories,rewards)

    def test_projected_baseline_is_mean_of_token_projections_not_projection_of_mean(self):
        config,policy,value,stories,rewards=fixture();config['value_baseline_bounds']=[-1,1]
        for receipt in (policy,value):receipt['value_baseline_bounds']=[-1,1]
        with self.assertRaises(AssertionError):report.summarize(config,policy,value,stories,rewards)
        means={number:0. for number in stories}
        result=report.summarize(config,policy,value,stories,rewards,means)
        self.assertAlmostEqual(result['rows'][0]['mean_policy_baseline'],0)
        self.assertAlmostEqual(result['rows'][0]['mean_token_advantage'],-.57)
        with self.assertRaises(AssertionError):report.summarize(config,policy,value,stories,rewards,dict(means,extra=0))

    def test_real_cpu_prepared_values_and_snapshot_config_are_used_for8(self):
        config,policy,value,stories,rewards=fixture();ids=list(range(8))
        config.update(batch_ids=ids,logical_batch_size=8,policy_version='update_0001',value_baseline_bounds=[-1,1])
        config.pop('initial_batch_ids')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch=root/'batch_0001'
            p.save(root/'config.json',dict(config,learning_rate=999))
            p.save(batch/'config.json',config);digest=report.sha(batch/'config.json')
            checkpoints={role:root/'checkpoints/update_0002'/role for role in ('policy','value')}
            for role,receipt in [('policy',policy),('value',value)]:
                p.save(checkpoints[role]/'manifest.json',{'changed_tensors':315})
                receipt.update(metrics=receipt['metrics'][:8],trajectories=8,config_sha256=digest,
                    checkpoint=str(checkpoints[role]),value_baseline_bounds=[-1,1])
                p.save(batch/f'update_{role}_verified.json',receipt)
            p.save(batch/'joint_update_verified.json',{'policy':str(checkpoints['policy']),'value':str(checkpoints['value']),
                'committed_policy_updates':1,'committed_value_updates':1})
            p.save(batch/'prepared/value/verified.json',{'phase':'value','count':8,'source_verified':True,
                'config_sha256':digest,'model_state_sha256':'frozen-values'})
            for number in ids:
                path=batch/'rollouts'/f'{number:03d}.json';story=stories[number]
                story['identity'].update(policy_version='update_0001',config_sha256=digest)
                p.save(path,story);reward=rewards[number];reward['source_sha256']=report.sha(path)
                p.save(batch/'rewards'/path.stem/'reward.json',reward)
                torch.save({'values':torch.tensor([-3.,3.2]),'source_sha256':report.sha(path),
                    'model_state_sha256':'frozen-values'},batch/'prepared/value'/(path.stem+'.pt'))
            with patch.object(report,'write_report') as write:
                report.main(batch=batch,output=root/'current_report')
            summary=write.call_args.args[0]
            self.assertEqual(summary['trajectories'],8)
            self.assertAlmostEqual(summary['rows'][0]['mean_token_advantage'],-.57)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(p.read(root/'config.json')['learning_rate'],999)

    def test_committed_batch_finds_retained_and_new_round_without_mixing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'current';old=Path(temporary)/'old'
            p.save(old/'config.json',{'historical':True})
            p.save(root/'config.json',{'source_previous_run':str(old),'source_previous_config_sha256':report.sha(old/'config.json'),
                'policy_checkpoint':str(old/'checkpoints/update_0001/policy'),'value_checkpoint':str(old/'checkpoints/update_0001/value')})
            for source,tag in [(old,'update_0001'),(root,'update_0002')]:
                p.save(source/'batch_0000/joint_update_verified.json',{'policy':str(source/'checkpoints'/tag/'policy'),
                    'value':str(source/'checkpoints'/tag/'value'),'committed_policy_updates':1,'committed_value_updates':1})
            self.assertEqual(report.committed_batch(root,'update_0001'),old/'batch_0000')
            self.assertEqual(report.committed_batch(root,'update_0002'),root/'batch_0000')
            p.save(old/'config.json',{'historical':'changed'})
            with self.assertRaises(AssertionError):report.committed_batch(root,'update_0001')

    def test_full_three_run_lineage_is_verified_and_older_models_remain_selectable(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs=[Path(temporary)/str(i) for i in range(3)]
            for i,root in enumerate(runs):
                config={} if not i else {'source_previous_run':str(runs[i-1]),
                    'source_previous_config_sha256':report.sha(runs[i-1]/'config.json'),
                    'policy_checkpoint':str(runs[i-1]/f'checkpoints/update_{i:04d}/policy'),
                    'value_checkpoint':str(runs[i-1]/f'checkpoints/update_{i:04d}/value')}
                p.save(root/'config.json',config)
                tag=f'update_{i+1:04d}'
                p.save(root/'batch_0000/joint_update_verified.json',{'policy':str(root/'checkpoints'/tag/'policy'),
                    'value':str(root/'checkpoints'/tag/'value'),'committed_policy_updates':1,'committed_value_updates':1})
            self.assertEqual(report.run_lineage(runs[2]),list(reversed(runs)))
            self.assertEqual(set(report.committed_batches(runs[2])),{'update_0001','update_0002','update_0003'})
            self.assertEqual(report.committed_batch(runs[2],'update_0001'),runs[0]/'batch_0000')
            p.save(runs[1]/'config.json',{'overwritten':True})
            with self.assertRaises(AssertionError):report.committed_batch(runs[2],'update_0001')

    def test_cycles_duplicate_tags_and_wrong_value_origin_are_rejected(self):
        for problem in ('cycle','duplicate','value_origin','uncommitted_parent'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as temporary:
                old=Path(temporary)/'old';root=Path(temporary)/'current'
                p.save(old/'config.json',{})
                p.save(root/'config.json',{'source_previous_run':str(old),'source_previous_config_sha256':report.sha(old/'config.json'),
                    'policy_checkpoint':str(old/'checkpoints/update_0001/policy'),'value_checkpoint':str(old/'checkpoints/update_0001/value')})
                for folder,tag in [(old,'update_0001'),(root,'update_0002')]:
                    p.save(folder/'batch_0000/joint_update_verified.json',{'policy':str(folder/'checkpoints'/tag/'policy'),
                        'value':str(folder/'checkpoints'/tag/'value'),'committed_policy_updates':1,'committed_value_updates':1})
                if problem=='cycle':
                    p.save(root/'config.json',{'source_previous_run':str(root),'source_previous_config_sha256':'unused'})
                elif problem=='uncommitted_parent':
                    cfg=p.read(root/'config.json');cfg.update(policy_checkpoint=str(old/'checkpoints/update_0999/policy'),
                        value_checkpoint=str(old/'checkpoints/update_0999/value'));p.save(root/'config.json',cfg)
                else:
                    path=root/'batch_0000/joint_update_verified.json';receipt=p.read(path)
                    if problem=='duplicate':
                        receipt.update(policy=str(root/'checkpoints/update_0001/policy'),value=str(root/'checkpoints/update_0001/value'))
                    else:receipt['value']=str(Path(temporary)/'outside/checkpoints/update_0002/value')
                    p.save(path,receipt)
                with self.assertRaises(AssertionError):report.committed_batches(root)

    def test_data_reuse_source_is_not_counted_as_a_numerical_model_predecessor(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary)/'data_source';root=Path(temporary)/'fp32_run'
            p.save(data/'config.json',{})
            p.save(root/'config.json',{'source_previous_run':str(data),'source_previous_config_sha256':report.sha(data/'config.json')})
            for folder in (root,data):
                p.save(folder/'batch_0000/joint_update_verified.json',{'policy':str(folder/'checkpoints/update_0001/policy'),
                    'value':str(folder/'checkpoints/update_0001/value'),'committed_policy_updates':1,'committed_value_updates':1})
            self.assertEqual(report.run_lineage(root),[root,data])
            self.assertEqual(report.committed_batches(root),{'update_0001':root/'batch_0000'})

    def test_one_update_is_not_128_updates_and_failures_stay_in_denominator(self):
        summary = report.summarize(*fixture())
        self.assertEqual(summary['policy_optimizer_steps'], 1)
        self.assertEqual(summary['trajectories'], 128)
        self.assertEqual(summary['EOS_count'], 127)
        self.assertAlmostEqual(summary['policy_batch_loss'], -1.28)
        self.assertAlmostEqual(summary['value_token_MSE_diagnostic'], .49)
        self.assertAlmostEqual(summary['rows'][0]['mean_token_advantage'], -.665)
        self.assertIn('pre-update', summary['limitations'][4])

    def test_mixed_rewards_baselines_returns_and_nonfinite_losses_are_rejected(self):
        for field, bad in [('mean_old_value', .2), ('regularized_return', -.5),
                           ('raw_reward', -.5), ('loss', float('nan'))]:
            with self.subTest(field=field):
                data = fixture()
                data[2]['metrics'][0][field] = bad
                with self.assertRaises(AssertionError):
                    report.summarize(*data)

    def test_wrong_order_split_and_unsupported_update_count_are_rejected(self):
        for problem in ('split', 'order', 'count'):
            with self.subTest(problem=problem):
                data = fixture()
                if problem == 'split':
                    data[0]['validation_ids'].append(0)
                elif problem == 'order':
                    data[1]['metrics'].reverse()
                else:
                    data[1]['optimizer_steps'] = 2
                with self.assertRaises(AssertionError):
                    report.summarize(*data)

    def test_no_receipt_means_no_report_or_fake_curve(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(report, 'OUT', Path(temporary)):
            with self.assertRaisesRegex(AssertionError, 'No committed RL update'):
                report.main()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_plot_and_csv_keep_all_128_trajectories(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report.write_report(report.summarize(*fixture()), folder)
            with Image.open(folder/'training_diagnostics.png') as image:
                self.assertEqual(image.size, (1950, 1200))
            with (folder/'trajectories.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 128)
            self.assertEqual(rows[-1]['ended_with_eos'], 'False')
            self.assertIn('optimizer-step progress', (folder/'summary.md').read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
