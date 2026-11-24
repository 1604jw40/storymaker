"""CPU boundaries for unchanged, length-weighted frozen-state value diagnostics."""
import copy
import json
import msvcrt
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

import torch
from storymaker import project as p

from storymaker.rlaif.code import rlaif_value_calibration as calibration
from storymaker.rlaif.code.rlaif_value_calibration import ready_for_calibration, summarize, trajectory_metrics


def committed_fixture(root,count):
    batch=root/'batch_0000';ids=list(range(count))
    config={'batch_ids':ids,'logical_batch_size':max(64,count),'train_ids':list(range(200)),
        'validation_ids':list(range(200,243)),'excluded_test_ids':list(range(243,304))}
    if count==128:config['initial_batch_ids']=config.pop('batch_ids')
    p.save(root/'config.json',config);p.save(batch/'config.json',config)
    roles={role:root/'checkpoints/update_0002'/role for role in ('policy','value')}
    joint={'policy':str(roles['policy']),'value':str(roles['value']),'committed_policy_updates':1,'committed_value_updates':1}
    p.save(batch/('first_update_verified.json' if count==128 else 'joint_update_verified.json'),joint)
    for role,checkpoint in roles.items():
        p.save(checkpoint/'manifest.json',{'fixture':True})
        p.save(batch/f'update_{role}_verified.json',{'phase':role,'config_sha256':calibration.file_digest(batch/'config.json'),
            'optimizer_steps':1,'trajectories':count,'saved_and_readback_verified':True,'checkpoint':str(checkpoint)})
    return batch,roles['value']


class ValueCalibrationTests(unittest.TestCase):
    def test_actual64_tail8_and_retained128_use_their_own_snapshot_and_count(self):
        for count in (64,8,128):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);batch,checkpoint=committed_fixture(root,count)
                p.save(root/'config.json',{'logical_batch_size':999})
                source,configuration,ids,_=calibration.calibration_source(root,checkpoint)
                self.assertEqual(source,batch);self.assertEqual(configuration,batch/'config.json')
                self.assertEqual(ids,list(range(count)))

    def test_source_refuses_wrong_role_changed_config_or_uncommitted_value(self):
        for problem in ('role','config','receipt'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);batch,checkpoint=committed_fixture(root,8)
                if problem=='role':checkpoint=checkpoint.with_name('policy')
                elif problem=='config':p.save(batch/'config.json',{'batch_ids':[]})
                else:
                    path=batch/'update_value_verified.json';row=p.read(path);row['saved_and_readback_verified']=False;p.save(path,row)
                with self.assertRaises((AssertionError,KeyError)):calibration.calibration_source(root,checkpoint)

    def test_zero_baseline_comparison_does_not_hide_a_still_weak_projected_critic(self):
        row=trajectory_metrics(torch.tensor([-2.,2.]),torch.tensor([-1.,.2]),.4,7,True)
        summary=summarize([row],[7])
        self.assertAlmostEqual(summary['zero_baseline_token_MSE'],.16)
        self.assertAlmostEqual(summary['before']['projected_token_MSE'],1.16)
        self.assertAlmostEqual(summary['after']['projected_token_MSE'],1.)
        self.assertGreater(summary['projected_after_minus_zero_MSE'],0)

    def test_explicit_update_checkpoint_plan_never_loads_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'checkpoints/update_0002/value'
            with patch.object(calibration,'OUT',root),patch('sys.argv',['calibration','--value-checkpoint',str(checkpoint)]), \
                 patch.object(calibration,'load_model') as load,patch('builtins.print') as output:
                calibration.main()
            load.assert_not_called();plan=json.loads(output.call_args.args[0])
            self.assertEqual(plan['output'],str(root/'value_calibration/update_0002'))
            self.assertEqual(plan['value_checkpoint'],str(checkpoint))
            self.assertFalse((root/'value_calibration').exists())
    def test_known_improvement_and_unfinished_output_are_preserved(self):
        row = trajectory_metrics(torch.tensor([-2., -2.]), torch.tensor([-1., -1.]), -1., 7, False)
        result = summarize([row], [7])
        self.assertEqual(result['before']['token_MSE'], 1.)
        self.assertEqual(result['after']['token_MSE'], 0.)
        self.assertEqual(result['MSE_after_minus_before'], -1.)
        self.assertEqual(result['non_EOS_trajectories'], 1)
        self.assertEqual(result['before']['outside_reward_range_fraction'], 1.)

    def test_token_weighting_and_deterioration_are_reported_without_filtering(self):
        rows = [trajectory_metrics(torch.zeros(4), torch.full((4,), 2.), 0., 1, True),
                trajectory_metrics(torch.zeros(1), torch.zeros(1), 0., 2, False)]
        result = summarize(rows, [1, 2])
        self.assertAlmostEqual(result['after']['token_MSE'], 3.2)
        self.assertAlmostEqual(result['MSE_after_minus_before'], 3.2)
        self.assertEqual(result['tokens'], 5)
        self.assertEqual(result['non_EOS_trajectories'], 1)

    def test_nonfinite_shape_reward_and_mixed_sources_are_rejected(self):
        for old, new, reward in [(torch.zeros(2), torch.zeros(3), 0.),
                                 (torch.tensor([float('nan')]), torch.zeros(1), 0.),
                                 (torch.zeros(1), torch.zeros(1), 1.1)]:
            with self.assertRaises(AssertionError):
                trajectory_metrics(old, new, reward, 1, True)
        row = trajectory_metrics(torch.zeros(1), torch.zeros(1), 0., 1, True)
        with self.assertRaises(AssertionError):
            summarize([row], [2])
        with self.assertRaises(AssertionError):
            summarize([row, row], [1, 1])

    def test_gpu_wait_requires_completed_validation_free_resources_and_fresh_monitor(self):
        monitor = {'at_utc': datetime.now(timezone.utc).isoformat(), 'alerts': [],
                   'inspection_error': None, 'ram': {'free_kib': 10 * 1024**2}}
        args = ['committed_update_verified', [], True, True, monitor]
        self.assertTrue(ready_for_calibration(*args))
        for index, value in [(0, 'running_matched_validation'), (1, [{'pid': 1}]), (2, False), (3, False)]:
            blocked = copy.deepcopy(args)
            blocked[index] = value
            self.assertFalse(ready_for_calibration(*blocked))
        for key, value in [('alerts', ['GPU_shared_memory>2GiB_review_only']),
                           ('inspection_error', 'CIM timeout'),
                           ('at_utc', (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()),
                           ('at_utc', (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()),
                           ('ram', {'free_kib': 8 * 1024**2})]:
            blocked = copy.deepcopy(monitor)
            blocked[key] = value
            self.assertFalse(ready_for_calibration(*args[:4], blocked))

    def test_duplicate_waiter_or_completed_run_cannot_overwrite_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'value_calibration'
            folder.mkdir(parents=True)
            state = folder / 'status.json'
            state.write_text('{"stage":"waiting"}', encoding='utf-8')
            with (folder / 'calibration.lock').open('a+b') as lock:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                with patch.object(calibration, 'OUT', root), patch.object(calibration.argparse.ArgumentParser,
                        'parse_args', return_value=Namespace(execute=True,value_checkpoint=None)), self.assertRaises(OSError):
                    calibration.main()
                self.assertEqual(json.loads(state.read_text())['stage'], 'waiting')
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            (folder / 'manifest.json').write_text('{}', encoding='utf-8')
            state.write_text('{"stage":"complete"}', encoding='utf-8')
            with patch.object(calibration, 'OUT', root), patch.object(calibration.argparse.ArgumentParser,
                    'parse_args', return_value=Namespace(execute=True,value_checkpoint=None)), self.assertRaises(AssertionError):
                calibration.main()
            self.assertEqual(json.loads(state.read_text())['stage'], 'complete')


if __name__ == '__main__':
    unittest.main()
