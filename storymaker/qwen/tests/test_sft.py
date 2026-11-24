import copy
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from storymaker import project as p
from storymaker.qwen.code import sft_v5_train as t
class V5SFTTests(unittest.TestCase):
    def test_worker_fit_preserves_task_contract_and_checks_memory_release(self):
        expected = {"source_ids": [1, 3], "learning_rate": 8e-5, "epochs": 5}
        for free_mib, should_fail in ((15000, False), (10000, True)):
            with tempfile.TemporaryDirectory(dir=p.RUN) as directory:
                folder = Path(directory) / "fold_2"

                def child(command, check):
                    self.assertEqual(command[-2], "--fit-task")
                    self.assertTrue(check)
                    task = p.read(Path(command[-1]))
                    self.assertEqual(task["train_ids"], [1, 3])
                    self.assertEqual(task["val_ids"], [4])
                    p.save(folder / "result.json", expected)

                with patch.object(
                    t.subprocess, "run", side_effect=child
                ) as launched, patch.object(
                    t,
                    "gpu_ready",
                    return_value={"free_mib": free_mib, "ready": not should_fail},
                ), patch.object(
                    t, "status"
                ), patch.object(
                    t.time, "sleep"
                ):
                    if should_fail:
                        with self.assertRaises(RuntimeError):
                            t.worker_fit(
                                folder,
                                [1, 3],
                                [4],
                                8e-5,
                                5,
                                "fold2",
                                -1,
                                {"python": "same-python"},
                            )
                    else:
                        self.assertEqual(
                            t.worker_fit(
                                folder,
                                [1, 3],
                                [4],
                                8e-5,
                                5,
                                "fold2",
                                -1,
                                {"python": "same-python"},
                            ),
                            expected,
                        )
                    self.assertEqual(launched.call_count, 1)
                    self.assertEqual(
                        p.read(folder / "worker_exit_gpu.json")["free_mib"], free_mib
                    )
                task = p.read(folder / "fit_task.json")
                task["learning_rate"] = 1e-4
                p.save(folder / "fit_task.json", task)
                with patch.object(t.subprocess, "run") as launched:
                    with self.assertRaises(AssertionError):
                        t.worker_fit(
                            folder,
                            [1, 3],
                            [4],
                            8e-5,
                            5,
                            "fold2",
                            -1,
                            {"python": "same-python"},
                        )
                    launched.assert_not_called()

    def test_no_preprocessing_or_gpu_access_before_approved_start(self):
        with patch.object(
            t.p, "read", return_value={"not_before": "2026-11-08T21:00:00+09:00"}
        ), patch.object(t, "datetime") as clock, patch.object(
            t, "inputs"
        ) as data, patch.object(
            t, "gpu_ready"
        ) as gpu, patch.object(
            t, "status"
        ) as status:
            clock.now.return_value = datetime(2026, 11, 8, 11, 59, tzinfo=timezone.utc)
            clock.fromisoformat.side_effect = datetime.fromisoformat
            t.run()
            data.assert_not_called()
            gpu.assert_not_called()
            status.assert_called_once_with(
                "waiting_start_time", not_before="2026-11-08T21:00:00+09:00"
            )

    def test_qwen_backbone_lora_path_matches_full_forward(self):
        import transformers

        if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
            self.skipTest("실제 SFT 환경의 Transformers에서 별도 실행")
        from peft import LoraConfig, get_peft_model

        config = transformers.Qwen3_5Config(
            text_config=dict(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=32,
                layer_types=["full_attention"],
                full_attention_interval=1,
                eos_token_id=2,
                pad_token_id=0,
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=64,
                num_position_embeddings=64,
            ),
        )
        model = get_peft_model(
            transformers.Qwen3_5ForConditionalGeneration(config),
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
            ),
        )
        ids = torch.randint(3, 128, (1, 12))
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[:, :5] = -100
        dense = model(
            input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
        ).loss
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base = model.get_base_model()
        out = base.model(
            input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
        )
        lean = t.chunked_ce(
            out.last_hidden_state[:, :-1], base.lm_head.weight, labels[:, 1:], 3
        )
        torch.testing.assert_close(lean, dense)
        lean.backward()
        self.assertTrue(
            any(
                x.grad is not None and x.grad.abs().sum() > 0
                for n, x in model.named_parameters()
                if "lora_" in n
            )
        )

    def test_chunked_ce_matches_dense_loss_and_gradients_with_masking(self):
        torch.manual_seed(42)
        h = torch.randn(2, 7, 5, requires_grad=True)
        w = torch.randn(13, 5, requires_grad=True)
        labels = torch.randint(13, (2, 7))
        labels[0, :3] = -100
        labels[1, :2] = -100
        dense = F.cross_entropy(
            F.linear(h, w).float().reshape(-1, 13),
            labels.reshape(-1),
            ignore_index=-100,
        )
        dh, dw = torch.autograd.grad(dense, (h, w))
        chunk = t.chunked_ce(h, w, labels, chunk_size=3)
        ch, cw = torch.autograd.grad(chunk, (h, w))
        torch.testing.assert_close(chunk, dense)
        torch.testing.assert_close(ch, dh)
        torch.testing.assert_close(cw, dw)
        self.assertTrue(torch.equal(ch[0, :3], torch.zeros_like(ch[0, :3])))
        with self.assertRaises(ValueError):
            t.chunked_ce(h, w, torch.full_like(labels, -100))

    def test_saved_folds_preserve_holdout_and_detect_leakage(self):
        config, records, folds = t.inputs()
        self.assertEqual(config["epochs_per_fold"], 10)
        self.assertEqual(config["search_parameters"], ["learning_rate"])
        self.assertEqual(len(records), 243)
        self.assertTrue(
            all(sum(x != -100 for x in r["labels"]) > 0 for r in records.values())
        )
        test = {
            r["source_id"]
            for r in p.read(p.PROCESSED / "dataset.json")
            if r["source_split"] == "test"
        }
        corrupt = copy.deepcopy(folds)
        corrupt[0]["train_ids"][0] = next(iter(test))
        with self.assertRaises(AssertionError):
            p.check_folds(corrupt, set(records), test)

    def test_gpu_gate_blocks_benchmark_even_when_utilization_is_low(self):
        process = type(
            "Result",
            (),
            {"stdout": '{"ProcessId":123,"CommandLine":"run_benchmarks.py"}'},
        )()
        with patch.object(
            t.subprocess, "check_output", return_value="15000, 0\n"
        ), patch.object(t.subprocess, "run", return_value=process):
            self.assertFalse(t.gpu_ready()["ready"])
        process.stdout = "[]"
        with patch.object(
            t.subprocess, "check_output", return_value="15000, 0\n"
        ), patch.object(t.subprocess, "run", return_value=process):
            self.assertTrue(t.gpu_ready()["ready"])

    def test_gpu_gate_handles_venv_parent_and_rejects_missing_host_output(self):
        process = type(
            "Result",
            (),
            {"stdout": '[{"ProcessId":456,"CommandLine":"storymaker.qwen.code.sft_v5_train.py"}]'},
        )()
        with patch.object(
            t.subprocess, "check_output", return_value="15000, 0\n"
        ), patch.object(t.subprocess, "run", return_value=process), patch.object(
            t.os, "getppid", return_value=456
        ):
            self.assertTrue(t.gpu_ready()["ready"])
        process.stdout = None
        with patch.object(
            t.subprocess, "check_output", return_value="15000, 0\n"
        ), patch.object(t.subprocess, "run", return_value=process):
            with self.assertRaises(RuntimeError):
                t.gpu_ready()


if __name__ == "__main__":
    unittest.main()
