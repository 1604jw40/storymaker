import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from storymaker import project as p
from storymaker.qwen.code import settings
class EpochExtensionTests(unittest.TestCase):
    def test_settings_preserve_existing_trainer_values(self):
        config = p.read(p.RUN / "config.json")
        args = settings.training_arguments(config, "same", 8e-5, 5, -1, True)
        expected = {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 8e-5,
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
            "max_grad_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "bf16": True,
            "seed": 42,
            "data_seed": 42,
            "logging_steps": 5,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
        for key, value in expected.items():
            self.assertEqual(args[key], value, key)
        ten = settings.training_arguments(config, "same", 8e-5, 10, -1, True)
        self.assertEqual(
            {k: v for k, v in args.items() if k != "num_train_epochs"},
            {k: v for k, v in ten.items() if k != "num_train_epochs"},
        )
        self.assertEqual(ten["num_train_epochs"], 10)

    def test_resume_from_five_to_ten_matches_uninterrupted_training(self):
        from transformers import Trainer, TrainingArguments, set_seed

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(0.05)
                self.linear = nn.Linear(4, 2)

            def forward(self, x, labels=None):
                logits = self.linear(self.drop(x))
                return {
                    "loss": nn.functional.cross_entropy(logits, labels),
                    "logits": logits,
                }

        dataset = [
            {"x": torch.tensor([i / 10, 0.2, 0.3, 0.4]), "labels": i % 2}
            for i in range(4)
        ]
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=p.RUN) as directory:

                def fit(name, epochs, resume=None):
                    set_seed(42)
                    model = TinyModel()
                    args = TrainingArguments(
                        output_dir=str(Path(directory) / name),
                        use_cpu=True,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        learning_rate=0.01,
                        lr_scheduler_type="constant",
                        optim="adamw_torch",
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        logging_steps=1,
                        report_to="none",
                        disable_tqdm=True,
                        seed=42,
                        data_seed=42,
                    )
                    trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=dataset,
                        eval_dataset=dataset,
                    )
                    trainer.train(resume_from_checkpoint=resume)
                    return trainer, model

                uninterrupted, reference = fit("full", 10)
                first, _ = fit("split", 5)
                self.assertEqual(first.state.global_step, 10)
                checkpoint = str(Path(directory) / "split/checkpoint-10")
                resumed, actual = fit("split", 10, checkpoint)
                self.assertEqual(resumed.state.global_step, 20)
                evaluations = [r for r in resumed.state.log_history if "eval_loss" in r]
                self.assertEqual([r["epoch"] for r in evaluations], list(range(1, 11)))
                for expected, got in zip(reference.parameters(), actual.parameters()):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                reference_losses = [
                    r["eval_loss"]
                    for r in uninterrupted.state.log_history
                    if "eval_loss" in r
                ]
                self.assertEqual(
                    [r["eval_loss"] for r in evaluations], reference_losses
                )
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
