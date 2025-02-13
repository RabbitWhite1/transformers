import os
import tempfile
from functools import partial

import numpy as np

from transformers import (
    TrainerCallback,
    TrainingArguments,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import get_tests_dir

if is_torch_available():
    import torch
    from torch import nn

    from transformers import TraceTrainer

PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


class StoreLossCallback(TrainerCallback):
    """
    Simple callback to store the loss.
    """

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


def ForCausalLMLoss(
    logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    if num_items_in_batch is None or disable_num_items_in_batch:
        loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="mean"
        )
    else:
        loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="sum"
        )
        loss = loss / num_items_in_batch
    return loss


class TrainerIntegrationCommon:

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
        ]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            self.assertEqual(log, log1)


class MyModule(nn.Module):
    def __init__(self, a=2, b=5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a).float())
        self.b = nn.Parameter(torch.tensor(b).float())
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        return y


def my_loss(y, labels, num_items_in_batch, disable_num_items_in_batch=False, name=""):
    if disable_num_items_in_batch:
        loss = ((y - labels) ** 2).mean()
        return loss
    else:
        loss = ((y - labels) ** 2).sum()
        loss = loss / num_items_in_batch
        return loss


def main():
    import datasets

    np.random.seed(42)
    num_data = 8
    x = np.random.normal(size=(num_data,)).astype(np.float32)
    y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(num_data,)).astype(np.float32)
    train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

    args_kwargs = {
        "report_to": "none",
        "logging_steps": 1,
        "num_train_epochs": 1,
        "max_steps": 1,
        "learning_rate": 0.001,
        "disable_tqdm": True,
        "max_grad_norm": None,
    }

    EXAMPLE_TYPE = os.environ.get("EXAMPLE_TYPE")

    if EXAMPLE_TYPE == "base":
        base_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
            )
            set_seed(42)
            model = MyModule()
            trainer = TraceTrainer(
                model,
                args,
                train_dataset=train_dataset,
                callbacks=[base_loss_callback],
                compute_loss_func=partial(my_loss, name="base"),
                do_backward=False,
                do_optimizer=False,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()
    elif EXAMPLE_TYPE == "accum":
        grad_accum_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )
            set_seed(42)
            model = MyModule()
            trainer = TraceTrainer(
                model,
                args,
                train_dataset=train_dataset,
                callbacks=[grad_accum_loss_callback],
                compute_loss_func=partial(my_loss, name="fixed"),
                do_backward=False,
                do_optimizer=False,
            )
            trainer.train()
    elif EXAMPLE_TYPE == "broken":
        broken_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )
            set_seed(42)
            model = MyModule()
            trainer = TraceTrainer(
                model,
                args,
                train_dataset=train_dataset,
                callbacks=[broken_loss_callback],
                compute_loss_func=partial(
                    my_loss, disable_num_items_in_batch=True, name="broken"
                ),
                do_backward=False,
                do_optimizer=False,
            )
            trainer.train()
    else:
        raise ValueError(f"Unknown EXAMPLE_TYPE: {EXAMPLE_TYPE}, should be 'base', 'accum' or 'broken'.")


if __name__ == "__main__":
    main()
