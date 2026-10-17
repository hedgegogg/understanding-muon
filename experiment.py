"""Small CPU optimizer experiments. Raw measurements stay in .local/."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

LOCAL = Path(__file__).resolve().parent / ".local"
WEEK_1 = {"seed": 11, "epochs": 5, "batch_size": 64,
          "learning_rates": {"adamw": 1e-3, "lion": 3e-4, "muon": 1e-2},
          "auxiliary_lr": 1e-3}


# Adapted from Google Research's Apache-2.0 Lion implementation.
# Copyright 2023 Google Research. All Rights Reserved.
# Source: https://github.com/google/automl/blob/6a54c8741e7c3265d4547c4f35f47a0391122dc5/lion/lion_pytorch.py
# Changes: omit closure and unused options; this week uses zero weight decay.
# Licensed under Apache-2.0. See THIRD_PARTY_LICENSES.txt for terms and notices.
class Lion(torch.optim.Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, {"lr": lr, "betas": (0.9, 0.99)})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                update = beta1 * momentum + (1 - beta1) * parameter.grad
                parameter.add_(update.sign(), alpha=-group["lr"])
                momentum.mul_(beta2).add_(parameter.grad, alpha=1 - beta2)


# CPU float32 port of Keller Jordan's MIT-licensed single-device Muon.
# Copyright (c) 2024 Keller Jordan. See THIRD_PARTY_LICENSES.txt.
# Source: https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py
# Changes: float32 instead of bfloat16, matrices only, no closure or decay options.
def newton_schulz(gradient):
    if gradient.ndim != 2:
        raise ValueError("Muon requires a two-dimensional hidden weight matrix.")
    x = gradient.float()
    if gradient.shape[0] > gradient.shape[1]:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        polynomial = -4.7750 * gram + 2.0315 * gram @ gram
        x = 3.4445 * x + polynomial @ x
    if gradient.shape[0] > gradient.shape[1]:
        x = x.mT
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, {"lr": lr, "momentum": 0.95})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                momentum.lerp_(gradient, 1 - group["momentum"])
                direction = gradient.lerp(momentum, group["momentum"])
                update = newton_schulz(direction)
                update *= max(1, update.shape[0] / update.shape[1]) ** 0.5
                parameter.add_(update, alpha=-group["lr"])


def load_data():
    digits = load_digits()
    indices = np.arange(len(digits.target))
    remaining, test = train_test_split(indices, test_size=497, random_state=2026,
                                       stratify=digits.target)
    train, validation = train_test_split(remaining, test_size=300, random_state=2026,
                                         stratify=digits.target[remaining])
    partitions = [set(partition) for partition in [train, validation, test]]
    assert [len(partition) for partition in partitions] == [1000, 300, 497]
    assert len(set.union(*partitions)) == 1797
    assert all(not partitions[i] & partitions[j] for i in range(3) for j in range(i))
    assert all(set(digits.target[partition]) == set(range(10))
               for partition in [train, validation, test])
    LOCAL.mkdir(exist_ok=True)
    saved = LOCAL / "data.npz"
    if saved.exists():
        with np.load(saved) as previous:
            for name, partition in zip(["train", "validation", "test"], [train, validation, test]):
                assert np.array_equal(previous[name], partition), "Saved split differs."
    else:
        np.savez(saved, train=train, validation=validation, test=test)
    x = torch.tensor(digits.data / 16, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    # Test examples are deliberately excluded from the training/evaluation interface.
    return {"train": (x[train], y[train]), "validation": (x[validation], y[validation])}


def make_model():
    torch.manual_seed(WEEK_1["seed"])
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, 10))


def make_optimizers(model, name):
    lr = WEEK_1["learning_rates"][name]
    adam = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, **adam)]
    if name == "lion":
        return [Lion(model.parameters(), lr=lr)]
    if name != "muon":
        raise ValueError(f"Unknown optimizer: {name}")
    hidden = [model[0].weight, model[2].weight]
    hidden_ids = {id(parameter) for parameter in hidden}
    auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in hidden_ids]
    assigned = [id(parameter) for parameter in hidden + auxiliary]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))
    assert set(assigned) == {id(parameter) for parameter in model.parameters()}
    return [Muon(hidden, lr=lr),
            torch.optim.AdamW(auxiliary, lr=WEEK_1["auxiliary_lr"], **adam)]


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    measured = {}
    for partition, (x, y) in data.items():
        logits = model(x)
        measured[f"{partition}_loss"] = float(F.cross_entropy(logits, y))
        measured[f"{partition}_accuracy"] = float((logits.argmax(1) == y).float().mean())
    if not all(np.isfinite(value) for value in measured.values()):
        raise FloatingPointError("Non-finite evaluation metric.")
    return measured


def append_csv(filename, row):
    path = LOCAL / filename
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with path.open(newline="") as handle:
            if next(csv.reader(handle)) != list(row):
                raise ValueError(f"Existing {filename} has different columns; preserve it before retrying.")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def train(name, data):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    model = make_model()
    optimizers = make_optimizers(model, name)
    loader = DataLoader(TensorDataset(*data["train"]), batch_size=WEEK_1["batch_size"],
                        shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(WEEK_1["seed"]))
    step = examples_seen = 0
    train_seconds = evaluation_seconds = 0.0
    initial = final = {}
    status, reason = "completed", ""

    def measure(epoch):
        nonlocal evaluation_seconds
        before = time.perf_counter()
        measured = evaluate(model, data)
        evaluation_seconds += time.perf_counter() - before
        append_csv("curves.csv", {"started_at_utc": started_at, "week": 1,
                                 "optimizer": name, "seed": WEEK_1["seed"],
                                 "epoch": epoch, "step": step, **measured})
        return measured

    try:
        initial = final = measure(0)
        for epoch in range(1, WEEK_1["epochs"] + 1):
            before = time.perf_counter()
            model.train()
            try:
                for x, y in loader:
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite minibatch loss.")
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("Non-finite gradient.")
                    for optimizer in optimizers:
                        optimizer.step()
                    step += 1
                    examples_seen += len(y)
                    if not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise FloatingPointError("Non-finite parameter after update.")
            finally:
                train_seconds += time.perf_counter() - before
            final = measure(epoch)
    except Exception as error:
        status = "diverged" if isinstance(error, FloatingPointError) else "failed"
        reason = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt"
    runtime = time.perf_counter() - started
    row = {"started_at_utc": started_at, "week": 1, "condition": "initial_runs",
           "optimizer": name, "seed": WEEK_1["seed"], "lr": WEEK_1["learning_rates"][name],
           "lr_aux": WEEK_1["auxiliary_lr"] if name == "muon" else "",
           "epochs_planned": WEEK_1["epochs"], "steps": step, "examples_seen": examples_seen,
           "initial_train_loss": initial.get("train_loss", ""),
           "initial_validation_loss": initial.get("validation_loss", ""),
           "final_train_loss": final.get("train_loss", ""),
           "final_validation_loss": final.get("validation_loss", ""),
           "final_train_accuracy": final.get("train_accuracy", ""),
           "final_validation_accuracy": final.get("validation_accuracy", ""),
           "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds,
           "runtime_seconds": runtime, "status": status, "reason": reason}
    append_csv("results.csv", row)
    print(f"{name}: {status}, {step} steps, validation loss {row['final_validation_loss']}, {runtime:.6f} seconds", flush=True)
    if status != "completed":
        raise RuntimeError(f"Run saved as {status}: {reason}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, choices=[1], required=True)
    parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_data()
    print("Week 1: three five-epoch CPU runs, seed 11, 80 steps each. Test remains unused.", flush=True)
    started = time.perf_counter()
    for index, name in enumerate(WEEK_1["learning_rates"]):
        row = train(name, data)
        if index == 0:
            # Use the first planned run as the measured pilot, without extra training.
            estimate = 2 * 10 * row["runtime_seconds"]
            full_week_estimate = 18 * (15 / 5) * 10 * row["runtime_seconds"]
            print(f"Measured first pilot: {row['runtime_seconds']:.6f} seconds; remaining two runs estimated at {estimate:.3f} seconds with a 10x allowance.", flush=True)
            print(f"Estimated 18-run full week: {full_week_estimate:.3f} seconds. Normal budget: 15 full epochs, 8 tuning epochs, 240 fixed steps.", flush=True)
            if full_week_estimate > 1800:
                raise RuntimeError("The normal budget may exceed 30 minutes. Review global fast mode before continuing.")
        if time.perf_counter() - started >= 1800:
            raise RuntimeError("Compute budget reached. Completed runs are saved; comparison is incomplete.")
    print("Saved three new runs in .local/results.csv and epoch metrics in .local/curves.csv.")


if __name__ == "__main__":
    main()
