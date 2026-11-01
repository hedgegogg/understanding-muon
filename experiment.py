"""Small CPU optimizer experiments. Raw measurements stay in .local/."""

import argparse
import csv
from datetime import datetime, timezone
import os
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
WEEK_2 = {"seeds": [11, 22, 33], "epochs": 8,
          "learning_rates": {"adamw": [1e-4, 1e-3, 1e-2],
                             "lion": [3e-5, 3e-4, 3e-3],
                             "muon": [1e-3, 1e-2, 1e-1]}}
WEEK_3 = {"seeds": [11, 22, 33], "epochs": 15}
# Week 2: lowest mean validation loss after eight epochs across seeds 11, 22, 33.
SELECTED_LRS = {"adamw": 1e-2, "lion": 3e-3, "muon": 1e-2, "auxiliary_adamw": 1e-2}


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


def make_model(seed=11):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, 10))


def make_optimizers(model, name, lr, lr_aux=1e-3):
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
            torch.optim.AdamW(auxiliary, lr=lr_aux, **adam)]


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


def train(name, data, week, seed, lr, epochs, lr_aux=1e-3):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    model = make_model(seed)
    optimizers = make_optimizers(model, name, lr, lr_aux)
    loader = DataLoader(TensorDataset(*data["train"]), batch_size=WEEK_1["batch_size"],
                        shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    step = examples_seen = 0
    train_seconds = evaluation_seconds = 0.0
    initial = final = {}
    status, reason = "completed", ""

    def measure(epoch):
        nonlocal evaluation_seconds
        before = time.perf_counter()
        measured = evaluate(model, data)
        evaluation_seconds += time.perf_counter() - before
        append_csv("curves.csv", {"started_at_utc": started_at, "week": week,
                                 "optimizer": name, "seed": seed,
                                 "epoch": epoch, "step": step, **measured})
        return measured

    try:
        initial = final = measure(0)
        for epoch in range(1, epochs + 1):
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
    row = {"started_at_utc": started_at, "week": week,
           "condition": {1: "initial_runs", 2: "lr_sweep", 3: "baseline"}[week],
           "optimizer": name, "seed": seed, "lr": lr,
           "lr_aux": lr_aux if name == "muon" else "",
           "epochs_planned": epochs, "steps": step, "examples_seen": examples_seen,
           "initial_train_loss": initial.get("train_loss", ""),
           "initial_validation_loss": initial.get("validation_loss", ""),
           "final_train_loss": final.get("train_loss", ""),
           "final_validation_loss": final.get("validation_loss", ""),
           "final_train_accuracy": final.get("train_accuracy", ""),
           "final_validation_accuracy": final.get("validation_accuracy", ""),
           "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds,
           "runtime_seconds": runtime, "status": status, "reason": reason}
    append_csv("results.csv", row)
    print(f"{name}, lr={lr:g}, seed={seed}: {status}, {step} steps, validation loss {row['final_validation_loss']}, {runtime:.6f} seconds", flush=True)
    if status not in {"completed", "diverged"}:
        raise RuntimeError(f"Run saved as {status}: {reason}")
    return row


def candidate_summaries(rows, name):
    summaries = []
    for lr in sorted({float(row["lr"]) for row in rows if row["optimizer"] == name}):
        # Repeated commands append new attempts; retain the latest value per seed.
        latest = {int(row["seed"]): row for row in rows
                  if row["optimizer"] == name and float(row["lr"]) == lr}
        runs = [latest.get(seed) for seed in WEEK_2["seeds"]]
        valid = [row is not None and row["status"] == "completed"
                 and int(row["steps"]) == 16 * WEEK_2["epochs"]
                 and np.isfinite(float(row["final_validation_loss"])) for row in runs]
        losses = [float(row["final_validation_loss"]) for row, good in zip(runs, valid) if good]
        eligible = all(valid)
        summaries.append({"lr": lr, "runs": runs, "successful": sum(valid),
                          "mean": float(np.mean(losses)) if eligible else None,
                          "std": float(np.std(losses, ddof=1)) if eligible else None})
    return summaries


def calibrate(data):
    with (LOCAL / "results.csv").open(newline="") as handle:
        pilots = [row for row in csv.DictReader(handle)
                  if int(row["week"]) == 1 and row["status"] == "completed"]
    if not pilots:
        raise RuntimeError("Complete week 1 before estimating the calibration budget.")
    estimate = 27 * WEEK_2["epochs"] * max(float(row["runtime_seconds"]) / int(row["epochs_planned"])
                                         for row in pilots) * 10
    print(f"Week 2: 27 new runs, 3456 updates; pilot-based estimate {estimate:.3f} seconds with a 10x allowance.", flush=True)
    if estimate > 1800:
        raise RuntimeError("Review global fast mode before calibration: estimated budget exceeds 30 minutes.")
    started = time.perf_counter()
    selected = {}
    for name, rates in WEEK_2["learning_rates"].items():
        rows = []
        lr_aux = selected["adamw"] if name == "muon" else 1e-3
        for lr in rates:
            for seed in WEEK_2["seeds"]:
                if time.perf_counter() - started >= 1800:
                    raise RuntimeError("Compute budget reached between runs. Calibration is incomplete.")
                rows.append(train(name, data, 2, seed, lr, WEEK_2["epochs"], lr_aux))
        candidates = [candidate for candidate in candidate_summaries(rows, name)
                      if candidate["mean"] is not None]
        if not candidates:
            raise RuntimeError(f"All {name} candidates are invalid. Inspect the implementation before retrying its whole grid once at one tenth the rates.")
        winner = min(candidates, key=lambda candidate: (candidate["mean"], candidate["lr"]))
        selected[name] = winner["lr"]
        print(f"Selected {name} lr={winner['lr']:g}: mean validation loss {winner['mean']:.6f}, sample SD {winner['std']:.6f}.", flush=True)
    print(f"Calibration completed. Selected rates: {selected}", flush=True)
    return selected


def baseline(data):
    with (LOCAL / "results.csv").open(newline="") as handle:
        pilots = [row for row in csv.DictReader(handle)
                  if int(row["week"]) == 2 and row["status"] == "completed"]
    if not pilots:
        raise RuntimeError("Complete week 2 before running the frozen-rate baseline.")
    for name in WEEK_2["learning_rates"]:
        eligible = [candidate for candidate in candidate_summaries(pilots, name)
                    if candidate["mean"] is not None]
        if not eligible or min(eligible, key=lambda row: (row["mean"], row["lr"]))["lr"] != SELECTED_LRS[name]:
            raise ValueError(f"Frozen {name} rate does not match the completed calibration.")
    if SELECTED_LRS["auxiliary_adamw"] != SELECTED_LRS["adamw"]:
        raise ValueError("Auxiliary AdamW must use the selected ordinary AdamW rate.")
    estimate = 9 * WEEK_3["epochs"] * max(float(row["runtime_seconds"]) / int(row["epochs_planned"])
                                        for row in pilots) * 10
    print(f"Week 3: 9 new runs, 2160 updates; measured-pilot estimate {estimate:.3f} seconds with a 10x allowance.", flush=True)
    if estimate > 1800:
        raise RuntimeError("Estimated baseline exceeds the 30-minute budget. Review before running.")
    started = time.perf_counter()
    rows = []
    for name in WEEK_2["learning_rates"]:
        for seed in WEEK_3["seeds"]:
            if time.perf_counter() - started >= 1800:
                raise RuntimeError("Compute budget reached between runs. Baseline is incomplete.")
            rows.append(train(name, data, 3, seed, SELECTED_LRS[name],
                              WEEK_3["epochs"], SELECTED_LRS["auxiliary_adamw"]))
    print(f"Saved {len(rows)} baseline runs; {sum(row['status'] == 'completed' for row in rows)}/9 completed.", flush=True)
    return rows


def plot_learning_rates():
    os.environ.setdefault("MPLCONFIGDIR", str(LOCAL / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (LOCAL / "results.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if int(row["week"]) == 2 and row["condition"] == "lr_sweep"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, name in zip(axes, WEEK_2["learning_rates"]):
        candidates = candidate_summaries(rows, name)
        if len(candidates) < 3:
            raise ValueError(f"Missing learning-rate results for {name}.")
        for candidate in candidates:
            for index, run in enumerate(candidate["runs"]):
                if run is not None and run["status"] == "completed":
                    axis.scatter(candidate["lr"], float(run["final_validation_loss"]),
                                 marker=["o", "s", "^"][index], color=f"C{index}", alpha=0.75,
                                 label=f"Seed {WEEK_2['seeds'][index]}" if candidate is candidates[0] else None)
            if candidate["mean"] is None:
                axis.text(candidate["lr"], 0.95, "invalid", transform=axis.get_xaxis_transform(),
                          color="red", ha="center")
        eligible = [candidate for candidate in candidates if candidate["mean"] is not None]
        axis.errorbar([candidate["lr"] for candidate in eligible],
                      [candidate["mean"] for candidate in eligible],
                      yerr=[candidate["std"] for candidate in eligible],
                      color="black", marker="_", capsize=4, label="Mean ± sample SD")
        axis.set(xscale="log", xlabel="Learning rate", title={"adamw": "AdamW", "lion": "Lion", "muon": "Muon + AdamW"}[name])
        axis.set_xticks([candidate["lr"] for candidate in candidates],
                        labels=[f"{candidate['lr']:g}" for candidate in candidates])
        axis.minorticks_off()
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation cross-entropy after 8 epochs")
    axes[0].legend(fontsize=8)
    figure.suptitle("Week 2: learning-rate sensitivity (three seeds)")
    figure.tight_layout()
    output = LOCAL.parent / "results/week2/learning_rates.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved learning-rate plot: {output.relative_to(LOCAL.parent)}")


def plot_losses(week):
    os.environ.setdefault("MPLCONFIGDIR", str(LOCAL / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (LOCAL / "results.csv").open(newline="") as handle:
        results = {(row["optimizer"], int(row["seed"])): row for row in csv.DictReader(handle)
                   if int(row["week"]) == week}
    with (LOCAL / "curves.csv").open(newline="") as handle:
        curves = list(csv.DictReader(handle))
    labels = {"adamw": "AdamW", "lion": "Lion", "muon": "Muon + AdamW"}
    seeds = [WEEK_1["seed"]] if week == 1 else WEEK_3["seeds"]
    planned_epochs = WEEK_1["epochs"] if week == 1 else WEEK_3["epochs"]
    epochs = list(range(planned_epochs + 1))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for name, label in labels.items():
        seed_curves = []
        for seed in seeds:
            result = results.get((name, seed))
            if result is None or result["status"] != "completed" or int(result["epochs_planned"]) != planned_epochs:
                raise ValueError(f"A completed {name} run at seed {seed} is required for the loss plot.")
            rows = sorted((row for row in curves if row["started_at_utc"] == result["started_at_utc"]
                           and row["optimizer"] == name and int(row["week"]) == week),
                          key=lambda row: int(row["epoch"]))
            if [int(row["epoch"]) for row in rows] != epochs:
                raise ValueError(f"The {name} seed-{seed} loss curve has missing or duplicate epochs.")
            for partition in ["train", "validation"]:
                losses = [float(row[f"{partition}_loss"]) for row in rows]
                if not np.isfinite(losses).all() or not np.isclose(losses[-1], float(result[f"final_{partition}_loss"])):
                    raise ValueError(f"The {name} {partition} curve does not match the saved result.")
            seed_curves.append(rows)
        for axis, partition in zip(axes, ["train", "validation"]):
            losses = np.array([[float(row[f"{partition}_loss"]) for row in rows] for rows in seed_curves])
            mean = losses.mean(axis=0)
            line, = axis.plot(epochs, mean, marker="o", label=label)
            if len(seeds) > 1:
                sd = losses.std(axis=0, ddof=1)
                axis.fill_between(epochs, mean - sd, mean + sd, color=line.get_color(), alpha=0.18)
    for axis, title in zip(axes, ["Training loss", "Validation loss"]):
        axis.set(title=title, xlabel="Epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Cross-entropy loss")
    title = "seed 11, provisional learning rates" if week == 1 else "frozen rates, mean ± sample SD (3 seeds)"
    figure.suptitle(f"Week {week}: digits, {title}")
    figure.tight_layout()
    output = LOCAL.parent / "results" / f"week{week}" / "losses.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved loss plot: {output.relative_to(LOCAL.parent)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--plot-only", action="store_true", help="Plot saved results without training.")
    args = parser.parse_args()
    if args.plot_only:
        plot_learning_rates() if args.week == 2 else plot_losses(args.week)
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_data()
    if args.week == 2:
        calibrate(data)
        plot_learning_rates()
        return
    if args.week == 3:
        rows = baseline(data)
        if all(row["status"] == "completed" for row in rows):
            plot_losses(3)
        else:
            print("Some baseline runs diverged. Inspect the saved statuses before summarizing.")
        return
    print("Week 1: three five-epoch CPU runs, seed 11, 80 steps each. Test remains unused.", flush=True)
    started = time.perf_counter()
    for index, name in enumerate(WEEK_1["learning_rates"]):
        row = train(name, data, 1, WEEK_1["seed"], WEEK_1["learning_rates"][name],
                    WEEK_1["epochs"], WEEK_1["auxiliary_lr"])
        if row["status"] != "completed":
            raise RuntimeError("Week-1 pilot diverged; inspect its saved measurements.")
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
    plot_losses(args.week)


if __name__ == "__main__":
    main()
