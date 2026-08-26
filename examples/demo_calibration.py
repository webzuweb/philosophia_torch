"""Reproducible calibration demo for ``VirtueRegularizer``.

The original compact article reported one deterministic seed.  This version can
repeat the experiment across several seeds and reports both classic hard-binned
ECE and the differentiable soft-ECE surrogate used during training.

Examples:
    PYTHONPATH=. python examples/demo_calibration.py
    PYTHONPATH=. python examples/demo_calibration.py --seeds 10 --json benchmarks/calibration.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from philosophia_torch import (
    VirtueRegularizer,
    expected_calibration_error,
    soft_expected_calibration_error,
)

D, C = 20, 4


def make_data(n: int, true_weights: torch.Tensor, generator: torch.Generator, noise: float = 0.7):
    inputs = torch.randn(n, D, generator=generator)
    clean_logits = inputs @ true_weights
    targets = (clean_logits + noise * torch.randn(n, C, generator=generator)).argmax(1)
    return inputs, targets


def train_once(seed: int, use_virtue: bool, epochs: int = 300) -> dict[str, float]:
    task_generator = torch.Generator().manual_seed(10_000 + seed)
    true_weights = torch.randn(D, C, generator=task_generator)
    data_generator = torch.Generator().manual_seed(20_000 + seed)
    train_x, train_y = make_data(2_000, true_weights, data_generator)
    test_x, test_y = make_data(2_000, true_weights, data_generator)

    torch.manual_seed(30_000 + seed)
    model = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, C))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    virtue = VirtueRegularizer(
        target_humility=0.99,  # compatibility spelling: target soft-ECE = 0.01
        beta_humility=8.0,
        target_open=None,
    )

    for _ in range(epochs):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        if use_virtue:
            loss = loss + virtue(logits.softmax(dim=1), train_y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        probabilities = model(test_x).softmax(dim=1)
        one_hot = F.one_hot(test_y, C).to(probabilities.dtype)
        return {
            "accuracy": float((probabilities.argmax(1) == test_y).float().mean()),
            "ece": float(expected_calibration_error(probabilities, test_y)),
            "soft_ece": float(soft_expected_calibration_error(probabilities, test_y)),
            "mean_confidence": float(probabilities.max(1).values.mean()),
            "nll": float(F.nll_loss(probabilities.clamp_min(1e-8).log(), test_y)),
            "brier": float((probabilities - one_hot).pow(2).sum(dim=1).mean()),
        }


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = rows[0].keys()
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [row[key] for row in rows]
        result[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")

    results = {"baseline": [], "virtue": []}
    for seed in range(args.seeds):
        results["baseline"].append(train_once(seed, False, args.epochs))
        results["virtue"].append(train_once(seed, True, args.epochs))

    summaries = {name: summarise(rows) for name, rows in results.items()}
    print("Calibration benchmark: baseline vs targeted epistemic humility\n")
    columns = ["accuracy", "ece", "soft_ece", "mean_confidence", "nll", "brier"]
    print(f"{'method':<12}" + "".join(f"{column:>17}" for column in columns))
    for name in ("baseline", "virtue"):
        values = summaries[name]
        formatted = []
        for column in columns:
            mean = values[column]["mean"]
            std = values[column]["std"]
            formatted.append(f"{mean:.4f}±{std:.4f}" if args.seeds > 1 else f"{mean:.4f}")
        print(f"{name:<12}" + "".join(f"{value:>17}" for value in formatted))

    payload = {
        "configuration": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "train_samples": 2000,
            "test_samples": 2000,
            "target_soft_ece": 0.01,
            "calibration_weight": 8.0,
        },
        "summary": summaries,
        "runs": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.json}")


if __name__ == "__main__":
    main()
