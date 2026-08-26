"""Synthetic shortcut benchmark for two different epoche formulations.

The compact two-view regularizer asks whether evidence changes the prediction
relative to a no-evidence baseline.  The explicit four-view loss additionally
identifies a suspected shortcut, trains without it, and tests it in isolation.
They are complementary, not interchangeable.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from philosophia_torch import EpocheLoss, EpocheRegularizer

EVIDENCE_DIM = 6


def make_data(n: int, seed: int, correlation: float = 0.97, flip_shortcut: bool = False):
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randint(0, 2, (n,), generator=generator)
    sign = targets.float().mul(2.0).sub(1.0)
    evidence = 0.55 * sign[:, None] + torch.randn(
        n, EVIDENCE_DIM, generator=generator
    )
    matches = torch.rand(n, generator=generator) < correlation
    shortcut = targets.clone()
    shortcut[~matches] = 1 - shortcut[~matches]
    if flip_shortcut:
        shortcut = 1 - shortcut
    shortcut_value = shortcut.float().mul(2.0).sub(1.0)[:, None]
    shortcut_value = shortcut_value + 0.05 * torch.randn(n, 1, generator=generator)
    return torch.cat([evidence, shortcut_value], dim=1), targets


def bracket_shortcut(inputs: torch.Tensor) -> torch.Tensor:
    result = inputs.clone()
    result[:, -1] = 0.0
    return result


def shortcut_only(inputs: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(inputs)
    result[:, -1] = inputs[:, -1]
    return result


def train_once(seed: int, mode: str, epochs: int = 200) -> dict[str, float]:
    train_x, train_y = make_data(3_000, 1_000 + seed)
    id_x, id_y = make_data(3_000, 2_000 + seed)
    ood_x, ood_y = make_data(3_000, 3_000 + seed, flip_shortcut=True)

    torch.manual_seed(4_000 + seed)
    model = nn.Sequential(
        nn.Linear(EVIDENCE_DIM + 1, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-2)
    compact = EpocheRegularizer(
        model,
        lambda_gain=1.0,
        lambda_prior=0.3,
        min_gain=0.05,
        bracket="zeros",
    )
    explicit = EpocheLoss(
        bracketed_task_weight=1.0,
        consistency_weight=0.5,
        prior_uniformity_weight=1.0,
        evidence_gain_weight=0.0,
    )
    hybrid = EpocheLoss(
        bracketed_task_weight=1.0,
        consistency_weight=0.5,
        prior_uniformity_weight=1.0,
        evidence_gain_weight=0.5,
        min_evidence_gain=0.05,
    )

    for _ in range(epochs):
        full_logits = model(train_x)
        if mode == "baseline":
            loss = F.cross_entropy(full_logits, train_y)
        elif mode == "compact":
            loss = F.cross_entropy(full_logits, train_y) + compact(
                train_x, logits_full=full_logits
            )
        else:
            bracketed_logits = model(bracket_shortcut(train_x))
            prior_only_logits = model(shortcut_only(train_x))
            objective = explicit if mode == "explicit" else hybrid
            no_evidence_logits = (
                model(torch.zeros_like(train_x)) if mode == "hybrid" else None
            )
            loss = objective(
                full_logits,
                train_y,
                bracketed_logits,
                prior_only_logits=prior_only_logits,
                no_evidence_logits=no_evidence_logits,
            ).loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    def accuracy(inputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            return float((model(inputs).argmax(1) == targets).float().mean())

    return {
        "id_accuracy": accuracy(id_x, id_y),
        "ood_accuracy": accuracy(ood_x, ood_y),
        "bracketed_accuracy": accuracy(bracket_shortcut(id_x), id_y),
        "shortcut_only_accuracy": accuracy(shortcut_only(id_x), id_y),
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        output[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    modes = ("baseline", "compact", "explicit", "hybrid")
    runs = {mode: [] for mode in modes}
    for seed in range(args.seeds):
        for mode in modes:
            runs[mode].append(train_once(seed, mode, args.epochs))
    summary = {mode: summarise(rows) for mode, rows in runs.items()}

    columns = (
        "id_accuracy",
        "ood_accuracy",
        "bracketed_accuracy",
        "shortcut_only_accuracy",
    )
    print("Shortcut benchmark: which kind of epoche is being optimised?\n")
    print(f"{'method':<12}" + "".join(f"{column:>24}" for column in columns))
    for mode in modes:
        formatted = [
            f"{summary[mode][column]['mean']:.4f}±{summary[mode][column]['std']:.4f}"
            for column in columns
        ]
        print(f"{mode:<12}" + "".join(f"{value:>24}" for value in formatted))

    payload = {
        "configuration": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "train_samples": 3000,
            "test_samples": 3000,
            "train_shortcut_correlation": 0.97,
            "ood_shortcut_flipped": True,
        },
        "summary": summary,
        "runs": runs,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.json}")


if __name__ == "__main__":
    main()
