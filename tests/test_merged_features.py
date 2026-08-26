from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from philosophia_torch import (
    DialecticalSynthesis,
    EpistemicVirtueLoss,
    EpocheLoss,
    HermeneuticCircleLoss,
    HermeneuticRefiner,
    PerspectivalEnsemble,
    PhilosophiaWrapper,
    TriadicSemiosis,
    SemiosisLoss,
    VirtueRegularizer,
    expected_calibration_error,
    hutchinson_jacobian_frobenius,
    js_divergence_from_logits,
    soft_expected_calibration_error,
)


def test_hard_and_soft_ece_are_small_for_confident_correct_predictions():
    probabilities = torch.tensor(
        [[0.99, 0.01], [0.02, 0.98], [0.97, 0.03], [0.01, 0.99]]
    )
    targets = torch.tensor([0, 1, 0, 1])
    assert expected_calibration_error(probabilities, targets).item() < 0.04
    assert soft_expected_calibration_error(probabilities, targets).item() < 0.04


def test_soft_ece_backpropagates_to_logits():
    logits = torch.randn(64, 4, requires_grad=True)
    targets = torch.randint(0, 4, (64,))
    loss = soft_expected_calibration_error(logits.softmax(-1), targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hutchinson_smoothness_uses_multiple_output_directions():
    torch.manual_seed(11)
    inputs = torch.randn(16, 5, requires_grad=True)
    linear = nn.Linear(5, 3, bias=False)
    outputs = linear(inputs)
    estimate = hutchinson_jacobian_frobenius(inputs, outputs, samples=32)
    exact = linear.weight.pow(2).sum()
    # Hutchinson is stochastic; 35% tolerance is ample for this tiny test.
    assert torch.isclose(estimate, exact, rtol=0.35, atol=0.35)
    estimate.backward()
    assert linear.weight.grad is not None


def test_epoche_four_view_loss_exposes_all_terms_and_gradients():
    full = torch.randn(12, 3, requires_grad=True)
    bracketed = torch.randn(12, 3, requires_grad=True)
    prior_only = torch.randn(12, 3, requires_grad=True)
    no_evidence = torch.randn(12, 3, requires_grad=True)
    target = torch.randint(0, 3, (12,))
    objective = EpocheLoss(
        consistency_weight=0.4,
        prior_uniformity_weight=0.6,
        evidence_gain_weight=0.8,
        min_evidence_gain=0.2,
    )
    output = objective(
        full,
        target,
        bracketed,
        prior_only_logits=prior_only,
        no_evidence_logits=no_evidence,
    )
    assert set(output.terms) >= {
        "epoche/full_task",
        "epoche/bracketed_task",
        "epoche/js_consistency",
        "epoche/prior_uniformity_kl",
        "epoche/evidence_gain_js",
        "epoche/evidence_gain_hinge",
    }
    output.loss.backward()
    for tensor in (full, bracketed, prior_only, no_evidence):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_epoche_gain_hinge_is_zero_after_margin():
    full = torch.tensor([[12.0, -12.0], [12.0, -12.0]])
    no_evidence = torch.zeros_like(full)
    target = torch.zeros(2, dtype=torch.long)
    output = EpocheLoss(
        bracketed_task_weight=0.0,
        consistency_weight=0.0,
        prior_uniformity_weight=0.0,
        evidence_gain_weight=1.0,
        min_evidence_gain=0.05,
    )(full, target, full, no_evidence_logits=no_evidence)
    assert output.terms["epoche/evidence_gain_js"].item() > 0.05
    assert output.terms["epoche/evidence_gain_hinge"].item() == 0.0


def test_trainable_hermeneutic_refiner_updates_both_parts_and_whole():
    torch.manual_seed(12)
    parts = torch.randn(4, 6, 8, requires_grad=True)
    refiner = HermeneuticRefiner(8, iterations=3)
    state = refiner(parts)
    assert not torch.allclose(state.parts, parts)
    assert len(state.part_history) == 4
    task = state.whole.pow(2).mean()
    output = HermeneuticCircleLoss()(state, task_loss=task)
    output.loss.backward()
    assert parts.grad is not None


def test_symmetric_dialectic_is_invariant_to_swapping_poles():
    torch.manual_seed(13)
    module = DialecticalSynthesis(10, symmetric=True, return_loss=True)
    thesis = torch.randn(7, 10)
    antithesis = torch.randn(7, 10)
    left = module(thesis, antithesis)
    right = module(antithesis, thesis)
    assert torch.allclose(left.synthesis, right.synthesis, atol=1e-6)
    synthesis, compatibility_loss = left
    assert synthesis.shape == thesis.shape
    assert compatibility_loss is not None and compatibility_loss.ndim == 0


def test_virtue_target_humility_compatibility_maps_to_target_ece():
    regularizer = VirtueRegularizer(
        target_humility=0.99,
        beta_humility=8.0,
        target_open=None,
    )
    assert math.isclose(regularizer.target_ece, 0.01, abs_tol=1e-9)
    logits = torch.randn(64, 4, requires_grad=True)
    targets = torch.randint(0, 4, (64,))
    loss = regularizer(logits.softmax(-1), targets)
    loss.backward()
    assert logits.grad is not None


def test_epistemic_virtue_support_counterevidence_and_selection():
    batch, classes = 16, 3
    logits = torch.randn(batch, classes, requires_grad=True)
    counter_logits = torch.randn(batch, classes, requires_grad=True)
    targets = torch.randint(0, classes, (batch,))
    support = torch.rand(batch)
    relevant = torch.randint(0, 2, (batch,)).float()
    selection = torch.randn(batch, requires_grad=True)
    output = EpistemicVirtueLoss()(
        logits,
        targets,
        support=support,
        counter_logits=counter_logits,
        counter_is_relevant=relevant,
        selection_logits=selection,
    )
    output.loss.backward()
    assert 0.0 <= output.terms["virtue/coverage"].item() <= 1.0
    assert logits.grad is not None
    assert counter_logits.grad is not None
    assert selection.grad is not None


def test_perspectival_js_disagreement_is_bounded_and_drives_abstention():
    logits = torch.tensor(
        [
            [[12.0, -12.0]],
            [[-12.0, 12.0]],
            [[12.0, -12.0]],
        ]
    )
    ensemble = PerspectivalEnsemble(
        3,
        disagreement_metric="js",
        abstain_threshold=0.1,
    )
    output = ensemble(logits)
    assert output["disagreement"].item() <= math.log(2.0) + 1e-5
    assert output["abstain"].item() is True


def test_semiosis_full_cycle_and_anti_collapse_are_differentiable():
    torch.manual_seed(14)
    sign = torch.randn(8, 5, requires_grad=True)
    obj = torch.randn(8, 6, requires_grad=True)
    module = TriadicSemiosis(5, 6, 7)
    state = module(sign, obj)
    output = SemiosisLoss()(sign, obj, state)
    output.loss.backward()
    assert sign.grad is not None and obj.grad is not None
    assert torch.isfinite(output.loss)


def test_wrapper_combines_compact_epoche_and_virtue_regularizers():
    torch.manual_seed(15)
    base = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
    wrapper = PhilosophiaWrapper(
        base,
        use_epoche=True,
        use_virtue=True,
        epoche_kwargs={"lambda_gain": 0.2},
        virtue_kwargs={
            "target_humility": 0.99,
            "beta_humility": 1.0,
            "target_open": None,
        },
    )
    inputs = torch.randn(24, 6)
    targets = torch.randint(0, 3, (24,))
    logits = wrapper(inputs)
    loss = F.cross_entropy(logits, targets) + wrapper.aux_loss(
        inputs, logits, targets=targets
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in wrapper.parameters())


def test_js_divergence_remains_bounded_for_distributions():
    logits_a = torch.tensor([[100.0, -100.0]])
    logits_b = torch.tensor([[-100.0, 100.0]])
    value = js_divergence_from_logits(logits_a, logits_b)
    assert 0.0 <= value.item() <= math.log(2.0) + 1e-6
