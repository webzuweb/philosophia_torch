from __future__ import annotations

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from philosophical_learning import (
    AbductiveHypothesisLoss,
    DialecticalLoss,
    DialecticalSynthesis,
    EpistemicVirtueLoss,
    EpocheLoss,
    HermeneuticCircleLoss,
    HermeneuticRefiner,
    SemiosisLoss,
    TriadicSemiosis,
    js_divergence_from_logits,
)


class PhilosophicalLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def assert_finite_scalar(self, tensor: torch.Tensor) -> None:
        self.assertEqual(tensor.ndim, 0)
        self.assertTrue(bool(torch.isfinite(tensor)))

    def test_js_identity_is_zero(self) -> None:
        logits = torch.randn(7, 4)
        value = js_divergence_from_logits(logits, logits)
        self.assertLess(float(value), 1e-7)

    def test_abduction_backward(self) -> None:
        logits = torch.randn(6, 3, 4, requires_grad=True)
        embeddings = torch.randn(6, 3, 5, requires_grad=True)
        target = torch.randint(0, 4, (6,))
        output = AbductiveHypothesisLoss()(
            logits,
            target,
            hypothesis_embeddings=embeddings,
            knowledge_penalty=torch.rand(6, 3),
        )
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(embeddings.grad)

    def test_epoche_backward(self) -> None:
        full = torch.randn(8, 3, requires_grad=True)
        bracketed = torch.randn(8, 3, requires_grad=True)
        prior = torch.randn(8, 3, requires_grad=True)
        target = torch.randint(0, 3, (8,))
        output = EpocheLoss()(full, target, bracketed, prior_only_logits=prior)
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(full.grad)
        self.assertIsNotNone(bracketed.grad)
        self.assertIsNotNone(prior.grad)

    def test_hermeneutic_backward_and_mask(self) -> None:
        refiner = HermeneuticRefiner(12, iterations=2)
        parts = torch.randn(5, 4, 12, requires_grad=True)
        mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
            dtype=torch.float32,
        )
        state = refiner(parts, part_mask=mask)
        head = nn.Linear(12, 3)
        target = torch.randint(0, 3, (5,))
        task = F.cross_entropy(head(state.whole), target)
        output = HermeneuticCircleLoss()(state, task_loss=task, part_mask=mask)
        self.assertEqual(state.parts.shape, parts.shape)
        self.assertEqual(state.whole.shape, (5, 12))
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(parts.grad)

    def test_hermeneutic_fixed_point_matches_article_formula(self) -> None:
        refiner = HermeneuticRefiner(6, iterations=2)
        parts = torch.randn(3, 4, 6)
        state = refiner(parts)
        output = HermeneuticCircleLoss()(state)
        expected = 0.5 * (
            F.mse_loss(state.whole_history[-1], state.whole_history[-2])
            + F.mse_loss(state.part_history[-1], state.part_history[-2])
        )
        self.assertTrue(
            torch.allclose(output.terms["hermeneutic/fixed_point"], expected)
        )

    def test_dialectical_backward(self) -> None:
        layer = DialecticalSynthesis(10)
        thesis = torch.randn(7, 10, requires_grad=True)
        antithesis = torch.randn(7, 10, requires_grad=True)
        state = layer(thesis, antithesis)
        head = nn.Linear(10, 4)
        target = torch.randint(0, 4, (7,))
        task_s = F.cross_entropy(head(state.synthesis), target)
        task_t = F.cross_entropy(head(thesis), target)
        task_a = F.cross_entropy(head(antithesis), target)
        output = DialecticalLoss()(
            thesis,
            antithesis,
            state,
            synthesis_task_loss=task_s,
            thesis_task_loss=task_t,
            antithesis_task_loss=task_a,
        )
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(thesis.grad)
        self.assertIsNotNone(antithesis.grad)

    def test_dialectical_gate_matches_article_formula(self) -> None:
        layer = DialecticalSynthesis(8)
        thesis = torch.randn(5, 8)
        antithesis = torch.randn(5, 8)
        state = layer(thesis, antithesis)
        midpoint = 0.5 * (thesis + antithesis)
        expected_conflict_candidate = (
            state.learned_gate * state.candidate
            + (1.0 - state.learned_gate) * midpoint
        )
        expected_synthesis = (
            state.agreement * midpoint
            + state.conflict * expected_conflict_candidate
        )
        self.assertTrue(bool(torch.all(state.learned_gate >= 0.0)))
        self.assertTrue(bool(torch.all(state.learned_gate <= 1.0)))
        self.assertTrue(
            torch.allclose(state.conflict_candidate, expected_conflict_candidate)
        )
        self.assertTrue(torch.allclose(state.synthesis, expected_synthesis))

    def test_virtue_backward(self) -> None:
        logits = torch.randn(9, 5, requires_grad=True)
        counter = torch.randn(9, 5, requires_grad=True)
        selection = torch.randn(9, requires_grad=True)
        target = torch.randint(0, 5, (9,))
        output = EpistemicVirtueLoss()(
            logits,
            target,
            support=torch.rand(9),
            counter_logits=counter,
            counter_is_relevant=torch.randint(0, 2, (9,)).float(),
            selection_logits=selection,
        )
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(counter.grad)
        self.assertIsNotNone(selection.grad)

    def test_semiosis_backward(self) -> None:
        module = TriadicSemiosis(7, 11, 13)
        sign = torch.randn(10, 7, requires_grad=True)
        obj = torch.randn(10, 11, requires_grad=True)
        state = module(sign, obj)
        output = SemiosisLoss()(sign, obj, state)
        self.assert_finite_scalar(output.loss)
        output.loss.backward()
        self.assertIsNotNone(sign.grad)
        self.assertIsNotNone(obj.grad)


if __name__ == "__main__":
    unittest.main()
