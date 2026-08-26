"""philosophia-torch: testable philosophy-inspired objectives for PyTorch.

This library treats philosophy as a source of *problem formulations*, not as a
claim that tensors literally understand Husserl, Gadamer, Hegel, Aristotle,
Nietzsche, or Peirce.  Every component is an engineering operationalisation
with an explicit state, update rule, objective, and failure mode.

The merged v1.0 API combines two earlier prototypes:

* detailed multi-view losses for epoche, hermeneutics, dialectics, epistemic
  virtues, and Peircean semiosis;
* practical wrappers, a lightweight hermeneutic loop, targeted virtue
  regularisation, perspectival abstention, and reproducible demos.

Abduction is included as a control: it is already an established research field,
but the generic scorers and hypothesis-bank loss are useful in ordinary models.
The only runtime dependency is PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = [
    "ObjectiveOutput",
    "js_divergence_from_logits",
    "expected_calibration_error",
    "soft_expected_calibration_error",
    "ensemble_mutual_information",
    "hutchinson_jacobian_frobenius",
    "input_smoothness",
    "variance_covariance_regularizer",
    "abduction_scores",
    "AbductiveScorer",
    "abductive_selection_loss",
    "AbductiveHypothesisLoss",
    "EpocheLoss",
    "epoche_penalty",
    "EpocheRegularizer",
    "hermeneutic_loss",
    "HermeneuticConsistency",
    "HermeneuticState",
    "HermeneuticRefiner",
    "HermeneuticCircleLoss",
    "aufhebung_step",
    "DialecticalState",
    "DialecticalSynthesis",
    "DialecticalLoss",
    "EpistemicVirtueLoss",
    "VirtueRegularizer",
    "perspectival_disagreement",
    "PerspectivalEnsemble",
    "SemiosisState",
    "TriadicSemiosis",
    "SemiosisLoss",
    "PhilosophiaWrapper",
]
__version__ = "1.0.0"


@dataclass
class ObjectiveOutput:
    """A scalar loss plus named component values for inspection and logging."""

    loss: Tensor
    terms: Dict[str, Tensor]

    def detached_terms(self) -> Dict[str, float]:
        """Convert scalar terms to ordinary Python floats for logging."""
        result: Dict[str, float] = {}
        for name, value in self.terms.items():
            if value.numel() != 1:
                raise ValueError(f"Term {name!r} is not scalar: {tuple(value.shape)}")
            result[name] = float(value.detach().cpu())
        return result


def _zero_like(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _require_rank(tensor: Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _require_same_batch(*named_tensors: Tuple[str, Tensor]) -> None:
    sizes = {name: int(tensor.shape[0]) for name, tensor in named_tensors}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Batch sizes do not match: {sizes}")


def js_divergence_from_logits(
    logits_p: Tensor,
    logits_q: Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Jensen-Shannon divergence between two categorical logit tensors.

    Args:
        logits_p: ``[..., C]`` logits.
        logits_q: Tensor with the same shape.
        reduction: ``"none"``, ``"mean"`` or ``"sum"``.  With ``"none"`` the
            class dimension is reduced and all leading dimensions are kept.
        eps: Numerical floor used when taking the logarithm of the mixture.
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            "logits_p and logits_q must have equal shapes, "
            f"got {tuple(logits_p.shape)} and {tuple(logits_q.shape)}"
        )
    if logits_p.ndim < 1:
        raise ValueError("Logits must have at least one dimension")

    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(eps))
    divergence = 0.5 * (
        torch.sum(p * (log_p - log_m), dim=-1)
        + torch.sum(q * (log_q - log_m), dim=-1)
    )

    if reduction == "none":
        return divergence
    if reduction == "mean":
        return divergence.mean()
    if reduction == "sum":
        return divergence.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")



def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    n_bins: int = 15,
) -> Tensor:
    """Classic hard-binned expected calibration error for reporting.

    This metric is intentionally not advertised as a training loss: hard bin
    assignment and ``argmax`` make it non-smooth.  Use
    :func:`soft_expected_calibration_error` inside an objective.
    """
    _require_rank(probabilities, 2, "probabilities")
    _require_rank(targets, 1, "targets")
    _require_same_batch(("probabilities", probabilities), ("targets", targets))
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets).to(probabilities.dtype)
    boundaries = torch.linspace(
        0.0, 1.0, n_bins + 1, device=probabilities.device, dtype=probabilities.dtype
    )
    ece = probabilities.new_zeros(())
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if in_bin.any():
            weight = in_bin.to(probabilities.dtype).mean()
            accuracy = correct[in_bin].mean()
            mean_confidence = confidence[in_bin].mean()
            ece = ece + weight * (accuracy - mean_confidence).abs()
    return ece


def soft_expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    n_bins: int = 15,
    eps: float = 1e-8,
) -> Tensor:
    """A differentiable soft-binning proxy for ECE.

    Confidence is assigned to neighbouring bins with a triangular kernel.
    Correctness still uses a detached current ``argmax`` selector, so gradients
    describe how confidence should move for the current decisions rather than
    differentiating through the discrete class choice.
    """
    _require_rank(probabilities, 2, "probabilities")
    _require_rank(targets, 1, "targets")
    _require_same_batch(("probabilities", probabilities), ("targets", targets))
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets).to(probabilities.dtype).detach()
    centers = torch.linspace(
        1.0 / (2.0 * n_bins),
        1.0 - 1.0 / (2.0 * n_bins),
        n_bins,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    width = 1.0 / float(n_bins)
    distance = (confidence[:, None] - centers[None, :]).abs()
    weights = (1.0 - distance / width).clamp_min(0.0)
    bin_mass = weights.sum(dim=0)
    bin_accuracy = (weights * correct[:, None]).sum(dim=0) / bin_mass.clamp_min(eps)
    bin_confidence = (weights * confidence[:, None]).sum(dim=0) / bin_mass.clamp_min(eps)
    normalised_mass = bin_mass / bin_mass.sum().clamp_min(eps)
    return (normalised_mass * (bin_accuracy - bin_confidence).abs()).sum()


def ensemble_mutual_information(
    member_probabilities: Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Epistemic disagreement ``H(mean p) - mean H(p)`` for an ensemble.

    Args:
        member_probabilities: ``[K, B, C]`` probabilities.
        reduction: ``"none"``, ``"mean"`` or ``"sum"`` over the batch.
    """
    _require_rank(member_probabilities, 3, "member_probabilities")
    mean_probability = member_probabilities.mean(dim=0)
    entropy_mean = -(
        mean_probability * mean_probability.clamp_min(eps).log()
    ).sum(dim=-1)
    member_entropy = -(
        member_probabilities * member_probabilities.clamp_min(eps).log()
    ).sum(dim=-1).mean(dim=0)
    mutual_information = (entropy_mean - member_entropy).clamp_min(0.0)
    if reduction == "none":
        return mutual_information
    if reduction == "mean":
        return mutual_information.mean()
    if reduction == "sum":
        return mutual_information.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def hutchinson_jacobian_frobenius(
    inputs: Tensor,
    outputs: Tensor,
    *,
    samples: int = 1,
) -> Tensor:
    """Estimate the mean squared Frobenius norm of ``d outputs / d inputs``.

    Unlike differentiating ``outputs.sum()``, this Hutchinson estimator does
    not lose Jacobian directions through cancellation.  It is intended as an
    optional smoothness/conscientiousness regulariser and requires
    ``inputs.requires_grad=True``.
    """
    if not inputs.requires_grad:
        raise ValueError("inputs.requires_grad must be True")
    if samples < 1:
        raise ValueError("samples must be positive")
    estimates: List[Tensor] = []
    for _ in range(samples):
        random_sign = torch.empty_like(outputs).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        scalar = (outputs * random_sign).sum()
        (gradient,) = torch.autograd.grad(
            scalar, inputs, create_graph=True, retain_graph=True
        )
        estimates.append(gradient.flatten(1).pow(2).sum(dim=1).mean())
    return torch.stack(estimates).mean()


def input_smoothness(inputs: Tensor, outputs: Tensor) -> Tensor:
    """Compatibility alias using the Hutchinson Jacobian-norm estimator."""
    return hutchinson_jacobian_frobenius(inputs, outputs, samples=1)


def variance_covariance_regularizer(
    representations: Tensor,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """VICReg-style anti-collapse regularisation.

    Returns ``(variance_penalty, covariance_penalty)``.  For a batch with a
    single sample, both terms are zero because covariance is not identifiable.
    """
    _require_rank(representations, 2, "representations")
    batch, dim = representations.shape
    if batch < 2 or dim == 0:
        zero = _zero_like(representations)
        return zero, zero

    centered = representations - representations.mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    std = torch.sqrt(variance + eps)
    variance_penalty = F.relu(target_std - std).pow(2).mean()

    covariance = centered.T @ centered / float(batch - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    covariance_penalty = off_diagonal.pow(2).sum() / float(dim)
    return variance_penalty, covariance_penalty



def abduction_scores(
    observation_log_likelihood: Tensor,
    *,
    complexity: Optional[Tensor] = None,
    conflict: Optional[Tensor] = None,
    simplicity_weight: float = 0.1,
    consistency_weight: float = 0.1,
) -> Tensor:
    """Score candidate explanations by likelihood, simplicity, and consistency."""
    score = observation_log_likelihood
    if complexity is not None:
        score = score - float(simplicity_weight) * complexity
    if conflict is not None:
        score = score - float(consistency_weight) * conflict
    return score


class AbductiveScorer(nn.Module):
    """Differentiable best-explanation selection over supplied hypotheses."""

    def __init__(
        self,
        *,
        simplicity_weight: float = 0.1,
        consistency_weight: float = 0.1,
        hard: bool = False,
        temperature: float = 1.0,
        lambda_simplicity: Optional[float] = None,
        lambda_consistency: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> None:
        super().__init__()
        if lambda_simplicity is not None:
            simplicity_weight = float(lambda_simplicity)
        if lambda_consistency is not None:
            consistency_weight = float(lambda_consistency)
        if tau is not None:
            temperature = float(tau)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.simplicity_weight = float(simplicity_weight)
        self.consistency_weight = float(consistency_weight)
        self.hard = bool(hard)
        self.temperature = float(temperature)

    def forward(
        self,
        observation_log_likelihood: Tensor,
        *,
        complexity: Optional[Tensor] = None,
        conflict: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        scores = abduction_scores(
            observation_log_likelihood,
            complexity=complexity,
            conflict=conflict,
            simplicity_weight=self.simplicity_weight,
            consistency_weight=self.consistency_weight,
        )
        if self.hard and self.training:
            weights = F.gumbel_softmax(
                scores,
                tau=self.temperature,
                hard=True,
                dim=-1,
            )
        else:
            weights = F.softmax(scores / self.temperature, dim=-1)
        return weights, scores


def abductive_selection_loss(
    observation_log_likelihood: Tensor,
    chosen_index: Tensor,
    *,
    complexity: Optional[Tensor] = None,
    conflict: Optional[Tensor] = None,
    simplicity_weight: float = 0.1,
    consistency_weight: float = 0.1,
) -> Tensor:
    """Supervised/weakly-supervised loss for a known preferred hypothesis."""
    scores = abduction_scores(
        observation_log_likelihood,
        complexity=complexity,
        conflict=conflict,
        simplicity_weight=simplicity_weight,
        consistency_weight=consistency_weight,
    )
    return F.cross_entropy(scores, chosen_index)


class AbductiveHypothesisLoss(nn.Module):
    """Soft best-explanation learning over K competing neural hypotheses.

    This is a generic engineering extension, not a claim that abduction is new
    to machine learning.  The model supplies ``K`` hypotheses for every sample;
    the objective rewards the lowest-energy explanation while optionally
    enforcing symbolic/physical knowledge and diversity.

    Formula (per sample):

        E_k = CE(z_k, y) + lambda_K * Omega_K(h_k)
        L_softmin = -tau * log sum_k exp(-E_k / tau)

    Diversity prevents all hypotheses from collapsing to the same explanation.
    A batch-level balance term discourages permanently unused slots.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.5,
        knowledge_weight: float = 1.0,
        diversity_weight: float = 0.1,
        diversity_margin: float = 1.0,
        balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if diversity_margin < 0:
            raise ValueError("diversity_margin must be non-negative")
        self.temperature = float(temperature)
        self.knowledge_weight = float(knowledge_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_margin = float(diversity_margin)
        self.balance_weight = float(balance_weight)

    def forward(
        self,
        hypothesis_logits: Tensor,
        target: Tensor,
        *,
        hypothesis_embeddings: Optional[Tensor] = None,
        knowledge_penalty: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        _require_rank(hypothesis_logits, 3, "hypothesis_logits")
        _require_rank(target, 1, "target")
        batch, hypotheses, classes = hypothesis_logits.shape
        if hypotheses < 1 or classes < 2:
            raise ValueError("Expected at least one hypothesis and two classes")
        _require_same_batch(("hypothesis_logits", hypothesis_logits), ("target", target))

        expanded_target = target[:, None].expand(batch, hypotheses).reshape(-1)
        per_hypothesis_ce = F.cross_entropy(
            hypothesis_logits.reshape(batch * hypotheses, classes),
            expanded_target,
            reduction="none",
        ).reshape(batch, hypotheses)

        if knowledge_penalty is None:
            knowledge = torch.zeros_like(per_hypothesis_ce)
        else:
            if knowledge_penalty.shape != (batch, hypotheses):
                raise ValueError(
                    "knowledge_penalty must have shape [B, K], got "
                    f"{tuple(knowledge_penalty.shape)}"
                )
            knowledge = knowledge_penalty

        energy = per_hypothesis_ce + self.knowledge_weight * knowledge
        softmin_per_sample = -self.temperature * torch.logsumexp(
            -energy / self.temperature, dim=1
        )
        # Add tau*log(K) so the value is on roughly the same scale as CE and is
        # exactly E for K=1.  This constant does not affect gradients.
        softmin = (
            softmin_per_sample
            + self.temperature * torch.log(
                torch.tensor(float(hypotheses), device=energy.device, dtype=energy.dtype)
            )
        ).mean()

        weights = F.softmax(-energy / self.temperature, dim=1)
        usage = weights.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / float(hypotheses))
        balance = torch.sum(usage * (torch.log(usage.clamp_min(1e-8)) - torch.log(uniform)))

        diversity = _zero_like(hypothesis_logits)
        if hypothesis_embeddings is not None:
            _require_rank(hypothesis_embeddings, 3, "hypothesis_embeddings")
            if hypothesis_embeddings.shape[:2] != (batch, hypotheses):
                raise ValueError(
                    "hypothesis_embeddings must have leading shape [B, K], got "
                    f"{tuple(hypothesis_embeddings.shape)}"
                )
            if hypotheses > 1:
                distances = torch.cdist(hypothesis_embeddings, hypothesis_embeddings, p=2)
                mask = ~torch.eye(
                    hypotheses, device=distances.device, dtype=torch.bool
                ).unsqueeze(0)
                pair_penalties = F.relu(self.diversity_margin - distances).pow(2)
                diversity = pair_penalties.masked_select(mask.expand_as(pair_penalties)).mean()

        loss = (
            softmin
            + self.diversity_weight * diversity
            + self.balance_weight * balance
        )
        terms = {
            "abduction/total": loss,
            "abduction/softmin_energy": softmin,
            "abduction/best_ce": per_hypothesis_ce.min(dim=1).values.mean(),
            "abduction/expected_knowledge": (weights * knowledge).sum(dim=1).mean(),
            "abduction/diversity_penalty": diversity,
            "abduction/slot_balance_kl": balance,
        }
        return ObjectiveOutput(loss=loss, terms=terms)


class EpocheLoss(nn.Module):
    """Four-view operationalisation of phenomenological bracketing.

    The two most common interpretations of "bracketing" solve different
    problems and should not be conflated:

    * nuisance bracketing: remove a suspected shortcut while retaining genuine
      evidence; full and bracketed predictions should remain correct and close;
    * evidence removal: remove the phenomenon itself; the full prediction should
      differ from the no-evidence baseline by at least a bounded margin.

    Optional ``prior_only_logits`` expose the suspected shortcut in isolation.
    """

    def __init__(
        self,
        *,
        bracketed_task_weight: float = 1.0,
        consistency_weight: float = 0.25,
        prior_uniformity_weight: float = 0.25,
        evidence_gain_weight: float = 0.0,
        min_evidence_gain: float = 0.05,
    ) -> None:
        super().__init__()
        if min_evidence_gain < 0:
            raise ValueError("min_evidence_gain must be non-negative")
        self.bracketed_task_weight = float(bracketed_task_weight)
        self.consistency_weight = float(consistency_weight)
        self.prior_uniformity_weight = float(prior_uniformity_weight)
        self.evidence_gain_weight = float(evidence_gain_weight)
        self.min_evidence_gain = float(min_evidence_gain)

    def forward(
        self,
        full_logits: Tensor,
        target: Tensor,
        bracketed_logits: Tensor,
        *,
        prior_only_logits: Optional[Tensor] = None,
        no_evidence_logits: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        _require_rank(full_logits, 2, "full_logits")
        _require_rank(bracketed_logits, 2, "bracketed_logits")
        _require_rank(target, 1, "target")
        if full_logits.shape != bracketed_logits.shape:
            raise ValueError("full_logits and bracketed_logits must have equal shapes")
        _require_same_batch(
            ("full_logits", full_logits),
            ("bracketed_logits", bracketed_logits),
            ("target", target),
        )

        full_task = F.cross_entropy(full_logits, target)
        bracketed_task = F.cross_entropy(bracketed_logits, target)
        consistency = js_divergence_from_logits(full_logits, bracketed_logits)

        prior_uniformity = _zero_like(full_logits)
        if prior_only_logits is not None:
            if prior_only_logits.shape != full_logits.shape:
                raise ValueError("prior_only_logits must match full_logits")
            prior_probability = F.softmax(prior_only_logits, dim=-1)
            classes = prior_probability.shape[-1]
            prior_uniformity = (
                prior_probability
                * (
                    torch.log(prior_probability.clamp_min(1e-8))
                    + math.log(float(classes))
                )
            ).sum(dim=-1).mean()

        evidence_gain = _zero_like(full_logits)
        evidence_gain_penalty = _zero_like(full_logits)
        if no_evidence_logits is not None:
            if no_evidence_logits.shape != full_logits.shape:
                raise ValueError("no_evidence_logits must match full_logits")
            evidence_gain = js_divergence_from_logits(full_logits, no_evidence_logits)
            evidence_gain_penalty = F.relu(
                evidence_gain.new_tensor(self.min_evidence_gain) - evidence_gain
            )

        loss = (
            full_task
            + self.bracketed_task_weight * bracketed_task
            + self.consistency_weight * consistency
            + self.prior_uniformity_weight * prior_uniformity
            + self.evidence_gain_weight * evidence_gain_penalty
        )
        terms = {
            "epoche/total": loss,
            "epoche/full_task": full_task,
            "epoche/bracketed_task": bracketed_task,
            "epoche/js_consistency": consistency,
            "epoche/prior_uniformity_kl": prior_uniformity,
            "epoche/evidence_gain_js": evidence_gain,
            "epoche/evidence_gain_hinge": evidence_gain_penalty,
        }
        return ObjectiveOutput(loss=loss, terms=terms)


def epoche_penalty(
    logits_full: Tensor,
    logits_prior: Tensor,
    *,
    mode: str = "evidence_gain",
) -> Tensor:
    """Compatibility helper for the compact two-view epoche formulation."""
    if mode == "evidence_gain":
        return js_divergence_from_logits(logits_full, logits_prior)
    if mode == "suppress_prior":
        probability = F.softmax(logits_prior, dim=-1)
        classes = probability.shape[-1]
        return (
            probability
            * (probability.clamp_min(1e-8).log() + math.log(float(classes)))
        ).sum(dim=-1).mean()
    raise ValueError(f"Unknown mode: {mode!r}")


class EpocheRegularizer(nn.Module):
    """A model wrapper for the simpler evidence-gain version of epoche.

    This preserves the compact API of the second prototype.  It does *not*
    identify a nuisance prior by itself; use :class:`EpocheLoss` with explicit
    bracketed and prior-only views for shortcut removal.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lambda_gain: float = 1.0,
        lambda_prior: float = 0.0,
        min_gain: float = 0.05,
        bracket: Union[str, Callable[[Tensor], Tensor]] = "zeros",
    ) -> None:
        super().__init__()
        if min_gain < 0:
            raise ValueError("min_gain must be non-negative")
        self.model = model
        self.lambda_gain = float(lambda_gain)
        self.lambda_prior = float(lambda_prior)
        self.min_gain = float(min_gain)
        self.bracket = bracket

    def _bracketed_input(self, inputs: Tensor) -> Tensor:
        if callable(self.bracket):
            return self.bracket(inputs)
        if self.bracket == "zeros":
            return torch.zeros_like(inputs)
        if self.bracket == "mean":
            return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
        raise ValueError(f"Unknown bracket: {self.bracket!r}")

    def forward(
        self,
        inputs: Tensor,
        logits_full: Optional[Tensor] = None,
    ) -> Tensor:
        if logits_full is None:
            logits_full = self.model(inputs)
        logits_without_evidence = self.model(self._bracketed_input(inputs))
        gain = js_divergence_from_logits(logits_full, logits_without_evidence)
        regularizer = logits_full.new_zeros(())
        if self.lambda_gain != 0.0:
            regularizer = regularizer + self.lambda_gain * F.relu(
                gain.new_tensor(self.min_gain) - gain
            )
        if self.lambda_prior != 0.0:
            probability = F.softmax(logits_without_evidence, dim=-1)
            classes = probability.shape[-1]
            distance_from_uniform = (
                probability
                * (probability.clamp_min(1e-8).log() + math.log(float(classes)))
            ).sum(dim=-1).mean()
            regularizer = regularizer + self.lambda_prior * distance_from_uniform
        return regularizer


def _l2_normalise(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(eps)


def hermeneutic_loss(
    part_representations: Tensor,
    whole_representation: Tensor,
    *,
    mask: Optional[Tensor] = None,
    part_in_context: Optional[Tensor] = None,
    whole_weight: float = 1.0,
    context_weight: float = 0.5,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Tensor:
    """Cosine part-whole consistency for a lightweight plug-in loop."""
    if alpha is not None:
        whole_weight = float(alpha)
    if beta is not None:
        context_weight = float(beta)
    _require_rank(part_representations, 3, "part_representations")
    _require_rank(whole_representation, 2, "whole_representation")
    batch, count, _ = part_representations.shape
    if mask is None:
        mask = torch.ones(batch, count, device=part_representations.device)
    if mask.shape != (batch, count):
        raise ValueError("mask must have shape [B, N]")
    expanded_mask = mask.to(part_representations.dtype).unsqueeze(-1)
    denominator = expanded_mask.sum(dim=1).clamp_min(1.0)
    aggregate = (part_representations * expanded_mask).sum(dim=1) / denominator
    total = float(whole_weight) * (
        1.0 - F.cosine_similarity(
            _l2_normalise(aggregate),
            _l2_normalise(whole_representation),
            dim=-1,
        )
    ).mean()
    if part_in_context is not None:
        if part_in_context.shape != part_representations.shape:
            raise ValueError("part_in_context must match part_representations")
        cosine = F.cosine_similarity(
            _l2_normalise(part_representations),
            _l2_normalise(part_in_context),
            dim=-1,
        )
        valid_mean = (
            cosine * mask.to(cosine.dtype)
        ).sum(dim=1) / mask.to(cosine.dtype).sum(dim=1).clamp_min(1.0)
        total = total + float(context_weight) * (1.0 - valid_mean).mean()
    return total


class HermeneuticConsistency(nn.Module):
    """A lightweight attention-based whole↔parts refinement loop.

    It reweights parts relative to the current whole and updates the whole by a
    momentum mixture.  For full two-way reinterpretation, use
    :class:`HermeneuticRefiner`.
    """

    def __init__(
        self,
        *,
        n_turns: int = 3,
        temperature: float = 1.0,
        momentum: float = 0.5,
        tau: Optional[float] = None,
    ) -> None:
        super().__init__()
        if tau is not None:
            temperature = float(tau)
        if n_turns < 1:
            raise ValueError("n_turns must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must lie in [0, 1]")
        self.n_turns = int(n_turns)
        self.temperature = float(temperature)
        self.momentum = float(momentum)

    def forward(
        self,
        part_representations: Tensor,
        whole_representation: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        _require_rank(part_representations, 3, "part_representations")
        _require_rank(whole_representation, 2, "whole_representation")
        batch, count, _ = part_representations.shape
        if mask is None:
            mask = torch.ones(batch, count, device=part_representations.device)
        if mask.shape != (batch, count):
            raise ValueError("mask must have shape [B, N]")
        whole = whole_representation
        negative_infinity = torch.finfo(part_representations.dtype).min
        for _ in range(self.n_turns):
            scores = torch.einsum(
                "bnd,bd->bn",
                _l2_normalise(part_representations),
                _l2_normalise(whole),
            ) / self.temperature
            scores = scores.masked_fill(mask == 0, negative_infinity)
            attention = F.softmax(scores, dim=1).unsqueeze(-1)
            aggregate = (part_representations * attention).sum(dim=1)
            whole = self.momentum * whole + (1.0 - self.momentum) * aggregate
        loss = hermeneutic_loss(
            part_representations,
            whole,
            mask=mask,
            whole_weight=1.0,
            context_weight=0.0,
        )
        return whole, loss


@dataclass
class HermeneuticState:
    """State trajectory produced by :class:`HermeneuticRefiner`."""

    parts: Tensor
    whole: Tensor
    part_history: List[Tensor]
    whole_history: List[Tensor]


class HermeneuticRefiner(nn.Module):
    """Iteratively interpret parts through the whole and the whole through parts.

    All representations use a shared dimensionality ``dim``.  A GRU cell first
    updates the whole from a pooled summary of the current parts.  A second GRU
    cell then reinterprets every part in the context of the updated whole.
    Weights are tied across iterations, making the block usable as a recurrent
    refinement layer on top of CNN, Transformer, graph, or document encoders.
    """

    def __init__(
        self,
        dim: int,
        *,
        iterations: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.dim = int(dim)
        self.iterations = int(iterations)
        self.whole_cell = nn.GRUCell(dim, dim)
        self.part_cell = nn.GRUCell(dim, dim)
        self.part_norm = nn.LayerNorm(dim)
        self.whole_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        parts: Tensor,
        whole: Optional[Tensor] = None,
        *,
        part_mask: Optional[Tensor] = None,
    ) -> HermeneuticState:
        _require_rank(parts, 3, "parts")
        batch, count, dim = parts.shape
        if dim != self.dim:
            raise ValueError(f"Expected parts dimension {self.dim}, got {dim}")
        if count < 1:
            raise ValueError("At least one part is required")

        if part_mask is not None:
            if part_mask.shape != (batch, count):
                raise ValueError("part_mask must have shape [B, N]")
            mask = part_mask.to(dtype=parts.dtype).unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            pooled = (parts * mask).sum(dim=1) / denominator
        else:
            mask = None
            pooled = parts.mean(dim=1)

        current_parts = parts
        current_whole = pooled if whole is None else whole
        _require_rank(current_whole, 2, "whole")
        if current_whole.shape != (batch, dim):
            raise ValueError(f"whole must have shape {(batch, dim)}")

        part_history: List[Tensor] = [current_parts]
        whole_history: List[Tensor] = [current_whole]

        for _ in range(self.iterations):
            if mask is None:
                pooled = current_parts.mean(dim=1)
            else:
                denominator = mask.sum(dim=1).clamp_min(1.0)
                pooled = (current_parts * mask).sum(dim=1) / denominator

            next_whole = self.whole_cell(
                self.dropout(pooled), current_whole
            )
            next_whole = self.whole_norm(next_whole)

            contextual_input = next_whole[:, None, :].expand(batch, count, dim)
            next_parts = self.part_cell(
                self.dropout(contextual_input).reshape(batch * count, dim),
                current_parts.reshape(batch * count, dim),
            ).reshape(batch, count, dim)
            next_parts = self.part_norm(next_parts)
            if mask is not None:
                # Padded parts remain unchanged and do not leak into later pools.
                next_parts = next_parts * mask + current_parts * (1.0 - mask)

            current_whole = next_whole
            current_parts = next_parts
            part_history.append(current_parts)
            whole_history.append(current_whole)

        return HermeneuticState(
            parts=current_parts,
            whole=current_whole,
            part_history=part_history,
            whole_history=whole_history,
        )


class HermeneuticCircleLoss(nn.Module):
    """Part-whole consistency and fixed-point regularisation for a refinement loop."""

    def __init__(
        self,
        *,
        consistency_weight: float = 0.25,
        fixed_point_weight: float = 0.1,
        trajectory_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.consistency_weight = float(consistency_weight)
        self.fixed_point_weight = float(fixed_point_weight)
        self.trajectory_weight = float(trajectory_weight)

    def forward(
        self,
        state: HermeneuticState,
        *,
        task_loss: Optional[Tensor] = None,
        part_mask: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        if len(state.part_history) != len(state.whole_history):
            raise ValueError("Part and whole histories must have equal length")
        if len(state.part_history) < 2:
            raise ValueError("Hermeneutic history must contain an update")

        reference = state.whole
        task = _zero_like(reference) if task_loss is None else task_loss
        if task.ndim != 0:
            task = task.mean()

        consistency_terms: List[Tensor] = []
        for parts_t, whole_t in zip(state.part_history[1:], state.whole_history[1:]):
            if part_mask is None:
                pooled = parts_t.mean(dim=1)
            else:
                if part_mask.shape != parts_t.shape[:2]:
                    raise ValueError("part_mask must match [B, N]")
                mask = part_mask.to(dtype=parts_t.dtype).unsqueeze(-1)
                pooled = (parts_t * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            consistency_terms.append(F.mse_loss(whole_t, pooled))
        consistency = torch.stack(consistency_terms).mean()

        fixed_whole = F.mse_loss(state.whole_history[-1], state.whole_history[-2])
        fixed_parts = F.mse_loss(state.part_history[-1], state.part_history[-2])
        fixed_point = 0.5 * (fixed_whole + fixed_parts)

        trajectory = _zero_like(reference)
        if self.trajectory_weight != 0.0 and len(state.whole_history) > 2:
            step_sizes = []
            for previous, current in zip(state.whole_history[:-1], state.whole_history[1:]):
                step_sizes.append((current - previous).pow(2).mean())
            # Penalise increases in step size rather than movement itself.
            increases = [F.relu(b - a) for a, b in zip(step_sizes[:-1], step_sizes[1:])]
            if increases:
                trajectory = torch.stack(increases).mean()

        loss = (
            task
            + self.consistency_weight * consistency
            + self.fixed_point_weight * fixed_point
            + self.trajectory_weight * trajectory
        )
        terms = {
            "hermeneutic/total": loss,
            "hermeneutic/task": task,
            "hermeneutic/part_whole_consistency": consistency,
            "hermeneutic/fixed_point": fixed_point,
            "hermeneutic/trajectory": trajectory,
        }
        return ObjectiveOutput(loss=loss, terms=terms)


def aufhebung_step(
    thesis: Tensor,
    antithesis: Tensor,
    synthesis: Tensor,
    *,
    preservation_weight: float = 1.0,
    balance_weight: float = 1.0,
    lambda_preserve: Optional[float] = None,
    lambda_resolve: Optional[float] = None,
) -> Tensor:
    """Simple compatibility loss: preserve both poles and avoid one-sided collapse."""
    if lambda_preserve is not None:
        preservation_weight = float(lambda_preserve)
    if lambda_resolve is not None:
        balance_weight = float(lambda_resolve)
    keep_thesis = 1.0 - F.cosine_similarity(synthesis, thesis, dim=-1)
    keep_antithesis = 1.0 - F.cosine_similarity(synthesis, antithesis, dim=-1)
    preservation = (keep_thesis + keep_antithesis).mean()
    balance = (keep_thesis - keep_antithesis).abs().mean()
    return float(preservation_weight) * preservation + float(balance_weight) * balance


@dataclass
class DialecticalState:
    synthesis: Tensor
    agreement: Tensor
    conflict: Tensor
    candidate: Tensor
    learned_gate: Tensor
    conflict_candidate: Tensor
    legacy_loss: Optional[Tensor] = None

    def __iter__(self):
        """Allow legacy ``synthesis, loss = module(a, b)`` unpacking."""
        yield self.synthesis
        yield self.legacy_loss


class DialecticalSynthesis(nn.Module):
    """Agreement-aware differentiable proxy for dialectical *Aufhebung*.

    By default the operator is permutation-invariant in thesis and antithesis:
    swapping the two poles produces the same synthesis.  Set ``symmetric=False``
    only when their roles are semantically ordered.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: Optional[int] = None,
        agreement_temperature: float = 1.0,
        gamma: float = 0.5,
        dropout: float = 0.0,
        symmetric: bool = True,
        return_loss: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if agreement_temperature <= 0:
            raise ValueError("agreement_temperature must be positive")
        hidden = int(hidden_dim or max(dim, 64))
        self.dim = int(dim)
        self.agreement_temperature = float(agreement_temperature)
        self.gamma = float(gamma)
        self.symmetric = bool(symmetric)
        self.return_loss = bool(return_loss)
        self.network = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Tanh(),
        )
        self.output_gate = nn.Sequential(nn.Linear(4 * dim, dim), nn.Sigmoid())

    def _features(self, thesis: Tensor, antithesis: Tensor) -> Tensor:
        if self.symmetric:
            mean = 0.5 * (thesis + antithesis)
            absolute_difference = torch.abs(thesis - antithesis)
            product = thesis * antithesis
            second_moment = 0.5 * (thesis.pow(2) + antithesis.pow(2))
            return torch.cat(
                [mean, absolute_difference, product, second_moment], dim=-1
            )
        return torch.cat(
            [thesis, antithesis, thesis - antithesis, thesis * antithesis],
            dim=-1,
        )

    def forward(
        self,
        thesis: Tensor,
        antithesis: Tensor,
        return_loss: Optional[bool] = None,
    ) -> DialecticalState:
        if thesis.shape != antithesis.shape:
            raise ValueError("thesis and antithesis must have equal shapes")
        if thesis.ndim < 2 or thesis.shape[-1] != self.dim:
            raise ValueError(
                f"Expected tensors [..., {self.dim}], got {tuple(thesis.shape)}"
            )
        midpoint = 0.5 * (thesis + antithesis)
        features = self._features(thesis, antithesis)
        lift = self.network(features)
        candidate = midpoint + self.gamma * lift
        agreement = torch.exp(
            -torch.abs(thesis - antithesis) / self.agreement_temperature
        ).clamp(0.0, 1.0)
        conflict = 1.0 - agreement
        learned_gate = self.output_gate(features)
        conflict_candidate = learned_gate * candidate + (1.0 - learned_gate) * midpoint
        synthesis = agreement * midpoint + conflict * conflict_candidate
        compute_loss = self.return_loss if return_loss is None else bool(return_loss)
        compatibility_loss = (
            aufhebung_step(thesis, antithesis, synthesis)
            if compute_loss
            else None
        )
        return DialecticalState(
            synthesis=synthesis,
            agreement=agreement,
            conflict=conflict,
            candidate=candidate,
            learned_gate=learned_gate,
            conflict_candidate=conflict_candidate,
            legacy_loss=compatibility_loss,
        )


class DialecticalLoss(nn.Module):
    """Preserve agreement, resolve conflict, and require useful synthesis."""

    def __init__(
        self,
        *,
        preservation_weight: float = 0.1,
        novelty_weight: float = 0.02,
        novelty_margin: float = 0.1,
        resolution_weight: float = 0.25,
        resolution_margin: float = 0.0,
    ) -> None:
        super().__init__()
        self.preservation_weight = float(preservation_weight)
        self.novelty_weight = float(novelty_weight)
        self.novelty_margin = float(novelty_margin)
        self.resolution_weight = float(resolution_weight)
        self.resolution_margin = float(resolution_margin)

    def forward(
        self,
        thesis: Tensor,
        antithesis: Tensor,
        state: DialecticalState,
        *,
        synthesis_task_loss: Tensor,
        thesis_task_loss: Optional[Tensor] = None,
        antithesis_task_loss: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        if thesis.shape != antithesis.shape or thesis.shape != state.synthesis.shape:
            raise ValueError("thesis, antithesis, and synthesis shapes must match")

        task = synthesis_task_loss.mean() if synthesis_task_loss.ndim else synthesis_task_loss
        agreement_mass = state.agreement.sum().clamp_min(1e-8)
        preservation = (
            state.agreement * (state.synthesis - thesis).pow(2)
            + state.agreement * (state.synthesis - antithesis).pow(2)
        ).sum() / (2.0 * agreement_mass)

        midpoint = 0.5 * (thesis + antithesis)
        distance_from_midpoint = torch.abs(state.synthesis - midpoint)
        conflict_mass = state.conflict.sum().clamp_min(1e-8)
        novelty = (
            state.conflict
            * F.relu(self.novelty_margin - distance_from_midpoint).pow(2)
        ).sum() / conflict_mass

        resolution = _zero_like(task)
        if thesis_task_loss is not None and antithesis_task_loss is not None:
            thesis_baseline = thesis_task_loss.mean() if thesis_task_loss.ndim else thesis_task_loss
            antithesis_baseline = (
                antithesis_task_loss.mean()
                if antithesis_task_loss.ndim
                else antithesis_task_loss
            )
            best_parent = torch.minimum(
                thesis_baseline.detach(), antithesis_baseline.detach()
            )
            resolution = F.relu(task + self.resolution_margin - best_parent)

        loss = (
            task
            + self.preservation_weight * preservation
            + self.novelty_weight * novelty
            + self.resolution_weight * resolution
        )
        terms = {
            "dialectical/total": loss,
            "dialectical/task": task,
            "dialectical/preservation": preservation,
            "dialectical/novelty": novelty,
            "dialectical/resolution": resolution,
            "dialectical/mean_conflict": state.conflict.mean(),
        }
        return ObjectiveOutput(loss=loss, terms=terms)


class EpistemicVirtueLoss(nn.Module):
    """Unified, testable proxies for epistemic virtues.

    The names are philosophical labels for measurable behaviours, not claims
    that a tensor possesses moral character:

    * accuracy -> cross entropy;
    * calibration -> Brier score;
    * humility -> suppress confidence on current errors;
    * honesty -> confidence should not exceed supplied evidence/support;
    * openness -> react to relevant counterevidence but remain stable to an
      irrelevant perturbation;
    * restraint -> optional selective prediction / abstention objective.
    """

    def __init__(
        self,
        *,
        calibration_weight: float = 0.1,
        humility_weight: float = 0.1,
        honesty_weight: float = 0.1,
        openness_weight: float = 0.1,
        selectivity_weight: float = 0.25,
        confidence_ceiling_on_error: float = 0.5,
        counterevidence_margin: float = 0.1,
        target_coverage: float = 0.8,
        coverage_penalty_weight: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= confidence_ceiling_on_error <= 1.0:
            raise ValueError("confidence_ceiling_on_error must lie in [0, 1]")
        if not 0.0 <= target_coverage <= 1.0:
            raise ValueError("target_coverage must lie in [0, 1]")
        self.calibration_weight = float(calibration_weight)
        self.humility_weight = float(humility_weight)
        self.honesty_weight = float(honesty_weight)
        self.openness_weight = float(openness_weight)
        self.selectivity_weight = float(selectivity_weight)
        self.confidence_ceiling_on_error = float(confidence_ceiling_on_error)
        self.counterevidence_margin = float(counterevidence_margin)
        self.target_coverage = float(target_coverage)
        self.coverage_penalty_weight = float(coverage_penalty_weight)

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        *,
        support: Optional[Tensor] = None,
        counter_logits: Optional[Tensor] = None,
        counter_is_relevant: Optional[Tensor] = None,
        selection_logits: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        _require_rank(logits, 2, "logits")
        _require_rank(target, 1, "target")
        _require_same_batch(("logits", logits), ("target", target))
        batch, classes = logits.shape
        if classes < 2:
            raise ValueError("At least two classes are required")

        per_sample_ce = F.cross_entropy(logits, target, reduction="none")
        accuracy = per_sample_ce.mean()
        probabilities = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(target, num_classes=classes).to(probabilities.dtype)
        calibration = (probabilities - one_hot).pow(2).sum(dim=-1).mean()

        confidence, prediction = probabilities.max(dim=-1)
        wrong = prediction.ne(target).to(probabilities.dtype).detach()
        humility = (
            wrong
            * F.relu(confidence - self.confidence_ceiling_on_error).pow(2)
        ).sum() / wrong.sum().clamp_min(1.0)

        honesty = _zero_like(logits)
        if support is not None:
            if support.shape not in {(batch,), (batch, 1)}:
                raise ValueError("support must have shape [B] or [B, 1]")
            bounded_support = support.reshape(batch).to(logits.dtype).clamp(0.0, 1.0)
            honesty = F.relu(confidence - bounded_support).pow(2).mean()

        openness = _zero_like(logits)
        if counter_logits is not None:
            if counter_logits.shape != logits.shape:
                raise ValueError("counter_logits must match logits")
            if counter_is_relevant is None:
                raise ValueError(
                    "counter_is_relevant is required when counter_logits is supplied"
                )
            if counter_is_relevant.shape not in {(batch,), (batch, 1)}:
                raise ValueError("counter_is_relevant must have shape [B] or [B, 1]")
            relevance = counter_is_relevant.reshape(batch).to(logits.dtype).clamp(0.0, 1.0)
            js = js_divergence_from_logits(logits, counter_logits, reduction="none")
            react = relevance * F.relu(self.counterevidence_margin - js).pow(2)
            remain_stable = (1.0 - relevance) * js.pow(2)
            openness = (react + remain_stable).mean()

        selectivity = _zero_like(logits)
        coverage = torch.ones((), device=logits.device, dtype=logits.dtype)
        selective_risk = accuracy
        if selection_logits is not None:
            if selection_logits.shape not in {(batch,), (batch, 1)}:
                raise ValueError("selection_logits must have shape [B] or [B, 1]")
            accept = torch.sigmoid(selection_logits.reshape(batch))
            coverage = accept.mean()
            selective_risk = (accept * per_sample_ce).sum() / accept.sum().clamp_min(1e-8)
            coverage_penalty = F.relu(self.target_coverage - coverage).pow(2)
            selectivity = (
                selective_risk
                + self.coverage_penalty_weight * coverage_penalty
            )

        loss = (
            accuracy
            + self.calibration_weight * calibration
            + self.humility_weight * humility
            + self.honesty_weight * honesty
            + self.openness_weight * openness
            + self.selectivity_weight * selectivity
        )
        terms = {
            "virtue/total": loss,
            "virtue/accuracy_ce": accuracy,
            "virtue/calibration_brier": calibration,
            "virtue/humility": humility,
            "virtue/honesty": honesty,
            "virtue/openness": openness,
            "virtue/selectivity": selectivity,
            "virtue/coverage": coverage,
            "virtue/selective_risk": selective_risk,
        }
        return ObjectiveOutput(loss=loss, terms=terms)


class VirtueRegularizer(nn.Module):
    """Targeted regulariser for calibration, openness, and smoothness.

    Calibration is a monotonic desideratum and is therefore targeted near zero
    ECE.  Quantities that genuinely have a Goldilocks zone, such as ensemble
    disagreement, may be targeted at an interior value.

    ``target_humility`` is accepted for compatibility and maps to
    ``target_ece = 1 - target_humility``.
    """

    def __init__(
        self,
        *,
        target_ece: float = 0.01,
        target_humility: Optional[float] = None,
        target_open: Optional[float] = 0.1,
        target_conscientiousness: Optional[float] = None,
        target_consc: Optional[float] = None,
        beta_calibration: float = 1.0,
        beta_humility: Optional[float] = None,
        beta_open: float = 0.1,
        beta_conscientiousness: float = 0.0,
        beta_consc: Optional[float] = None,
        n_bins: int = 15,
        smoothness_samples: int = 1,
    ) -> None:
        super().__init__()
        if target_consc is not None:
            target_conscientiousness = float(target_consc)
        if target_humility is not None:
            target_ece = 1.0 - float(target_humility)
        if beta_humility is not None:
            beta_calibration = float(beta_humility)
        if beta_consc is not None:
            beta_conscientiousness = float(beta_consc)
        if not 0.0 <= target_ece <= 1.0:
            raise ValueError("target_ece must lie in [0, 1]")
        self.target_ece = float(target_ece)
        self.target_open = target_open
        self.target_conscientiousness = target_conscientiousness
        self.beta_calibration = float(beta_calibration)
        self.beta_open = float(beta_open)
        self.beta_conscientiousness = float(beta_conscientiousness)
        self.n_bins = int(n_bins)
        self.smoothness_samples = int(smoothness_samples)

    def forward(
        self,
        probabilities: Tensor,
        targets: Tensor,
        *,
        member_probabilities: Optional[Tensor] = None,
        member_probs: Optional[Tensor] = None,
        inputs: Optional[Tensor] = None,
        outputs: Optional[Tensor] = None,
        openness_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if member_probabilities is None:
            member_probabilities = member_probs
        soft_ece = soft_expected_calibration_error(
            probabilities, targets, n_bins=self.n_bins
        )
        total = self.beta_calibration * (soft_ece - self.target_ece).pow(2)
        if member_probabilities is not None and self.target_open is not None:
            per_sample_mi = ensemble_mutual_information(
                member_probabilities, reduction="none"
            )
            if openness_mask is not None:
                if openness_mask.shape not in {
                    per_sample_mi.shape,
                    per_sample_mi.shape + (1,),
                }:
                    raise ValueError("openness_mask must match the batch")
                mask = openness_mask.reshape_as(per_sample_mi).to(per_sample_mi.dtype)
                openness = (
                    (per_sample_mi - float(self.target_open)).pow(2) * mask
                ).sum() / mask.sum().clamp_min(1.0)
            else:
                openness = (per_sample_mi - float(self.target_open)).pow(2).mean()
            total = total + self.beta_open * openness
        if (
            inputs is not None
            and outputs is not None
            and self.target_conscientiousness is not None
            and self.beta_conscientiousness != 0.0
        ):
            smoothness = hutchinson_jacobian_frobenius(
                inputs,
                outputs,
                samples=self.smoothness_samples,
            )
            total = total + self.beta_conscientiousness * (
                smoothness - float(self.target_conscientiousness)
            ).pow(2)
        return total

    @torch.no_grad()
    def report(
        self,
        probabilities: Tensor,
        targets: Tensor,
        *,
        member_probabilities: Optional[Tensor] = None,
        member_probs: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        if member_probabilities is None:
            member_probabilities = member_probs
        result = {
            "ece": float(
                expected_calibration_error(
                    probabilities, targets, n_bins=self.n_bins
                ).cpu()
            ),
            "soft_ece": float(
                soft_expected_calibration_error(
                    probabilities, targets, n_bins=self.n_bins
                ).cpu()
            ),
        }
        result["humility(1-ECE)"] = 1.0 - result["ece"]
        if member_probabilities is not None:
            result["open(MI)"] = float(
                ensemble_mutual_information(member_probabilities).cpu()
            )
        return result


def perspectival_disagreement(
    member_probabilities: Tensor,
    *,
    metric: str = "js",
    eps: float = 1e-8,
) -> Tensor:
    """Return per-sample disagreement across ``K`` perspectives."""
    _require_rank(member_probabilities, 3, "member_probabilities")
    members, batch, _ = member_probabilities.shape
    if members < 2:
        raise ValueError("At least two perspectives are required")
    if metric == "mutual_information":
        return ensemble_mutual_information(member_probabilities, reduction="none")
    total = member_probabilities.new_zeros(batch)
    pairs = 0
    for left in range(members):
        for right in range(left + 1, members):
            p = member_probabilities[left]
            q = member_probabilities[right]
            if metric == "js":
                mixture = 0.5 * (p + q)
                divergence = 0.5 * (
                    (p * (p.clamp_min(eps).log() - mixture.clamp_min(eps).log())).sum(-1)
                    + (q * (q.clamp_min(eps).log() - mixture.clamp_min(eps).log())).sum(-1)
                )
            elif metric == "symmetric_kl":
                divergence = (
                    p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())
                ).sum(-1) + (
                    q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())
                ).sum(-1)
            else:
                raise ValueError(f"Unknown disagreement metric: {metric!r}")
            total = total + divergence
            pairs += 1
    return total / float(pairs)


class PerspectivalEnsemble(nn.Module):
    """Aggregate perspectives and abstain when their conflict is excessive."""

    def __init__(
        self,
        n_perspectives: int,
        *,
        abstain_threshold: float = 0.15,
        disagreement_metric: str = "js",
        target_disagreement: Optional[float] = None,
        disagreement_weight: float = 0.0,
        beta_disagree: Optional[float] = None,
    ) -> None:
        super().__init__()
        if n_perspectives < 2:
            raise ValueError("At least two perspectives are required")
        if beta_disagree is not None:
            disagreement_weight = float(beta_disagree)
        self.n_perspectives = int(n_perspectives)
        self.abstain_threshold = float(abstain_threshold)
        self.disagreement_metric = disagreement_metric
        self.target_disagreement = target_disagreement
        self.disagreement_weight = float(disagreement_weight)

    def forward(self, member_logits: Tensor) -> Dict[str, Tensor]:
        _require_rank(member_logits, 3, "member_logits")
        if member_logits.shape[0] != self.n_perspectives:
            raise ValueError("First dimension must equal n_perspectives")
        member_probabilities = F.softmax(member_logits, dim=-1)
        probabilities = member_probabilities.mean(dim=0)
        disagreement = perspectival_disagreement(
            member_probabilities, metric=self.disagreement_metric
        )
        abstain = disagreement > self.abstain_threshold
        regularizer = member_logits.new_zeros(())
        if self.target_disagreement is not None and self.disagreement_weight != 0.0:
            regularizer = self.disagreement_weight * (
                disagreement.mean() - float(self.target_disagreement)
            ).pow(2)
        return {
            "probs": probabilities,
            "probabilities": probabilities,
            "member_probabilities": member_probabilities,
            "abstain": abstain,
            "disagree": disagreement,
            "disagreement": disagreement,
            "reg": regularizer,
        }


@dataclass
class SemiosisState:
    interpretant: Tensor
    next_sign: Tensor
    next_interpretant: Tensor
    reconstructed_sign: Tensor
    reconstructed_object: Tensor


class TriadicSemiosis(nn.Module):
    """A differentiable sign-object-interpretant recursion.

    ``sign`` and ``object`` may come from different encoders/modalities.  The
    module creates an interpretant, turns it into a new sign, and interprets the
    new sign again against the same object.  Reconstruction heads make the
    triad grounded rather than a free-floating latent code.
    """

    def __init__(
        self,
        sign_dim: int,
        object_dim: int,
        interpretant_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "sign_dim": sign_dim,
            "object_dim": object_dim,
            "interpretant_dim": interpretant_dim,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        hidden = int(hidden_dim or max(interpretant_dim, 64))
        self.sign_dim = int(sign_dim)
        self.object_dim = int(object_dim)
        self.interpretant_dim = int(interpretant_dim)

        def mlp(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output_dim),
            )

        self.interpreter = mlp(sign_dim + object_dim, interpretant_dim)
        self.sign_generator = mlp(interpretant_dim + object_dim, sign_dim)
        self.sign_decoder = mlp(object_dim + interpretant_dim, sign_dim)
        self.object_decoder = mlp(sign_dim + interpretant_dim, object_dim)

    def interpret(self, sign: Tensor, object_representation: Tensor) -> Tensor:
        return self.interpreter(torch.cat([sign, object_representation], dim=-1))

    def forward(self, sign: Tensor, object_representation: Tensor) -> SemiosisState:
        _require_rank(sign, 2, "sign")
        _require_rank(object_representation, 2, "object_representation")
        _require_same_batch(("sign", sign), ("object", object_representation))
        if sign.shape[-1] != self.sign_dim:
            raise ValueError(f"Expected sign dimension {self.sign_dim}")
        if object_representation.shape[-1] != self.object_dim:
            raise ValueError(f"Expected object dimension {self.object_dim}")

        interpretant = self.interpret(sign, object_representation)
        next_sign = self.sign_generator(
            torch.cat([interpretant, object_representation], dim=-1)
        )
        next_interpretant = self.interpret(next_sign, object_representation)
        reconstructed_sign = self.sign_decoder(
            torch.cat([object_representation, interpretant], dim=-1)
        )
        reconstructed_object = self.object_decoder(
            torch.cat([sign, interpretant], dim=-1)
        )
        return SemiosisState(
            interpretant=interpretant,
            next_sign=next_sign,
            next_interpretant=next_interpretant,
            reconstructed_sign=reconstructed_sign,
            reconstructed_object=reconstructed_object,
        )


class SemiosisLoss(nn.Module):
    """Grounding, recursive consistency, and anti-collapse for a semiotic triad."""

    def __init__(
        self,
        *,
        sign_reconstruction_weight: float = 1.0,
        object_reconstruction_weight: float = 1.0,
        recursion_weight: float = 0.25,
        variance_weight: float = 0.05,
        covariance_weight: float = 0.005,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.sign_reconstruction_weight = float(sign_reconstruction_weight)
        self.object_reconstruction_weight = float(object_reconstruction_weight)
        self.recursion_weight = float(recursion_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.target_std = float(target_std)

    def forward(
        self,
        sign: Tensor,
        object_representation: Tensor,
        state: SemiosisState,
        *,
        task_loss: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        if sign.shape != state.reconstructed_sign.shape:
            raise ValueError("reconstructed_sign must match sign")
        if object_representation.shape != state.reconstructed_object.shape:
            raise ValueError("reconstructed_object must match object_representation")

        task = _zero_like(sign) if task_loss is None else task_loss
        if task.ndim:
            task = task.mean()
        sign_reconstruction = F.mse_loss(state.reconstructed_sign, sign)
        object_reconstruction = F.mse_loss(
            state.reconstructed_object, object_representation
        )

        first = F.normalize(state.interpretant, dim=-1)
        second = F.normalize(state.next_interpretant, dim=-1)
        recursion = (1.0 - (first * second).sum(dim=-1)).mean()

        combined_interpretants = torch.cat(
            [state.interpretant, state.next_interpretant], dim=0
        )
        variance, covariance = variance_covariance_regularizer(
            combined_interpretants, target_std=self.target_std
        )

        loss = (
            task
            + self.sign_reconstruction_weight * sign_reconstruction
            + self.object_reconstruction_weight * object_reconstruction
            + self.recursion_weight * recursion
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        terms = {
            "semiosis/total": loss,
            "semiosis/task": task,
            "semiosis/sign_reconstruction": sign_reconstruction,
            "semiosis/object_reconstruction": object_reconstruction,
            "semiosis/recursive_consistency": recursion,
            "semiosis/variance": variance,
            "semiosis/covariance": covariance,
        }
        return ObjectiveOutput(loss=loss, terms=terms)


class PhilosophiaWrapper(nn.Module):
    """Thin add-on that attaches epoche and targeted virtue terms to a model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        use_epoche: bool = False,
        use_virtue: bool = False,
        epoche_kwargs: Optional[Dict[str, object]] = None,
        virtue_kwargs: Optional[Dict[str, object]] = None,
        auxiliary_weight: float = 1.0,
        aux_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        if aux_weight is not None:
            auxiliary_weight = float(aux_weight)
        self.model = model
        self.auxiliary_weight = float(auxiliary_weight)
        self.epoche = (
            EpocheRegularizer(model, **(epoche_kwargs or {}))
            if use_epoche
            else None
        )
        self.virtue = (
            VirtueRegularizer(**(virtue_kwargs or {}))
            if use_virtue
            else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)

    def aux_loss(
        self,
        inputs: Tensor,
        logits: Tensor,
        *,
        targets: Optional[Tensor] = None,
        member_probabilities: Optional[Tensor] = None,
        member_probs: Optional[Tensor] = None,
        openness_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if member_probabilities is None:
            member_probabilities = member_probs
        total = logits.new_zeros(())
        if self.epoche is not None:
            total = total + self.epoche(inputs, logits_full=logits)
        if self.virtue is not None and targets is not None:
            total = total + self.virtue(
                F.softmax(logits, dim=-1),
                targets,
                member_probabilities=member_probabilities,
                openness_mask=openness_mask,
            )
        return self.auxiliary_weight * total
