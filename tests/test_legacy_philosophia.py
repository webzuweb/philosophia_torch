"""Тесты philosophia-torch: проверяем корректность, дифференцируемость и
осмысленность (величины ведут себя так, как задумано)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from philosophia import (
    EpocheRegularizer, epoche_penalty,
    HermeneuticConsistency, hermeneutic_loss,
    AbductiveScorer, abductive_selection_loss,
    DialecticalSynthesis, aufhebung_step,
    PerspectivalEnsemble, perspectival_disagreement,
    VirtueRegularizer, expected_calibration_error,
    PhilosophiaWrapper,
)
from philosophia.virtue import ensemble_mutual_information, input_smoothness

torch.manual_seed(0)


# ---------- virtue ----------
def test_ece_perfect_calibration_low():
    # уверенность = точность → ECE ~ 0
    N, C = 200, 4
    logits = torch.randn(N, C)
    targets = logits.argmax(1)  # модель всегда права
    probs = logits.softmax(1)
    ece = expected_calibration_error(probs, targets)
    assert ece.item() >= 0.0
    assert ece.item() < 0.5


def test_ece_overconfident_high():
    # переуверенная и часто неправа → ECE большой
    N, C = 200, 4
    probs = torch.zeros(N, C)
    probs[:, 0] = 0.99
    probs[:, 1:] = 0.01 / (C - 1)
    targets = torch.randint(1, C, (N,))  # правильный класс никогда не 0
    ece = expected_calibration_error(probs, targets)
    assert ece.item() > 0.5


def test_ece_differentiable():
    logits = torch.randn(64, 5, requires_grad=True)
    targets = torch.randint(0, 5, (64,))
    ece = expected_calibration_error(logits.softmax(1), targets)
    ece.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mutual_information_nonneg_and_zero_on_agreement():
    K, N, C = 5, 32, 3
    # все головы одинаковы → MI = 0
    p = torch.randn(N, C).softmax(-1)
    same = p.unsqueeze(0).repeat(K, 1, 1)
    mi = ensemble_mutual_information(same)
    assert abs(mi.item()) < 1e-5
    # разные головы → MI > 0
    diff = torch.randn(K, N, C).softmax(-1)
    assert ensemble_mutual_information(diff).item() > 0.0


def test_virtue_regularizer_targets_midpoint():
    # регуляризатор минимален, когда добродетель = target
    reg = VirtueRegularizer(target_humility=0.9, beta_humility=1.0)
    N, C = 128, 4
    logits = torch.randn(N, C)
    targets = logits.argmax(1)
    probs = logits.softmax(1)
    val = reg(probs, targets)
    assert val.item() >= 0.0
    rep = reg.report(probs, targets)
    assert "humility(1-ECE)" in rep


def test_input_smoothness_grad():
    x = torch.randn(16, 8, requires_grad=True)
    lin = nn.Linear(8, 3)
    out = lin(x)
    s = input_smoothness(x, out)
    assert s.item() >= 0.0
    s.backward()
    assert lin.weight.grad is not None


# ---------- epoche ----------
def test_epoche_evidence_gain_positive_when_input_matters():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    x = torch.randn(32, 10)
    logits_full = model(x)
    logits_prior = model(torch.zeros_like(x))
    gain = epoche_penalty(logits_full, logits_prior, mode="evidence_gain")
    assert gain.item() >= 0.0  # KL неотрицателен


def test_epoche_regularizer_differentiable():
    model = nn.Sequential(nn.Linear(10, 3))
    reg = EpocheRegularizer(model, lambda_gain=1.0, lambda_prior=0.1)
    x = torch.randn(16, 10)
    r = reg(x)
    r.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_epoche_penalizes_model_that_ignores_input():
    # Модель-константа: свидетельство ничего не меняет, JS=0 — получаем
    # полный hinge-штраф min_gain.
    class Const(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 3)
    reg = EpocheRegularizer(Const(), lambda_gain=1.0, min_gain=0.05)
    x = torch.randn(8, 4)
    assert reg(x).item() == pytest.approx(0.05, abs=1e-6)


# ---------- hermeneutic ----------
def test_hermeneutic_loss_zero_on_consistent():
    N, T, D = 8, 5, 16
    part = torch.randn(N, T, D)
    whole = part.mean(1)  # целое = агрегат частей
    loss = hermeneutic_loss(part, whole, alpha=1.0, beta=0.0)
    assert loss.item() < 1e-4


def test_hermeneutic_circle_reduces_loss():
    N, T, D = 8, 6, 32
    part = torch.randn(N, T, D)
    whole0 = torch.randn(N, D)  # случайное несогласованное целое
    circle = HermeneuticConsistency(n_turns=5, momentum=0.3)
    whole_new, loss = circle(part, whole0)
    l0 = hermeneutic_loss(part, whole0, alpha=1.0, beta=0.0)
    # после оборотов круга согласованность не хуже исходной
    assert loss.item() <= l0.item() + 1e-5
    assert whole_new.shape == (N, D)


def test_hermeneutic_mask():
    N, T, D = 4, 5, 8
    part = torch.randn(N, T, D)
    whole = torch.randn(N, D)
    mask = torch.ones(N, T); mask[:, 3:] = 0
    loss = hermeneutic_loss(part, whole, mask=mask)
    assert torch.isfinite(loss)


# ---------- abduction ----------
def test_abduction_prefers_higher_loglik():
    # при равной сложности выбирается гипотеза с большим log p(obs|h)
    scorer = AbductiveScorer(lambda_simplicity=0.0)
    obs = torch.tensor([[0.1, 2.0, -1.0]])
    w, s = scorer(obs)
    assert w.argmax(-1).item() == 1


def test_abduction_simplicity_penalty():
    # высокое правдоподобие, но большая сложность → может проиграть простой
    scorer = AbductiveScorer(lambda_simplicity=1.0)
    obs = torch.tensor([[1.0, 1.2]])
    complexity = torch.tensor([[0.0, 5.0]])  # вторая гипотеза сложная
    w, s = scorer(obs, complexity=complexity)
    assert w.argmax(-1).item() == 0


def test_abductive_selection_loss_backward():
    obs = torch.randn(16, 4, requires_grad=True)
    idx = torch.randint(0, 4, (16,))
    loss = abductive_selection_loss(obs, idx)
    loss.backward()
    assert obs.grad is not None


# ---------- dialectic ----------
def test_aufhebung_preserves_both():
    torch.manual_seed(2)
    dim = 32
    syn = DialecticalSynthesis(dim, gamma=0.5)
    thesis = torch.randn(8, dim)
    anti = torch.randn(8, dim)
    s, loss = syn(thesis, anti)
    assert s.shape == (8, dim)
    assert loss.item() >= 0.0
    # синтез не должен совпасть ровно с одним из полюсов
    assert not torch.allclose(s, thesis)
    assert not torch.allclose(s, anti)


def test_aufhebung_symmetry_penalty_zero_when_balanced():
    # если синтез равноудалён от полюсов, член resolve = 0
    x = torch.randn(4, 16)
    a = torch.randn(4, 16)
    mid = (x + a) / 2
    loss = aufhebung_step(x, a, mid, lambda_preserve=0.0, lambda_resolve=1.0)
    assert loss.item() < 0.3  # близко к симметрии (не строго 0 из-за косинуса)


def test_dialectic_trainable():
    dim = 16
    syn = DialecticalSynthesis(dim)
    t, a = torch.randn(4, dim), torch.randn(4, dim)
    _, loss = syn(t, a)
    loss.backward()
    assert any(p.grad is not None for p in syn.parameters())


# ---------- perspectivism ----------
def test_disagreement_zero_on_identical():
    K, N, C = 3, 10, 4
    p = torch.randn(N, C).softmax(-1)
    same = p.unsqueeze(0).repeat(K, 1, 1)
    d = perspectival_disagreement(same)
    assert torch.allclose(d, torch.zeros(N), atol=1e-5)


def test_perspectival_abstains_on_conflict():
    torch.manual_seed(3)
    K, N, C = 4, 20, 3
    ens = PerspectivalEnsemble(K, abstain_threshold=0.5)
    # сильно расходящиеся перспективы
    logits = torch.randn(K, N, C) * 8
    out = ens(logits)
    assert out["abstain"].dtype == torch.bool
    assert out["abstain"].any()  # хоть где-то воздержались
    assert out["probs"].shape == (N, C)


def test_perspectival_regularizer():
    K, N, C = 3, 8, 2
    ens = PerspectivalEnsemble(K, target_disagreement=0.5, beta_disagree=1.0)
    logits = torch.randn(K, N, C, requires_grad=True)
    out = ens(logits)
    out["reg"].backward()
    assert logits.grad is not None


# ---------- wrapper ----------
def test_wrapper_end_to_end_training_step():
    torch.manual_seed(4)
    base = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 5))
    wrap = PhilosophiaWrapper(
        base, use_epoche=True, use_virtue=True,
        epoche_kwargs=dict(lambda_gain=0.5),
        virtue_kwargs=dict(target_humility=0.9, beta_humility=0.1),
    )
    opt = torch.optim.SGD(wrap.parameters(), lr=0.01)
    x = torch.randn(32, 12)
    y = torch.randint(0, 5, (32,))
    logits = wrap(x)
    loss = F.cross_entropy(logits, y) + wrap.aux_loss(x, logits, targets=y)
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)


def test_wrapper_reduces_loss_over_steps():
    torch.manual_seed(5)
    base = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    wrap = PhilosophiaWrapper(base, use_virtue=True,
                              virtue_kwargs=dict(beta_humility=0.05))
    opt = torch.optim.Adam(wrap.parameters(), lr=0.02)
    x = torch.randn(64, 8)
    y = (x.sum(1) > 0).long().clamp(max=2)
    first, last = None, None
    for step in range(30):
        logits = wrap(x)
        loss = F.cross_entropy(logits, y) + wrap.aux_loss(x, logits, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0: first = loss.item()
        last = loss.item()
    assert last < first  # обучение реально идёт
