# philosophia-torch 1.0.0

Экспериментальный add-on поверх PyTorch: философски мотивированные, но
технически проверяемые функции потерь, регуляризаторы и refinement-блоки.

> Это не утверждение, что тензоры обладают сознанием, добродетелью или
> «понимают» философов. Каждый компонент — инженерная операционализация с
> явным состоянием, формулой, логируемыми членами и возможностью проиграть
> обычному baseline.

## Что объединено в версии 1.0

Версия собирает лучшее из двух прототипов:

- практичный wrapper, perspectivism, однофайловый модуль и calibration-demo;
- подробные multi-view losses, fixed-point герменевтика, task-aware диалектика,
  эпистемические добродетели и Peircean semiosis;
- обратная совместимость с импортами обоих вариантов.

## Установка

Минимальная зависимость — `torch`.

```bash
pip install torch
```

Для тестов:

```bash
pip install pytest
```

Запуск прямо из распакованной папки:

```bash
PYTHONPATH=. python -c "import philosophia_torch; print(philosophia_torch.__version__)"
```

Доступны три формы импорта:

```python
from philosophia_torch import EpocheLoss, VirtueRegularizer
from philosophia import PhilosophiaWrapper
from philosophical_learning import TriadicSemiosis
```

`philosophia_torch.py` можно отдельно скопировать в свой проект: он зависит
только от PyTorch.

## Быстрый старт

```python
import torch.nn as nn
import torch.nn.functional as F
from philosophia_torch import PhilosophiaWrapper

base = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)

model = PhilosophiaWrapper(
    base,
    use_virtue=True,
    virtue_kwargs={
        "target_ece": 0.01,
        "beta_calibration": 8.0,
        "target_open": None,
    },
)

logits = model(x)
loss = F.cross_entropy(logits, y) + model.aux_loss(x, logits, targets=y)
loss.backward()
```

## Компоненты

| Компонент | Назначение |
|---|---|
| `AbductiveScorer` | выбор готовой гипотезы по likelihood, simplicity и conflict |
| `AbductiveHypothesisLoss` | soft best-explanation для банка нейронных гипотез |
| `EpocheRegularizer` | compact full-vs-no-evidence evidence gain |
| `EpocheLoss` | full / bracketed / prior-only / no-evidence objective |
| `HermeneuticConsistency` | лёгкий attention-based part→whole цикл |
| `HermeneuticRefiner` | обучаемое двустороннее обновление частей и целого |
| `HermeneuticCircleLoss` | consistency, fixed-point и trajectory penalties |
| `DialecticalSynthesis` | agreement-aware симметричный latent synthesis |
| `DialecticalLoss` | task, preservation, resolution и novelty |
| `PerspectivalEnsemble` | disagreement и abstention для нескольких перспектив |
| `VirtueRegularizer` | target soft-ECE, MI и Jacobian smoothness |
| `EpistemicVirtueLoss` | accuracy, Brier, humility, honesty, openness, selectivity |
| `TriadicSemiosis` | sign→object→interpretant→next sign recursion |
| `SemiosisLoss` | reconstruction, recursive consistency и anti-collapse |
| `PhilosophiaWrapper` | подключение compact regularizers к любой `nn.Module` |

## Важная разница между двумя вариантами эпохе́

`EpocheRegularizer` отвечает на вопрос: **текущее свидетельство вообще меняет
ответ относительно режима без свидетельства?**

`EpocheLoss` может отвечать на другой вопрос: **сохранится ли правильный ответ,
если удалить заранее указанный shortcut, и что shortcut предсказывает сам по
себе?**

```python
from philosophia_torch import EpocheLoss

objective = EpocheLoss(
    bracketed_task_weight=1.0,
    consistency_weight=0.5,
    prior_uniformity_weight=1.0,
)

out = objective(
    full_logits,
    targets,
    bracketed_logits,
    prior_only_logits=prior_only_logits,
)

out.loss.backward()
print(out.detached_terms())
```

## Эксперименты

### Калибровка

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python examples/demo_calibration.py --seeds 10
```

На включённой синтетической задаче:

| Метод | Accuracy | Hard ECE ↓ | Soft ECE ↓ | NLL ↓ |
|---|---:|---:|---:|---:|
| baseline | 0.8629 ± 0.0199 | 0.1018 ± 0.0189 | 0.1253 ± 0.0207 | 0.6623 ± 0.1542 |
| + virtue | 0.8714 ± 0.0216 | 0.0814 ± 0.0180 | 0.1052 ± 0.0202 | 0.4600 ± 0.1145 |

### Shortcut-shift для эпохе́

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python examples/demo_shortcut_epoche.py --seeds 10
```

| Метод | ID accuracy | OOD accuracy | Shortcut-only accuracy |
|---|---:|---:|---:|
| baseline | 0.9784 ± 0.0038 | 0.5382 ± 0.0252 | 0.9695 ± 0.0028 |
| compact | 0.9789 ± 0.0037 | 0.5404 ± 0.0233 | 0.9695 ± 0.0028 |
| explicit | 0.9778 ± 0.0049 | 0.6078 ± 0.0198 | 0.6654 ± 0.0908 |
| hybrid | 0.9779 ± 0.0055 | 0.6088 ± 0.0180 | 0.6488 ± 0.1015 |

JSON с каждым seed лежит в `benchmarks/`.

## Тесты

```bash
PYTHONPATH=. python -m pytest -q
```

Ожидаемый результат:

```text
45 passed
```

В тестах сохранены оба legacy-набора и добавлены проверки:

- hard/soft ECE и gradient flow;
- Hutchinson Jacobian norm;
- четыре views эпохе́;
- двусторонняя герменевтика и fixed point;
- swap-invariance диалектического synthesis;
- bounded JS и abstention;
- semiosis anti-collapse;
- совместимость старых аргументов и импортов.

## Ограничения

- Это research prototype, не production framework.
- Философские названия — смысловые ярлыки для измеримых поведений.
- Synthetic benchmarks не заменяют сравнение на Waterbirds, CIFAR-C,
  long-document QA, real OOD и multimodal datasets.
- Явный `EpocheLoss` требует предметно осмысленного способа построить
  bracketed и prior-only views.
- `HermeneuticRefiner`, `DialecticalSynthesis` и `TriadicSemiosis` дают
  представления; decoder/head остаётся частью вашей модели.

## Статья и отчёт

- `article/philosophy_neural_learning_habr_merged.md`
- `article/Filosofiya-obucheniya-neyrosetey-Habr-merged.docx`
- `COMPARISON_AND_MERGE_REPORT.md`
- `figures/loss_surfaces/` — формульные двумерные срезы функций потерь

## Лицензия

MIT.
