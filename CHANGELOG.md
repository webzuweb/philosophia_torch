# Changelog

## 1.0.0

- Объединены `philosophia-torch` 0.1.0 и `philosophical_learning` 0.2.0.
- Добавлены classic hard ECE и отдельный differentiable soft-ECE.
- `VirtueRegularizer` разделяет monotonic calibration и Goldilocks targets.
- Input smoothness заменён на Hutchinson estimator нормы якобиана.
- `EpocheLoss` поддерживает full, bracketed, prior-only и no-evidence views.
- Сохранён compact `EpocheRegularizer` с bounded JS + hinge.
- Добавлены lightweight и trainable варианты герменевтического круга.
- `DialecticalSynthesis` стал симметричным по умолчанию.
- Добавлены `PerspectivalEnsemble` и bounded JS disagreement.
- Сохранены `TriadicSemiosis` и `SemiosisLoss`.
- Добавлен `PhilosophiaWrapper` и compatibility imports.
- Добавлены два multi-seed benchmark и JSON с полными результатами.
- 45 тестов, включая оба legacy-набора.
