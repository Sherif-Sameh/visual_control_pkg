from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pytest

from visual_control_pkg.metrics import AccumulatorMetric, ComposeMetric, FunctionalMetric

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.unit
@pytest.mark.parametrize("red", ["sum", "mean", "cnt"])
def test_accumulator_metric(red: Literal["sum", "mean", "cnt"]) -> None:
    n_samples = 5
    sample_size = 2
    metric = AccumulatorMetric(name="acc", argname="sample", red=red)

    # Test `update()` and `compute()` methods
    samples = []
    for _ in range(n_samples):
        sample = np.random.normal(0, 1, size=(sample_size, 3))
        samples.append(sample.copy())
        metric.update(sample=sample)
    match red:
        case "sum":
            true_values = np.concatenate(samples, axis=0).sum(axis=0)
        case "mean":
            true_values = np.concatenate(samples, axis=0).mean(axis=0)
        case "cnt":
            true_values = np.array([n_samples * sample_size])
    assert np.allclose(metric.compute(), true_values)

    # Test `reset()` method
    metric.reset()
    state = metric._state
    assert np.allclose(state, np.zeros_like(state)) and metric._count == 0

    # Test that updates with other argnames do not change state
    metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    state = metric._state
    assert np.allclose(state, np.zeros_like(state)) and metric._count == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "func,red", [(np.fabs, "sum"), (lambda x: x + 1, "mean"), (np.square, "cnt")]
)
def test_functional_metric(
    func: Callable[[NDArray], NDArray], red: Literal["sum", "mean", "cnt"]
) -> None:
    n_samples = 5
    sample_size = 2
    metric = FunctionalMetric(
        name="func_acc", metric=AccumulatorMetric(name="acc", argname="sample", red=red), func=func
    )

    # Test `update()` and `compute()` methods
    func_samples = []
    for _ in range(n_samples):
        sample = np.random.normal(0, 1, size=(sample_size, 3))
        func_samples.append(func(sample))
        metric.update(sample=sample)
    match red:
        case "sum":
            true_values = np.concatenate(func_samples, axis=0).sum(axis=0)
        case "mean":
            true_values = np.concatenate(func_samples, axis=0).mean(axis=0)
        case "cnt":
            true_values = np.array([n_samples * sample_size])
    assert np.allclose(metric.compute(), true_values)

    # Test `reset()` method
    metric.reset()
    state = metric._metric._state
    assert np.allclose(state, np.zeros_like(state)) and metric._metric._count == 0

    # Test that updates with other argnames do not change state
    metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    state = metric._metric._state
    assert np.allclose(state, np.zeros_like(state)) and metric._metric._count == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "func,red", [(np.fabs, "sum"), (lambda x: x + 1, "mean"), (np.square, "cnt")]
)
def test_compose_metric(
    func: Callable[[NDArray], NDArray], red: Literal["sum", "mean", "cnt"]
) -> None:
    n_samples = 5
    sample_size = 2
    metric_1 = AccumulatorMetric(name="acc", argname="sample", red=red)
    metric_2 = FunctionalMetric(
        name="func_acc",
        metric=AccumulatorMetric(name="acc", argname="func_sample", red=red),
        func=func,
    )
    composed_metric = ComposeMetric(metrics=[metric_1, metric_2])

    # Test `update()` and `compute()` methods
    samples, func_samples = [], []
    for _ in range(n_samples):
        sample = np.random.normal(0, 1, size=(sample_size, 3))
        samples.append(sample.copy())
        func_samples.append(func(sample))
        composed_metric.update(sample=sample, func_sample=sample)
    match red:
        case "sum":
            true_values_1 = np.concatenate(samples, axis=0).sum(axis=0)
            true_values_2 = np.concatenate(func_samples, axis=0).sum(axis=0)
        case "mean":
            true_values_1 = np.concatenate(samples, axis=0).mean(axis=0)
            true_values_2 = np.concatenate(func_samples, axis=0).mean(axis=0)
        case "cnt":
            true_values_1 = np.array([n_samples * sample_size])
            true_values_2 = np.array([n_samples * sample_size])
    computed_metrics = composed_metric.compute()
    assert np.allclose(computed_metrics["acc"], true_values_1)
    assert np.allclose(computed_metrics["func_acc"], true_values_2)

    # Test `reset()` method
    composed_metric.reset()
    state_1 = composed_metric._metrics[0]._state
    count_1 = composed_metric._metrics[0]._count
    state_2 = composed_metric._metrics[1]._metric._state
    count_2 = composed_metric._metrics[1]._metric._count
    assert np.allclose(state_1, np.zeros_like(state_1)) and count_1 == 0
    assert np.allclose(state_2, np.zeros_like(state_2)) and count_2 == 0

    # Test that partial updates work correctly
    composed_metric.update(sample=sample)
    state_1 = composed_metric._metrics[0]._state
    count_1 = composed_metric._metrics[0]._count
    state_2 = composed_metric._metrics[1]._metric._state
    count_2 = composed_metric._metrics[1]._metric._count
    assert not np.allclose(state_1, np.zeros_like(state_1)) and count_1 != 0
    assert np.allclose(state_2, np.zeros_like(state_2)) and count_2 == 0

    # Test that updating with missing argnames does not change state
    composed_metric.reset()
    composed_metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    state_1 = composed_metric._metrics[0]._state
    count_1 = composed_metric._metrics[0]._count
    state_2 = composed_metric._metrics[1]._metric._state
    count_2 = composed_metric._metrics[1]._metric._count
    assert np.allclose(state_1, np.zeros_like(state_1)) and count_1 == 0
    assert np.allclose(state_2, np.zeros_like(state_2)) and count_2 == 0
