from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pytest
from visual_control_pkg.metrics import (
    AccumulatorMetric,
    ComposeMetric,
    FunctionalMetric,
)

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
    assert metric._state is None and metric._count == 0
    assert np.all(np.isnan(metric.compute()))

    # Test that updates with other argnames do not change state
    metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    assert metric._state is None and metric._count == 0
    assert np.all(np.isnan(metric.compute()))


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
        name="func_acc",
        metric=AccumulatorMetric(name="acc", argname="sample", red=red),
        func=func,
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
    assert metric._metric._state is None and metric._metric._count == 0
    assert np.all(np.isnan(metric.compute()))

    # Test that updates with other argnames do not change state
    metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    assert metric._metric._state is None and metric._metric._count == 0
    assert np.all(np.isnan(metric.compute()))


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
    computed_metrics = composed_metric.compute()
    assert np.all(np.isnan(computed_metrics["acc"]))
    assert np.all(np.isnan(computed_metrics["func_acc"]))

    # Test that partial updates work correctly
    composed_metric.update(sample=sample)
    computed_metrics = composed_metric.compute()
    composed_metric.reset()
    assert not np.any(np.isnan(computed_metrics["acc"]))
    assert np.all(np.isnan(computed_metrics["func_acc"]))

    # Test that updating with missing argnames does not change state
    composed_metric.update(other_values=np.random.normal(0, 1, size=(10, 2)))
    computed_metrics = composed_metric.compute()
    assert np.all(np.isnan(computed_metrics["acc"]))
    assert np.all(np.isnan(computed_metrics["func_acc"]))
