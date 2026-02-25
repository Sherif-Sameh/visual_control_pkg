from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visual_control_pkg.loggers import ConsoleLogger
from visual_control_pkg.loggers.csv import CSVLogger
from visual_control_pkg.loggers.wandb import WandBLogger
from visual_control_pkg.metrics import AccumulatorMetric, ComposeMetric


@pytest.mark.unit
def test_console_logger(capsys: pytest.CaptureFixture) -> None:
    with capsys.disabled():
        print("\n")
        # Initialize some metrics to log
        metric1 = AccumulatorMetric(name="PosError", argname="pos", red="mean")
        metric2 = AccumulatorMetric(name="RotError", argname="rot", red="mean")
        metrics = ComposeMetric(metrics=[metric1, metric2])

        # Update metrics with synthetic data
        pos = np.random.normal(0, 1, size=(10, 3))
        rot = np.random.normal(0, 1, size=(10, 4))
        metrics.update(pos=pos, rot=rot)

        # Initialize and test standard console logger
        print("Console Logger (n_log=1, n_flush=1, filter=None, config=Default)")
        logger = ConsoleLogger(n_log=1, n_flush=1, filter=None)
        logger.log(step=0.0, metrics=metrics.compute())
        print("\n")
        assert logger._log == {}

        # Initialize and test console logger with n_log and n_flush != 1
        n_log, n_flush = 2, 4
        print(f"Console Logger (n_log={n_log}, n_flush={n_flush}, filter=None, config=Default)")
        logger = ConsoleLogger(n_log=n_log, n_flush=n_flush, filter=None)
        for i in range(n_flush):
            metrics.reset()
            pos = np.random.normal(0, 1, size=(2, 3))
            rot = np.random.normal(0, 1, size=(2, 4))
            metrics.update(pos=pos, rot=rot)
            logger.log(step=0.1 * i, metrics=metrics.compute())
            if i >= (n_log - 1) and i != (n_flush - 1):
                assert logger._log != {}
        assert logger._log == {}
        print("\n")

        # Initialize and test console logger with filter
        print("Console Logger (n_log=1, n_flush=1, filter=Pos, config=Default)")
        logger = ConsoleLogger(n_log=1, n_flush=1, filter="Pos")
        metrics.update(pos=pos, rot=rot)
        logger.log(step=0.0, metrics=metrics.compute())
        assert logger._log == {}
        print("\n")

        # Initialize and test console logger with filter
        print("Console Logger (n_log=1, n_flush=1, filter=None, config=(precision=2, sign='-'))")
        logger = ConsoleLogger(
            n_log=1,
            n_flush=1,
            filter=None,
            config=ConsoleLogger.ArrayPrintOptions(precision=2, sign="-"),
        )
        metrics.update(pos=pos, rot=rot)
        logger.log(step=0.0, metrics=metrics.compute())
        assert logger._log == {}
        print("\n")


@pytest.mark.unit
def test_csv_logger():
    # Initialize some metrics to log
    metric1 = AccumulatorMetric(name="PosError", argname="pos", red="mean")
    metric2 = AccumulatorMetric(name="RotError", argname="rot", red="mean")
    metrics = ComposeMetric(metrics=[metric1, metric2])

    # Update metrics with synthetic data
    pos = np.random.normal(0, 1, size=(10, 3))
    rot = np.random.normal(0, 1, size=(10, 4))
    metrics.update(pos=pos, rot=rot)

    # Initialize and test standard CSV logger
    log_dir = Path(__file__).parent
    logger = CSVLogger(n_log=1, n_flush=1, filter=None, dir=log_dir)
    logger.log(step=0.0, metrics=metrics.compute())
    assert logger._log == {}
    df = pd.read_csv(logger._path)
    df.value = df.value.apply(literal_eval)
    df.set_index("name", inplace=True)
    assert df.loc[metric1.name] is not None
    assert df.loc[metric2.name] is not None
    assert df.shape[0] == 2

    # Attempt to append more logs to the same CSV file
    for i in range(1, 3):
        metrics.reset()
        pos = np.random.normal(0, 1, size=(2, 3))
        rot = np.random.normal(0, 1, size=(2, 4))
        metrics.update(pos=pos, rot=rot)
        logger.log(step=i * 0.05, metrics=metrics.compute())
    df = pd.read_csv(logger._path)
    df.value = df.value.apply(literal_eval)
    df.set_index("name", inplace=True)
    assert df.loc[metric1.name] is not None
    assert df.loc[metric2.name] is not None
    assert df.shape[0] == 6
    logger._path.unlink()  # Clean up test CSV file


@pytest.mark.unit
def test_wandb_logger(capsys: pytest.CaptureFixture):
    with capsys.disabled():
        # Initialize some metrics to log
        metric1 = AccumulatorMetric(name="PosError", argname="pos", red="mean")
        metric2 = AccumulatorMetric(name="RotError", argname="rot", red="mean")
        metrics = ComposeMetric(metrics=[metric1, metric2])

        # Update metrics with synthetic data
        pos = np.random.normal(0, 1, size=(10, 3))
        rot = np.random.normal(0, 1, size=(10, 4))
        metrics.update(pos=pos, rot=rot)

        # Initialize and test WandB logger
        logger = WandBLogger(
            n_log=1,
            n_flush=1,
            filter=None,
            config=WandBLogger.WandBConfig(
                entity="u1999168-girona",
                project="visual_control|PBVS",
                group="Test",
                dir=Path(__file__).parent,
                config={"test_attr1": 1, "test_attr2": 0.5, "test_attr3": {"a": 1, "b": 2, "c": 3}},
            ),
        )
        logger.log(step=0.0, metrics=metrics.compute())
        assert logger._log == {}

        # Attempt to append more logs with different steps
        metrics.reset()
        pos = np.random.normal(0, 1, size=(2, 3))
        rot = np.random.normal(0, 1, size=(2, 4))
        metrics.update(pos=pos, rot=rot)
        metrics_dict = metrics.compute()
        logger.log(step=0.05, metrics={"PosError": metrics_dict["PosError"]})
        logger.log(step=0.10, metrics={"RotError": metrics_dict["RotError"]})
        assert logger._log == {}

        # Attempt to add a new metric and log it
        metric3 = AccumulatorMetric(name="VelError", argname="vel", red="mean")
        vel = np.random.normal(0, 1, size=(10, 3))
        metric3.update(vel=vel)
        logger.log(step=0.10, metrics={"VelError": metric3.compute()})
        assert logger._log == {}
