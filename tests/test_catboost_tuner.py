"""test_catboost_tuner.py."""

import tempfile
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.catboost_tuner import CatBoostTuner
from catboost_incremental.logging_config import setup_logger

ray = pytest.importorskip("ray")
tune = pytest.importorskip("ray.tune")

logger = setup_logger()


def make_dummy_dataframe(n_samples=1000, n_features=5):
    """Make a dummy DataFrame for testing."""
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["outcome"] = y
    return df


class DummyLoader:
    """Dummy data loader for testing purposes."""

    dataset_path: str
    chunk_size: int = 50_000
    cat_features: ClassVar[list] = []
    text_features: ClassVar[list] = []
    embedding_features: ClassVar[list] = []

    def __init__(self, dataset_path):
        """Initialize the DummyLoader with a dataset path."""
        self.dataset_path = dataset_path

    def read_parquet(self):
        """Read the Parquet file and return a DataFrame."""
        df = make_dummy_dataframe()
        return ((df.drop(columns=["outcome"]).values, df["outcome"].values) for _ in range(3))


def test_ray_tuner_runs():
    """Run the tuning process and check if it completes successfully."""
    # sourcery skip: no-conditionals-in-tests
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Write real Parquet file to a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        df = make_dummy_dataframe()
        table = pa.Table.from_pandas(df)
        file_path = f"{tmpdir}/dummy.parquet"
        pq.write_table(table, file_path)

        # Use actual path so subprocess can read it
        loader = DummyLoader(dataset_path=file_path)
        trainer = CatBoostTrainer(data_loader=loader, label_col="outcome", model_config={})
        tuner = CatBoostTuner(trainer=trainer, metric="accuracy")

        param_space = {
            "iterations": tune.choice([10, 20]),
            "depth": tune.randint(2, 4),
            "learning_rate": tune.loguniform(0.01, 0.1),
        }

        result = tuner.tune(param_space=param_space, num_samples=2)

        logger.info(f"Best config: {result.config}")
        logger.info(f"Best score: {result.metrics['accuracy']}")
        assert isinstance(result.config, dict)
