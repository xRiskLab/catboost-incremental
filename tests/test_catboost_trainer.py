"""test_catboost_trainer.py."""

import unittest
from typing import ClassVar

import numpy as np
import pandas as pd

from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


class DummyLoader:
    """Dummy data loader for testing."""

    chunk_size: int = 50_000
    cat_features: ClassVar[list] = []
    text_features: ClassVar[list] = []
    embedding_features: ClassVar[list] = []


class TestTrainer(unittest.TestCase):
    """Unit tests for CatBoostTrainer class."""

    def make_generator(self, n_chunks=3, n_samples=5000, n_features=5):
        """Make a generator that yields (X, y) tuples."""
        for _ in range(n_chunks):
            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, 2, size=n_samples)
            yield X, y

    def make_test_df(self, n_samples=1000, n_features=5):
        """Make a DataFrame for testing."""
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
        df["outcome"] = y
        return df

    def test_incremental_trainer(self):
        """Test the incremental trainer and training stats."""
        logger.info("Testing incremental trainer")
        trainer = CatBoostTrainer(data_loader=DummyLoader(), label_col="outcome")

        train_chunks = self.make_generator()
        test_df = self.make_test_df()

        model = trainer.train(train_chunks, test_df)

        # Validate model
        self.assertTrue(hasattr(model, "predict"))

        # Validate training stats
        stats = trainer.training_stats
        self.assertGreater(len(stats["score"]), 0)
        final_score = stats["score"][-1]

        logger.success(f"Final score: {final_score}")
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)
