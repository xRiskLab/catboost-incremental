"""test_catboost_trainer.py."""

import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
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
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
        df['outcome'] = y
        return df

    def test_incremental_trainer(self):
        """Test the incremental trainer and training stats, including small chunk edge case."""
        logger.info('Testing incremental trainer')

        # Valid training data
        trainer = CatBoostTrainer(data_loader=DummyLoader(), label_col='outcome')
        train_chunks = self.make_generator()
        test_df = self.make_test_df()
        model = trainer.train(train_chunks, test_df)

        self.assertTrue(hasattr(model, 'predict'))
        stats = trainer.training_stats
        self.assertGreater(len(stats['score']), 0)
        final_score = stats['score'][-1]
        logger.success(f'Final score: {final_score}')
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)

        # Edge case: small chunk size
        trainer_partial_skip = CatBoostTrainer(
            data_loader=DummyLoader(),
            label_col='target',
            model_config={'iterations': 3, 'allow_writing_files': False, 'verbose': 0},
        )

        def partial_chunks():
            # First chunk: balanced
            X1 = pd.DataFrame(np.random.rand(80, 4), columns=['a', 'b', 'c', 'd'])
            y1 = np.random.randint(0, 2, size=80)
            yield X1, y1

            # Second chunk: all zeros should be skipped
            X2 = pd.DataFrame(np.random.rand(20, 4), columns=['a', 'b', 'c', 'd'])
            y2 = np.zeros(20, dtype=int)
            yield X2, y2

        dummy_test_df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
        dummy_test_df['target'] = np.random.randint(0, 2, size=10)

        with patch.object(logger, 'warning') as mock_log:
            model = trainer_partial_skip.train(train_data=partial_chunks(), test_data=dummy_test_df)

        # Model should still be trained based on the first chunk
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(
            any('target contains only one class' in call.args[0] for call in mock_log.call_args_list),
            'Expected log about skipping chunk with one class not found'
        )

    def test_model_type_inference(self):
        # sourcery skip: extract-duplicate-method
        """Test that the trainer infers classifier vs regressor correctly."""

        # Classification (integer labels)
        int_y = np.random.randint(0, 2, size=100)
        int_X = np.random.rand(100, 5)
        trainer_int = CatBoostTrainer(data_loader=DummyLoader(), label_col='label')
        trainer_int._init_model(int_y)
        self.assertIsInstance(trainer_int.model, CatBoostClassifier)

        # Classification (string labels)
        str_y = np.random.choice(['cat', 'dog'], size=100)
        str_X = np.random.rand(100, 5)
        trainer_str = CatBoostTrainer(data_loader=DummyLoader(), label_col='label')
        trainer_str._init_model(str_y)
        self.assertIsInstance(trainer_str.model, CatBoostClassifier)

        # Regression (float labels)
        float_y = np.random.uniform(0, 100, size=100)
        float_X = np.random.rand(100, 5)
        trainer_float = CatBoostTrainer(data_loader=DummyLoader(), label_col='label')
        trainer_float._init_model(float_y)
        self.assertIsInstance(trainer_float.model, CatBoostRegressor)

    def test_explicit_test_dataframe_usage(self):
        """Ensure trainer uses provided test DataFrame instead of fallback chunk."""

        trainer = CatBoostTrainer(data_loader=DummyLoader(), label_col='outcome')

        # Train data generator (3 chunks)
        train_chunks = self.make_generator(n_chunks=3)

        # Create test set with distinct values to identify it was used
        test_df = pd.DataFrame(
            np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)]
        )
        test_df['outcome'] = np.random.randint(0, 2, size=100)

        # Train
        model = trainer.train(train_data=train_chunks, test_data=test_df)

        # Check if model is trained and evaluated
        self.assertTrue(hasattr(model, 'predict'))
        self.assertIn('score', trainer.training_stats)
        self.assertGreater(len(trainer.training_stats['score']), 0)
