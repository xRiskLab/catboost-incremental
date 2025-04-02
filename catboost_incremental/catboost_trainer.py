"""catboost_trainer.py."""

from __future__ import annotations

import time
from typing import Any, Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error
from sklearn.model_selection import train_test_split

from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


class CatBoostTrainer:
    """CatBoostTrainer for incremental training of CatBoost models."""

    def __init__(
        self,
        data_loader,
        label_col: str,
        model_config: Optional[dict] = None,
        metric_fn: Optional[callable] = None,
    ) -> CatBoostTrainer:
        self.data_loader = data_loader
        self.label_col = label_col
        self.chunk_size = getattr(data_loader, 'chunk_size', 0)
        self.model_config = model_config or self.default_params()
        self.model = None  # defer model initialization
        self.metric_fn = metric_fn

        # For evaluation and tracking
        self.training_stats: dict[str, list[Union[float, int]]] = {
            'chunk_index': [],
            'duration': [],
            'score': [],
        }

    @staticmethod
    def default_params() -> dict:
        """Default parameters for CatBoost."""
        return dict(
            task_type='CPU',
            iterations=500,
            learning_rate=0.1,
            max_depth=3,
            verbose=0,
            allow_writing_files=False,
        )

    def _is_regression(self, y: np.ndarray) -> bool:
        """Check if the target variable is regression or classification."""
        logger.debug('Checking if target variable is regression or classification...')
        y = np.array(y)

        # Handle string labels for classification (categorical)
        if y.dtype.kind == 'O':  # 'O' indicates object (string) type
            is_regression = False  # It's a classification task if the label is a string
        else:
            # If the labels are numeric, we determine regression based on unique values
            is_regression = y.dtype.kind == 'f' and len(np.unique(y)) > 10

        logger.debug(f"Target is {'regression' if is_regression else 'classification'}")
        return is_regression

    def _init_model(self, y: np.ndarray):
        """Initialize the CatBoost model based on the target variable type."""
        if self.model is not None:
            return
        if self._is_regression(y):
            self.model = CatBoostRegressor(**self.model_config)
        else:
            self.model = CatBoostClassifier(**self.model_config)

    def train(
        self,
        train_data: Optional[Generator[Tuple[np.ndarray, np.ndarray], None, None]] = None,
        test_data: Optional[pd.DataFrame] = None,
        update_every_n_chunks: int = 5,
    ) -> Any:
        """
        Train the CatBoost model using data chunks or provided data.

        If train_data and test_data are None, it uses the data_loader to read and evaluate the data.
        """
        start_time = time.time()

        if train_data is None:
            if not self.data_loader:
                raise ValueError('Either train_data or a valid data_loader must be provided.')
            train_data = self.data_loader.read_parquet()

        # Convert test_data to a pandas DataFrame if it's a generator or None
        if test_data is None:
            if not self.data_loader:
                raise ValueError('Either train_data or a valid data_loader must be provided.')
            test_data = self.data_loader.read_parquet()

        try:
            first_chunk = next(train_data)
        except StopIteration as e:
            raise ValueError('Training data generator is empty.') from e

        self._init_model(first_chunk[1])

        last_chunk = first_chunk

        try:
            second_chunk = next(train_data)
            generator = self._prepend_chunks(first_chunk, second_chunk, train_data)
            self._train_incremental(generator, update_every_n_chunks)

            last_chunk = second_chunk
        except StopIteration:
            self._train_batch(*first_chunk)

        if isinstance(test_data, pd.DataFrame):
            test_y = test_data[self.label_col].to_numpy()
            test_X = test_data.drop(columns=[self.label_col])
        else:
            test_X, test_y = last_chunk

        test_pool = self._make_pool(test_X, test_y)
        duration = time.time() - start_time
        score = self.evaluate(test_pool)

        logger.debug(f'Training completed in {duration:.2f} seconds.')
        logger.debug(f'Final score: {score:.4f}')
        logger.debug(f'Training stats: {self.training_stats}')

        return self.model

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on a single batch of data."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

        train_pool = self._make_pool(X_train, y_train)
        val_pool = self._make_pool(X_val, y_val)
        start = time.time()
        self.model.fit(train_pool, eval_set=val_pool)
        duration = time.time() - start

        score = self._evaluate_pool(val_pool)
        self.training_stats['chunk_index'].append(0)
        self.training_stats['duration'].append(duration)
        self.training_stats['score'].append(score)

    # pylint: disable=too-many-locals
    def _train_incremental(
        self,
        generator: Generator[Tuple[np.ndarray, np.ndarray], None, None],
        update_every_n_chunks: int,
    ) -> None:
        """Train the model incrementally using data chunks."""
        init_model = None
        val_pool = None

        for i, (X_chunk, y_chunk) in enumerate(generator):
            start = time.time()

            if len(np.unique(y_chunk)) < 2:
                logger.warning(
                    f"Skipping chunk {i}: target contains only one class '{np.unique(y_chunk)[0]}'. "
                    f'Consider increasing the chunk size (currently {len(y_chunk)} rows).'
                )
                continue

            if i == 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_chunk, y_chunk, train_size=0.8, random_state=42
                )
                train_pool = self._make_pool(X_train, y_train)
                val_pool = self._make_pool(X_val, y_val)
                self.model.fit(train_pool, eval_set=val_pool)
                init_model = self.model
            elif i % update_every_n_chunks == 0:
                train_pool = self._make_pool(X_chunk, y_chunk)
                self.model.fit(train_pool, eval_set=val_pool, init_model=init_model)
                init_model = self.model

            duration = time.time() - start
            score = self._evaluate_pool(val_pool)

            self.training_stats['chunk_index'].append(i)
            self.training_stats['duration'].append(duration)
            self.training_stats['score'].append(score)

    def _evaluate_pool(self, pool: Pool) -> float:
        """Evaluate the model on the given pool."""
        logger.debug('Evaluating the pool...')
        y_true = pool.get_label()

        if isinstance(self.model, CatBoostClassifier):
            preds = self.model.predict(pool)
            preds_proba = self.model.predict_proba(pool)
            if self.metric_fn:
                return self.metric_fn(y_true, preds_proba)
            return accuracy_score(y_true, preds)
        else:
            preds = self.model.predict(pool)
            if self.metric_fn:
                return self.metric_fn(y_true, preds)
            return root_mean_squared_error(y_true, preds)

    def evaluate(self, data: Union[Pool, pd.DataFrame]) -> float:
        """Evaluate the model on a Pool or a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            y_true = data[self.label_col].to_numpy()
            X = data.drop(columns=[self.label_col])
            pool = self._make_pool(X, y_true)
        else:
            pool = data
            y_true = pool.get_label()

        # Now safe to proceed
        if isinstance(self.model, CatBoostRegressor):
            preds = self.model.predict(pool)
            return (
                self.metric_fn(y_true, preds)
                if self.metric_fn
                else root_mean_squared_error(y_true, preds)
            )

        if self.metric_fn is log_loss:
            preds_proba = self.model.predict_proba(pool)
            return log_loss(y_true, preds_proba)

        preds = self.model.predict(pool)
        return self.metric_fn(y_true, preds) if self.metric_fn else accuracy_score(y_true, preds)

    def serialize(self, path: str) -> None:
        """Serialize the trained model to a file."""
        self.model.save_model(path, format='cbm')

    def _make_pool(self, X: np.ndarray, y: np.ndarray) -> Pool:
        """Make a CatBoost Pool from the data."""
        logger.debug('Creating the pool...')
        return Pool(
            data=X,
            label=y,
            cat_features=self.model_config.get('cat_features', []),
            text_features=self.model_config.get('text_features', []),
            embedding_features=self.model_config.get('embedding_features', []),
        )

    def _prepend_chunks(self, first_chunk, second_chunk, generator):
        """Prepend the first two chunks to the generator."""

        def combined():
            yield first_chunk
            yield second_chunk
            yield from generator

        return combined()
