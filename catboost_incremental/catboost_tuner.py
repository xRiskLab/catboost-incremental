"""catboost_tuner.py."""

from __future__ import annotations

from typing import Optional

import pyarrow.dataset as ds
from ray import init, is_initialized, tune
from ray.tune.schedulers import ASHAScheduler

from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


def catboost_trainable(config, trainer_factory, label_col, metric):
    """Ray-compatible function that trains and evaluates a CatBoost model."""
    trainer: CatBoostTrainer = trainer_factory(config)
    train_data = trainer.data_loader.read_parquet()

    # Load test data from the full dataset
    dataset = ds.dataset(trainer.data_loader.dataset_path)
    test_data = dataset.to_table().to_pandas()

    model = trainer.train(train_data=train_data, test_data=test_data)

    X_test = test_data.drop(columns=label_col)
    y_true = test_data[label_col]
    y_pred = model.predict(X_test)

    if metric == 'accuracy' and y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)

    score = (y_pred == y_true).mean() if metric == 'accuracy' else None
    logger.info(f'Tuning score: {score}')
    tune.report({metric: score})


class CatBoostTuner:
    """
    CatBoost hyperparameter tuning using Ray Tune.
    """

    def __init__(
        self,
        trainer: CatBoostTrainer,
        metric: str = 'accuracy',
    ):
        self.trainer = trainer
        self.metric = metric
        self.label_col = trainer.label_col

    def tune(self, param_space: dict, num_samples: int = 5, mode: Optional[str] = None):
        """Perform hyperparameter tuning using Ray Tune."""
        if not is_initialized():
            init(ignore_reinit_error=True, include_dashboard=False)
            logger.info('Ray initialized.')

        resolved_mode = mode or ('min' if 'loss' in self.metric.lower() else 'max')

        # Inject config into new trainer instances
        def trainer_factory(config):
            return CatBoostTrainer(
                data_loader=self.trainer.data_loader,
                label_col=self.label_col,
                model_config=config,
            )

        tuner = tune.Tuner(
            tune.with_parameters(
                catboost_trainable,
                trainer_factory=trainer_factory,
                label_col=self.label_col,
                metric=self.metric,
            ),
            tune_config=tune.TuneConfig(
                metric=self.metric,
                mode=resolved_mode,
                scheduler=ASHAScheduler(max_t=50, grace_period=1),
                num_samples=num_samples,
            ),
            param_space=param_space,
        )

        return tuner.fit().get_best_result()
