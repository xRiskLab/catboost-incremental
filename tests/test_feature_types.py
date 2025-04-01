"""test_feature_types.py."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from faker import Faker

from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


class DummyLoader:
    """Dummy data loader for testing."""

    chunk_size: int = 50_000
    cat_features: ClassVar[list] = []
    text_features: ClassVar[list] = []
    embedding_features: ClassVar[list] = []


# Data generators
def make_batch_df(kind="numerical", label="target"):
    """Make a batch DataFrame."""
    fake = Faker()
    if kind == "numerical":
        df = pd.DataFrame(np.random.rand(1000, 10), columns=[f"num_{i}" for i in range(10)])
    elif kind == "text":
        df = pd.DataFrame({"text": [fake.sentence() * 5 for _ in range(1000)]})
        df["text"] = df["text"].astype(str)
    elif kind == "embedding":
        df = pd.DataFrame({"embedding": [np.random.rand(768).tolist() for _ in range(1000)]})
        # print(df.head(5))
    else:
        raise ValueError(kind)
    df[label] = np.random.randint(0, 2, size=1000)
    return df


def make_chunks(kind="numerical", chunks=3, label="target"):
    """Create a generator that yields chunks of data."""
    fake = Faker()
    for _ in range(chunks):
        if kind == "numerical":
            df = pd.DataFrame(np.random.rand(500, 5), columns=[f"num_{i}" for i in range(5)])
        elif kind == "text":
            df = pd.DataFrame({"text": [fake.sentence() * 5 for _ in range(500)]})
            df["text"] = df["text"].astype(str)
        elif kind == "embedding":
            df = pd.DataFrame({"embedding": [np.random.rand(768).tolist() for _ in range(500)]})
        else:
            raise ValueError(kind)
        df[label] = np.random.randint(0, 2, size=500)
        y = df[label].values
        df = df.drop(columns=label)
        yield df, y


# Test matrix
test_matrix = [
    {"name": "numerical_batch", "kind": "numerical", "mode": "batch"},
    {"name": "numerical_incremental", "kind": "numerical", "mode": "incremental"},
    {"name": "text_batch", "kind": "text", "mode": "batch", "text_features": ["text"]},
    {"name": "text_incremental", "kind": "text", "mode": "incremental", "text_features": ["text"]},
    {
        "name": "embedding_batch",
        "kind": "embedding",
        "mode": "batch",
        "embedding_features": ["embedding"],
    },
    {
        "name": "embedding_incremental",
        "kind": "embedding",
        "mode": "incremental",
        "embedding_features": ["embedding"],
    },
]


@pytest.mark.parametrize("config", test_matrix)
def test_catboost_training(config):
    """Test CatBoost training with different feature types."""
    loader = DummyLoader()
    loader.text_features = config.get("text_features", [])
    loader.embedding_features = config.get("embedding_features", [])

    model_config = {
        "iterations": 10,
        "cat_features": loader.cat_features,
        "text_features": loader.text_features,
        "embedding_features": loader.embedding_features,
        "allow_writing_files": False,
        "verbose": 0,
    }

    trainer = CatBoostTrainer(data_loader=loader, label_col="target", model_config=model_config)

    train_data, test_df = get_train_test_data(config)

    trainer.train(train_data, test_df)
    acc = trainer.evaluate(test_df)
    assert 0 <= acc <= 1
    logger.success(f"Passed: {config['name']} - Accuracy: {acc:.4f}")


def get_train_test_data(config):
    """Create training and testing data based on the configuration."""
    if config["mode"] == "batch":
        df = make_batch_df(config["kind"])
        train = [(df.drop(columns="target"), df["target"])]
        test_df = df
    else:
        train = make_chunks(config["kind"])
        test_df = make_batch_df(config["kind"])
    return train, test_df
