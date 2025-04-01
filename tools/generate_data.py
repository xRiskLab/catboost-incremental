import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

app = typer.Typer()


def generate_synthetic_parquet(
    output_path: str,
    n_samples: int = 100_000,
    n_features: int = 10,
    chunk_size: int = 20_000,
    num_partitions: int = 10,
    correlation_strength: float = 0.8,
):
    """
    Generates synthetic classification data with correlated features and saves it
    to a partitioned Parquet dataset using PyArrow.

    Args:
        output_path (str): Path to save the Parquet dataset.
        n_samples (int): Number of samples to generate.
        n_features (int): Number of feature columns.
        chunk_size (int): Number of rows per chunk.
        num_partitions (int): Number of partitions.
        correlation_strength (float): Strength of correlation between features and target (0 to 1).
    """
    print(
        f"Generating synthetic data with {n_samples} samples, "
        f"{n_features} features, {num_partitions} partitions, "
        f"correlation_strength={correlation_strength}..."
    )

    feature_columns = [f"feat_{i}" for i in range(n_features)]
    all_columns = feature_columns + ["target", "partition_id"]  # noqa: RUF005

    os.makedirs(output_path, exist_ok=True)  # noqa: PTH103

    for i in range(0, n_samples, chunk_size):
        size = min(chunk_size, n_samples - i)

        # Binary target
        y = np.random.binomial(1, 0.5, size=size)

        # Create a base signal that correlates with the target
        signal = y * correlation_strength + np.random.normal(
            scale=1 - correlation_strength, size=size
        )

        # Generate correlated features from the signal
        X = np.zeros((size, n_features))
        for j in range(n_features):
            noise = np.random.normal(scale=0.3, size=size)
            X[:, j] = signal + noise

        # Add partition IDs
        partition_id = np.random.randint(0, num_partitions, size)

        df = pd.DataFrame(X, columns=feature_columns)
        df["target"] = y
        df["partition_id"] = partition_id

        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=["partition_id"],
            existing_data_behavior="overwrite_or_ignore",
        )

        print(f"Saved {i + size}/{n_samples} samples...")

    print(f"Parquet dataset saved at: {output_path}")


@app.command()
def main(
    output_path: str = "data",
    n_samples: int = 200_000,
    n_features: int = 10,
    chunk_size: int = 20_000,
    num_partitions: int = 10,
    correlation_strength: float = 0.8,
):
    """
    CLI command to generate partitioned synthetic data with correlations and save as Parquet.
    """
    generate_synthetic_parquet(
        output_path,
        n_samples,
        n_features,
        chunk_size,
        num_partitions,
        correlation_strength,
    )


if __name__ == "__main__":
    app()
