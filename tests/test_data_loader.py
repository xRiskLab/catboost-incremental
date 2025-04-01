"""test_data_loader.py."""

import unittest
from unittest.mock import MagicMock, patch

import boto3
import pandas as pd
import pyarrow as pa
from moto import mock_aws

from catboost_incremental.data_loader import DataLoader
from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


class TestDataLoader(unittest.TestCase):
    """Test class for DataLoader."""

    def test_initialization_and_validation(self):
        """Test initialization and validation of DataLoader."""
        loader = DataLoader(dataset_path="s3://bucket/data")
        self.assertEqual(loader.dataset_path, "s3://bucket/data")

        with self.assertRaises(ValueError):
            DataLoader(dataset_path="invalid_path")

    @mock_aws
    @patch("awswrangler.s3.read_parquet")
    def test_read_parquet_s3_partitioned(self, mock_read_parquet):
        """Test reading a partitioned Parquet dataset from mocked S3 via awswrangler."""
        bucket = "test-bucket"
        region = "us-east-1"
        base_path = f"s3://{bucket}/partitioned-data/"

        df = pd.DataFrame({"feature1": [1], "feature2": [2], "target": [0]})

        # Return a generator that yields our test dataframe
        mock_read_parquet.return_value = iter([df])

        boto3.client("s3", region_name=region).create_bucket(Bucket=bucket)

        loader = DataLoader(dataset_path=base_path)
        generator = loader.read_parquet()

        X, y = next(generator)
        self.assertEqual(X.shape, (1, 2))
        self.assertEqual(y.tolist(), [0])

        # Ensure read_parquet was called with the correct parameters
        mock_read_parquet.assert_called_with(path=base_path, chunked=20_000, columns=None)

    @patch("pyarrow.dataset.dataset")
    def test_read_parquet_local(self, mock_ds_dataset):
        """Test reading local Parquet files."""
        df = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1], "partition_id": ["a", "b"]}
        )
        table = pa.Table.from_pandas(df)

        mock_fragment = MagicMock()
        mock_fragment.scanner.return_value.to_table.return_value = table

        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = [mock_fragment]
        mock_ds_dataset.return_value = mock_dataset

        loader = DataLoader(
            dataset_path="mock_data.parquet",
            partition_id_col="partition_id",
        )
        generator = loader.read_parquet()
        X, y = next(generator)

        print(X.head())

        assert X.shape == (2, 2)
        assert y.tolist() == [0, 1]


class TestChunkingBehavior(unittest.TestCase):
    """Test class for chunking behavior."""

    def test_chunking_behavior(self):
        """Test chunking behavior of DataLoader."""
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "feature2": list(range(100)),
                "target": [0] * 50 + [1] * 50,
            }
        )
        table = pa.Table.from_pandas(df)

        mock_fragment = MagicMock()
        mock_fragment.scanner.return_value.to_table.return_value = table

        with patch("pyarrow.dataset.dataset") as mock_ds:
            self._read_chunks(mock_fragment, mock_ds)

    def _read_chunks(self, mock_fragment, mock_ds):
        """Read chunks from the dataset."""
        mock_ds.return_value.get_fragments.return_value = [mock_fragment]
        loader = DataLoader(dataset_path="mock_data.parquet", chunk_size=25, label_col="target")
        generator = loader.read_parquet()
        chunks = list(generator)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0][0].shape[0], 25)
        self.assertEqual(chunks[3][0].shape[0], 25)


if __name__ == "__main__":
    unittest.main()
