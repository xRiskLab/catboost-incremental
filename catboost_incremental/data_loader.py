"""data_loader.py."""

from typing import ClassVar, Generator, List, Optional, Tuple

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pydantic import BaseModel, Field, field_validator, ConfigDict

from catboost_incremental.logging_config import setup_logger

logger = setup_logger()


class DataValidation(BaseModel):
    """Base class for data validation."""

    dataset_path: str = Field(
        ...,
        description="Path to the partitioned dataset (e.g., '/data' or 's3://bucket/data').",
    )
    chunk_size: int = Field(
        20_000,
        description='Chunk size for reading the dataset.',
    )
    use_cols: Optional[List[str]] = Field(
        None,
        description='List of columns to read from the dataset.',
    )
    label_col: Optional[str] = Field(
        None,
        description='Column name for the target variable.',
    )
    partition_id_col: Optional[str] = Field(
        None,
        description='Column name for partition ID.',
    )
    boto3_session: Optional[boto3.Session] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('dataset_path')
    def check_dataset_path(cls, v: str) -> str:
        """Check if the dataset path is valid."""
        if not (v.endswith('.parquet') or v.endswith('/') or v.startswith('s3://')):
            raise ValueError(
                "dataset_path must end with '.parquet', '/' (for dirs), or start with 's3://'"
            )
        return v


class DataLoader(DataValidation):
    """DataLoader class for reading and processing datasets."""

    cat_features: ClassVar[list] = []
    text_features: ClassVar[list] = []
    embedding_features: ClassVar[list] = []

    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 20_000,
        use_cols: Optional[List[str]] = None,
        partition_id_col: Optional[str] = None,
        label_col: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            chunk_size=chunk_size,
            use_cols=use_cols,
            partition_id_col=partition_id_col,
            label_col=label_col,
        )
        self.boto3_session = (
            boto3_session or boto3.Session()
        )  # use default session if none provided

    def read_parquet(self) -> Generator[Tuple, None, None]:
        """Read Parquet files from the specified dataset path."""
        if str(self.dataset_path).startswith('s3://'):
            return self._read_parquet_s3()
        return self._read_parquet_local()

    def _read_parquet_local(self) -> Generator[Tuple, None, None]:
        """Read Parquet files from a local directory."""
        logger.debug('Reading Parquet files from local directory.')
        dataset = ds.dataset(self.dataset_path, format='parquet')
        fragments = list(dataset.get_fragments())
        logger.debug(
            f'Found {len(fragments)} fragments in the dataset. Chunk size: {self.chunk_size}.'
        )
        for fragment in fragments:
            scanner = fragment.scanner(columns=self.use_cols)
            table = scanner.to_table()
            df_chunk = table.to_pandas()
            for start in range(0, len(df_chunk), self.chunk_size):
                chunk = df_chunk.iloc[start : start + self.chunk_size]
                yield self._extract_features_and_target(chunk)

    def _read_parquet_s3(self) -> Generator[Tuple, None, None]:
        """Read Parquet files from S3."""
        logger.debug('Reading Parquet files from S3 with chunk_size={self.chunk_size}.')
        for df_chunk in wr.s3.read_parquet(
            path=self.dataset_path,
            chunked=self.chunk_size,
            columns=self.use_cols,
            boto3_session=self.boto3_session,
        ):
            for start in range(0, len(df_chunk), self.chunk_size):
                chunk = df_chunk.iloc[start : start + self.chunk_size]
                yield self._extract_features_and_target(chunk)

    def _extract_features_and_target(
        self, df_chunk: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features and target from the DataFrame chunk."""
        df = df_chunk.copy()

        # Drop partition column first
        if self.partition_id_col and self.partition_id_col in df.columns:
            df = df.drop(columns=[self.partition_id_col])

        # Extract label
        if self.label_col and self.label_col in df.columns:
            y = df[self.label_col]
            X = df.drop(columns=[self.label_col])
        else:
            # Fallback: last column is label
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

        if X.empty:
            raise ValueError(
                'DataFrame is empty after dropping label and partition columns.'
            )

        # Normalize label dtype:
        if y.dtype in ['string', 'object']:
            # If string labels (classification), convert to category codes
            y = y.astype('category').cat.codes.astype('int64')
        else:
            # Otherwise ensure it's plain numpy array with numeric dtype
            y = y.to_numpy()

        return X, y
