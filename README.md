# catboost-incremental

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![CatBoost](https://img.shields.io/badge/catboost-1.2.7-orange.svg)](https://catboost.ai/)
[![Ray](https://img.shields.io/badge/ray-2.44.1-g.svg)](https://docs.ray.io/en/latest/)

Author: https://github.com/deburky

## 📚 Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Dataset preparation](#dataset-preparation)
- [Incremental learning](#incremental-learning)
- [Install](#install)
- [Training](#training)
- [Tuning](#tuning)
- [Serving](#serving)
  - [Jupyter notebooks](#jupyter-notebooks)
  - [Command line](#command-line)
- [Docker](#docker)
- [License](#license)

## 🪪 Introduction

This repo contains an ML training and serving system for **CatBoost + Ray** incremental learning. CatBoost and Ray are a powerful combination for training and serving machine learning models. 

CatBoost is a gradient boosting library that is particularly effective for categorical / text / embedding features, while Ray is a distributed computing framework that allows you to scale your machine learning workloads.

The package serves to address the following use cases:

* Training on datasets that do not not fit into memory (e.g., S3 parquet).
* Tuning hyperparameters with Ray Tune on large datasets.
* Serving incrementally trained CatBoost models with Ray Serve.

## ⚙️ Prerequisites

- Python 3.9+
- Ray >= 2.0
- uv
- Data in Parquet format (partitioned)
- Docker (optional)

## 🗂 Dataset preparation

Your dataset should be partitioned in Parquet files, with a partition column (e.g., `partition_id`).

Note if `label_col` is not provided to `DataLoader`, the last column is used as label.

You can generate the dataset using the following code:

```python
uv run tools/generate_data.py
```

## 🔄 Incremental learning

CatBoost supports incremental learning mode, allowing to train models on a large data chunks. This is useful when the dataset is too large to fit into memory all at once.

In CatBoost, incremental training is supported on CPU starting from version `0.15`. [GPU issue](https://github.com/catboost/catboost/issues/860) is still open.

- [**Stackoverflow**: Catboost training model for huge data(~22GB) with multiple chunks](https://stackoverflow.com/questions/47019215/catboost-training-model-for-huge-data22gb-with-multiple-chunks)
- [**catboost**: Training Catboost models with multiple chunks of data](https://github.com/catboost/catboost/issues/152)

The implementation of Ray Tuner is adopted from ["Example of using Ray Tune and CatBoost"](https://www.richarddecal.com/data-science/example-of-using-ray-tune-and-cat-boost/) by Richard Decal ([GitHub code](https://github.com/crypdick/ray-catboost/blob/main/train.py)).

The serving API is the author's implementation and is not based on any existing code.

## 💻 Installation

Install from GitHub with pip:

```bash
pip install git+https://github.com/xRiskLab/catboost-incremental.git
```

Or from the local directory with `uv`:

```bash
uv pip install -e .
```

Examples of package usage are provided in the `notebooks/` directory.

## 💻 Training

For training you can use:

```python
import pyarrow.dataset as ds
from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.data_loader import DataLoader

# Load full dataset
dataset_path = "../data/"
dataset = ds.dataset(dataset_path)
full_df = dataset.to_table().to_pandas().sample(1000)
label = "target"

data_loader = DataLoader(
    dataset_path, chunk_size=1000, partition_id_col="partition_id", label_col=label
)
trainer = CatBoostTrainer(
    data_loader=data_loader,
    label_col=label,
    model_config={"allow_writing_files": False},
)

model = trainer.train()
score = trainer.evaluate(full_df)
print(f"Accuracy: {score:.4f}")
```
## 🪛 Tuning

```python
from pathlib import Path
import pyarrow.dataset as ds
from ray import tune
from catboost_incremental.catboost_trainer import CatBoostTrainer
from catboost_incremental.catboost_tuner import CatBoostTuner
from catboost_incremental.data_loader import DataLoader

dataset_path = str(Path("../data/").resolve()) + "/"

dataset = ds.dataset(dataset_path)
full_df = dataset.to_table().to_pandas()

label = "target"

# Setup DataLoader
data_loader = DataLoader(
    dataset_path,
    chunk_size=10_000,
    partition_id_col="partition_id",
    label_col=label,
)

# Create data_loader and trainer
data_loader = DataLoader(
    dataset_path, chunk_size=10_000, partition_id_col="partition_id", label_col=label
)
trainer = CatBoostTrainer(data_loader=data_loader, label_col=label)

# Generate train_data generator and test set
train_data = trainer.data_loader.read_parquet()
test_df = ds.dataset(dataset_path).to_table().to_pandas()

# Initialize tuner
tuner = CatBoostTuner(trainer=trainer, metric="accuracy")

result = tuner.tune(
    param_space={
        "iterations": tune.choice([50, 100]),
        "learning_rate": tune.uniform(0.01, 0.3),
        "depth": tune.choice([4, 6, 8]),
        "verbose": 0,
        "allow_writing_files": False,
    },
    num_samples=5,
)

print(f"Best config: {result.config}")
```

## 🧬 Serving

## Jupyter notebooks

If you're running inside a Jupyter notebook, you can serve the model using:

```python
# Start Ray and Serve
ray.init(ignore_reinit_error=True)

# Deploy and run the app
app = CatBoostModelDeployment.bind(model_path="models/cb_model.cbm")
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
serve.run(app, route_prefix="/predict")
```

## Command line

You can start Ray cluster from the command line:

```bash
source .venv/bin/activate
ray start --head
```

You should see in the terminal:

```plain
Local node IP: 127.0.0.1

--------------------
Ray runtime started.
--------------------
```

Start the Ray Serve API server by running:

```bash
uv run catboost_incremental/serve_ray.py
```

You should see:

```plain
2025-04-01 01:06:29,822 INFO worker.py:1660 -- Connecting to existing Ray cluster at address: 127.0.0.1:6379...
2025-04-01 01:06:29,826 INFO worker.py:1843 -- Connected to Ray cluster. View the dashboard at 127.0.0.1:8265 
```

Test the endpoint with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"0": 1.2, "1": 3.4, "2": 0.0, "3": 2.1, "4": 5.5, "5": 0.7, "6": 1.0, "7": 4.2, "8": 0.9, "9": 3.3}'
```

Stop the server with:

```bash
ray stop
```

You should see:

```json
{"proba":[0.8514547423772816,0.14854525762271836]}
```

### Docker

```bash
docker build -t ray-catboost-serve .
```

And run the container:

```bash
docker run -d -p 8000:8000 -p 8265:8265 ray-catboost-serve
```

Send a request to the server:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"0": 1.2, "1": 3.4, "2": 0.0, "3": 2.1, "4": 5.5, "5": 0.7, "6": 1.0, "7": 4.2, "8": 0.9, "9": 3.3}'
```

You should see:

```json
{"proba":[0.8514547423772816,0.14854525762271836]}
```

To stop Docker (and Ray) run:
```
docker stop <container_id>
```

For example, my container ID is 'a5ae3ebc6f' so I run:
```
docker stop a5ae3ebc6f
```

**Note:** Docker containers exit by default once the entrypoint script finishes execution.  
To keep the container alive, the entrypoint script (`entrypoint.sh`) ends with:
`exec tail -f /dev/null`
(See [this Stack Overflow thread](https://stackoverflow.com/questions/28212380/why-docker-container-exits-immediately)).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.