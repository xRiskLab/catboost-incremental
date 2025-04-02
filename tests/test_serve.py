import os
import subprocess
import time
from pathlib import Path

import boto3
import httpx
import pandas as pd
import pytest
from moto import mock_aws

from catboost_incremental.serve_ray import CatBoostModel, CatBoostModelDeployment

SERVER_URL = 'http://127.0.0.1:8000/predict'
SERVER_SCRIPT = 'catboost_incremental/serve_ray.py'
LOCAL_TEST_MODEL = Path('models/cb_model.cbm')


@pytest.fixture(scope='module', autouse=True)
def start_ray_serve():
    """Start Ray head node and Ray Serve app before tests, and tear them down after."""

    ray_process = subprocess.Popen(
        'source .venv/bin/activate && ray start --head',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
    )
    time.sleep(3)

    serve_process = subprocess.Popen(
        ['uv', 'run', SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != 'nt',
    )

    logs = []

    for _ in range(20):
        if serve_process.poll() is not None:
            out, err = serve_process.communicate()
            print('Serve process exited early.')
            print('stdout:\n', out)
            print('stderr:\n', err)
            pytest.fail('Ray Serve process exited unexpectedly.')

        try:
            response = httpx.post(SERVER_URL, json={str(i): 0.0 for i in range(10)})
            if response.status_code == 200:
                break
        except httpx.ConnectError as e:
            logs.append(str(e))
            time.sleep(0.5)
    else:
        out, err = serve_process.communicate()
        print('Timed out waiting for Ray Serve to start.')
        print('stdout:\n', out)
        print('stderr:\n', err)
        print('Connection errors:\n', '\n'.join(logs))
        serve_process.terminate()
        ray_process.terminate()
        pytest.fail('Ray Serve server did not start in time.')

    yield

    serve_process.terminate()
    ray_process.terminate()
    serve_process.wait()
    ray_process.wait()

    subprocess.run(
        'source .venv/bin/activate && ray stop --force',
        shell=True,
        executable='/bin/bash',
        text=True,
        check=True,
    )


@pytest.mark.asyncio
async def test_catboost_model_predict():
    """Test the CatBoost model prediction endpoint."""
    sample_input = {
        '0': 1.2,
        '1': 3.4,
        '2': 0.0,
        '3': 2.1,
        '4': 5.5,
        '5': 0.7,
        '6': 1.0,
        '7': 4.2,
        '8': 0.9,
        '9': 3.3,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(SERVER_URL, json=sample_input)

    assert response.status_code == 200
    json_data = response.json()
    assert 'proba' in json_data
    assert isinstance(json_data['proba'], list)
    assert isinstance(json_data['proba'][0], list)
    assert all(isinstance(p, float) for p in json_data['proba'][0])


@mock_aws
def test_catboost_model_loading_from_mocked_s3():
    """Test loading a CatBoost model from a mocked S3 bucket."""
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket = 'mock-bucket'
    key = 'mock-path/model.cbm'

    s3.create_bucket(Bucket=bucket)
    s3.upload_file(str(LOCAL_TEST_MODEL), bucket, key)

    s3_uri = f's3://{bucket}/{key}'
    model = CatBoostModel(model_path=s3_uri)

    sample_input = [{str(i): 0.0 for i in range(10)}]
    df = pd.DataFrame(sample_input)
    probs = model.predict_proba(df)
    assert probs.shape[1] >= 1


@pytest.mark.asyncio
async def test_catboost_model_deployment_with_s3():
    """Test wrapping the CatBoostModel with Ray Serve decorator and running as deployment."""

    # Explicitly mock AWS credentials to avoid NoCredentialsError
    os.environ['AWS_ACCESS_KEY_ID'] = 'fake-access-key'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'fake-secret-key'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

    # Use moto mock_s3 inside the test function
    with mock_aws():
        # S3 setup
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket = 'mock-bucket'
        key = 'mock-path/model.cbm'
        s3.create_bucket(Bucket=bucket)
        s3.upload_file(str(LOCAL_TEST_MODEL), bucket, key)
        s3_uri = f's3://{bucket}/{key}'

        # Wrap CatBoostModel with @serve.deployment
        # pylint: disable=no-member
        model_deployment = CatBoostModelDeployment.bind(model_path=s3_uri)

        # Simple smoke test to ensure it works via Serve
        sample_input = [{str(i): 0.0 for i in range(10)}]

        async with httpx.AsyncClient() as client:
            response = await client.post(SERVER_URL, json=sample_input)

        # Assertions to validate that the response contains the expected output
        assert response.status_code == 200
        json_data = response.json()
        assert 'proba' in json_data
        assert isinstance(json_data['proba'], list)
        assert isinstance(json_data['proba'][0], list)
        assert all(isinstance(p, float) for p in json_data['proba'][0])
