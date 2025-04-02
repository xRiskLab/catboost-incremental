import tempfile
from urllib.parse import urlparse

import boto3
import pandas as pd
from catboost import CatBoostClassifier
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse


class CatBoostModel:
    """Reusable logic for loading and using a CatBoost model."""

    def __init__(self, model_path='models/cb_model.cbm'):
        self.model = CatBoostClassifier()
        self.model.load_model(self._resolve_model_path(model_path))

    def _resolve_model_path(self, model_path: str) -> str:
        if model_path.startswith('s3://'):
            return self._download_from_s3(model_path)
        return model_path

    def _download_from_s3(self, s3_uri: str) -> str:
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        s3 = boto3.client('s3')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cbm')
        s3.download_file(bucket, key, tmp_file.name)
        return tmp_file.name

    def predict_proba(self, input_df: pd.DataFrame):
        return self.model.predict_proba(input_df)


@serve.deployment
class CatBoostModelDeployment:
    """Ray Serve wrapper around CatBoost model."""

    def __init__(self, model_path='models/cb_model.cbm'):
        self.wrapper = CatBoostModel(model_path)

    async def __call__(self, request: Request) -> JSONResponse:
        if request.method != 'POST':
            return JSONResponse({'error': 'Only POST allowed'}, status_code=405)

        data = await request.json()
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        probs = self.wrapper.predict_proba(df).tolist()
        return JSONResponse({'proba': probs})


if __name__ == '__main__':
    import ray

    ray.init(address='auto', namespace='serve')
    serve.start(detached=False, http_options={'host': '0.0.0.0'})

    # pylint: disable=no-member
    app = CatBoostModelDeployment.bind()
    serve.run(app, route_prefix='/predict')
