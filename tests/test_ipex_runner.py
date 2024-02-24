from typing import Dict, TypedDict

import bentoml
from bentoml.io import JSON
import bentomlx
from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings


class IrisFeature(TypedDict, total=False):
    pk: str
    sepal_len: float | int
    sepal_width: float
    petal_len: float | int


# iris_clf_runner = bentoml.ipex.get("iris_clf:latest").to_runner()
# change like this
iris_clf_runner = bentomlx.pytorch.get("iris_clf:latest").to_runner(intel_optimize=True)
xxx_runner = bentomlx.transformers.get("xxx:latest").to_runner(intel_optimize=True)
xxx_tf_runner = bentomlx.tensorflow.get("xxx:latest").to_runner(intel_optimize=True)

# support only in bentoml-extension
# model type such as ipex, tensorflow, onnx
xxx_ov_runner = bentomlx.openvino.get("xxx:latest").to_runner(intel_optimize=True)


svc = bentoml.Service("iris_classifier_svc", runners=[iris_clf_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(feature_keys: list[str]) -> Dict[str, list[float]]:
    # features: list[IrisFeature] = await repository.async_filter(ids=feature_keys).get_many()
    # features: list[list[float]] = await repository.async_filter(ids=feature_keys,_nokey=True).get_many()
    feature = iris_clf_runner.run()
    print(feature)
    return {"result": []}
