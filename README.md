# bentomlx (bentoml-extensions) [![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org) ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)


#### `todo`:  plan for 2024
[[Project]bentoml-extensions alpha release ](https://github.com/users/KimSoungRyoul/projects/2)
* FeatureStore Runner [ODM],
* optimize cpu inference [ipex, ovms]


## FeatureStore
* `pip install bentomlx[featurestore-redis]`
* `pip install bentomlx[featurestore-aerospike]`

~~~Python
import logging
from typing import Dict, TypedDict

import bentoml
import numpy as np
from bentoml.io import JSON


import bentomlx
from bentomlx.feature_repo import DBSettings


class IrisFeature(TypedDict, total=False):
    pk: str
    sepal_len: float | int
    sepal_width: float
    petal_len: float | int
    petal_width: float


# db_settings = DBSettings(namespace="test", hosts=["127.0.0.1:3000"], use_shared_connection=True)
db_settings = DBSettings()  # EXPORT ENV BENTOML_REPO_NAMESPACE=test; BENTOML_REPO__HOSTS=localhost:3000; BENTOML_REPO__USE_SHARED_CONNECTION=true

repo_runner = bentomlx.feature_repo.aerospike_fs(db_settings).to_repo_runner(entity_name="iris_features", embedded=True)

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_svc", runners=[repo_runner, iris_clf_runner])

logger = logging.getLogger("bentoml")


@svc.api(
    input=JSON.from_sample(["pk1", "pk2", "pk3"]),
    output=JSON(),
)
async def classify(feature_keys: list[str]) -> Dict[str, list[int]]:
    # features: list[list[float]] = await repository.get_many.async_run(pks=feature_keys, _nokey=True) #  [[4.9, 3.0, 1.4, 0.2], [5.1 3.5 1.4 0.3], [5.5 2.5 4.  1.3]]
    # features: list[IrisFeature] = repo_runner.get_many.run(pks=feature_keys) # input_arr = [{"pk": "pk1": "sepal_len":4.9,  "sepal_width":3.  "petal_len":1.4, "petal_width": 0.2], ... ]
    features: np.array = repo_runner.get_many.run(pks=feature_keys, _numpy=True) # input_arr = np.array([[4.9, 3.0, 1.4, 0.2], [5.1 3.5 1.4 0.3], [5.5 2.5 4.  1.3]])
    result: np.ndarray = await iris_clf_runner.predict.async_run(features)
    return {"result": result.tolist()}

~~~



## CPU Optimized Runner
  * `bentomlx[ipex]`
  * `bentomlx[ovms]` `like a bentoml[triton]`

~~~Python
import bentoml
import bentomlx


#iris_clf_runner = bentoml.pytorch.get("iris_clf:latest").to_runner()
# change like this
iris_clf_runner = bentomlx.pytorch.get("iris_clf:latest").to_runner(intel_optimize=True)
xxx_runner = bentomlx.transformers.get("xxx:latest").to_runner(intel_optimize=True)
xxx_tf_runner = bentomlx.tensorflow.get("xxx:latest").to_runner(intel_optimize=True)


# support only in bentoml-extension
# model type such as pytorch, tensorflow, onnx
xxx_ov_runner = bentomlx.openvino.get("xxx:latest").to_runner(intel_optimize=True)
# or
xxx_ov_runner = bentomlx.pytorch.get("xxx:latest").to_runner(openvino=True, post_quant=True)

# intel bert op
# https://www.intel.com/content/www/us/en/developer/articles/guide/bert-ai-inference-amx-4th-gen-xeon-scalable.html
# ?? need discussion about Out of ML serving framework responsibility
#https://github.com/intel/light-model-transformer/tree/main/BERT
xxx_ov_runner = bentomlx.experimental.light_model_transformer.bert.get("xxx:latest").to_runner(post_quant=True,quant_dtype=torch.float32)

~~~


## Post(Runtime) Model Compression (oneapi nncl)
  * post quant ?
  * ...





![스크린샷 2023-11-27 오후 3 18 18](https://github.com/KimSoungRyoul/bentoml-extensions/assets/24240623/8b922a8f-99e6-4d69-a713-a03f3f7b0d27)
