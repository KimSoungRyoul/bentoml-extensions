# bentoml-extensions [![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org) ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)


#### `todo`:  plan for 2024
[[Project]bentoml-extensions alpha release ](https://github.com/users/KimSoungRyoul/projects/2)
* FeatureStore Runner [ODM], 
* optimize cpu inference [ipex, ovms]


## FeatureStore 
* `pip install bentoml-extensions[featurestore-redis]`
* `pip install bentoml-extensions[featurestore-aerospike]`

~~~Python
import bentoml
from bentoml.io import JSON
from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings

import bentoml_extensions as bx

from numpy.typing import NDArray


class RedisConfig(BaseSettings):
  redis_dsn: RedisDsn = Field('redis://user:pass@localhost:6379/1')

redisconfig = RedisConfig() 


class IrisFeature(bx.TypedDict, total=False):
  pk: str
  sepal_len: float | int
  sepal_width: float
  petal_len: float | int


repository = bx.featurestore.redis(config=redisconfig, data_model=IrisFeature).to_runner(embedded=True)
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_svc", runners=[iris_clf_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(feature_keys: list[str]) -> Dict[str,list[float]]:
  
  # features: list[IrisFeature] = await repository.filter(ids=feature_keys).aget_many()
  features: list[list[float]] = await repository.async_filter(ids=feature_keys, _nokey=True).get_many() # async_get_many()
  #features: list[list[float]] = repository.filter(ids=feature_keys, _nokey=True).get_many()
  # features: IrisFeature = await repository.filter(ids=feature_keys).aget()


  
  features: NDArray = await repository.afilter(ids=feature_keys,_return_np=True)
  inputs: torch.tensor =  torch.from_numpy(features)
  result = await iris_clf_runner.predict.async_run(inputs)
  return { "result": result.tolist() }
  

~~~



## CPU Optimized Runner
  * `bentoml-extensions[ipex]`
  * `bentoml-extensions[ovms]` `like a bentoml[triton]`

~~~Python
import bentoml
import bentoml_extensions as bentomlx


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
