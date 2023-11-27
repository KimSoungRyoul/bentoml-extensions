# bentoml-extensions [![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
#### A.K.A GukBap


#### `todo`:  plan for 2024
* FeatureStore Runner [ODM], 
* optimize cpu inference [ipex, ovms]


## FeatureStore 

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
  
  # features: list[IrisFeature] = await repository.async_filter(ids=feature_keys)
  # features: list[list[float]] = await repository.async_filter(ids=feature_keys,_nokey=True)
  features: NDArray = await repository.async_filter(ids=feature_keys,_return_np=True)
  inputs: torch.tensor =  torch.from_numpy(features)
  result = await iris_clf_runner.predict.async_run(inputs)
  return { "result": result.tolist() }
  

~~~

* `pip install bentoml-extensions[featurestore-redis]`
* `pip install bentoml-extensions[featurestore-aerospike]`


## CPU Optimized Runner
  * `bentoml-extensions[ipex]`
  * `bentoml-extensions[ovms]` `like a bentoml[triton]`



## Post(Runtime) Model Compression (oneapi nncl)
  * post quant ?
  * ...





![스크린샷 2023-11-27 오후 3 18 18](https://github.com/KimSoungRyoul/bentoml-extensions/assets/24240623/8b922a8f-99e6-4d69-a713-a03f3f7b0d27)
