from typing import TypedDict, Dict

import bentoml
from bentoml.io import JSON
from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings

from bentoml_extensions import feature_store


class IrisFeature(TypedDict, total=False):
    pk: str
    sepal_len: float | int
    sepal_width: float
    petal_len: float | int


class DBSettings(feature_store.aerospike.DBSettings):
    namespace: str = "test"
    hosts: list[tuple[str, int]] = [("127.0.0.1", 3000), ]


repository  = feature_store.aerospike(DBSettings()).to_runner()
iris_clf_runner = bentoml.pytorch.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_svc", runners=[repository])


@svc.api(input=JSON(), output=JSON())
async def classify(feature_keys: list[str]) -> Dict[str ,list[float]]:

    # features: list[IrisFeature] = await repository.async_filter(ids=feature_keys)
    # features: list[list[float]] = await repository.async_filter(ids=feature_keys,_nokey=True)
    feature = repository.get.run(set_="users", pk="i_am_key")
    print(feature)
    return { "result":[] }