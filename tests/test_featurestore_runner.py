from typing import TypedDict, Dict

import bentoml
import bentoml_extensions as bentomlx
from bentoml.io import JSON

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings


class IrisFeature(TypedDict, total=False):
    pk: str
    sepal_len: float | int
    sepal_width: float
    petal_len: float | int


class DBSettings(TypedDict):
    namespace: str
    hosts: list[tuple[str, int]]


db_settings = DBSettings(namespace="test", hosts=[("127.0.0.1", 3000)])

repository = bentomlx.feature_store.aerospike(db_settings).to_runner()
iris_clf_runner = bentoml.pytorch.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_svc", runners=[repository])


@svc.api(input=JSON(), output=JSON())
async def classify(feature_keys: list[str]) -> Dict[str, list[float]]:
    # features: list[IrisFeature] = repository.filter(ids=feature_keys).get_many()
    # features: list[list[float]] = await repository.afilter(ids=feature_keys,_nokey=True).get_many()
    feature = repository.get.run(set_="users", pk="i_am_key")
    print(feature)
    return {"result": []}
