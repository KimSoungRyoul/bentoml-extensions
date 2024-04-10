import logging
from typing import TypedDict

import bentoml
from bentoml.io import NumpyNdarray, JSON

import bentomlx

logger = logging.getLogger("bentoml")

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()

redis_settings = {
    "HOST": "localhost",
    "PORT": 6379,
    "DB": 0,
}


class RedisRecord(TypedDict):
    pk: str
    feature1: float
    feature2: float


redis_fs_repo_runner = bentomlx.featurestore.runner.redis(
    db_settings=redis_settings, record_class=RedisRecord, embedded=True
)

svc = bentoml.Service("linear_regression", runners=[reg_runner, redis_fs_repo_runner])


@svc.api(input=JSON.from_sample(["feature_key1", "feature_key2", "feature_key3"]),
         output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]]))
async def predict(feature_key_list: list[str]):
    np_arr = await redis_fs_repo_runner.get_many.async_run(pks=feature_key_list, np=True)  # np=True -> return ndarray
    result = await reg_runner.predict.async_run(np_arr)
    return result
