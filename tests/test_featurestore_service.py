import logging
from functools import lru_cache
from typing import Dict, TypedDict

import bentoml
import numpy as np
from async_lru import alru_cache
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
db_settings = (
    DBSettings()
)  # EXPORT ENV BENTOML_REPO_NAMESPACE=test; BENTOML_REPO__HOSTS=localhost:3000; BENTOML_REPO__USE_SHARED_CONNECTION=true

repo_runner = bentomlx.feature_repo.aerospike_fs(db_settings).to_repo_runner(
    entity_name="iris_features", embedded=True
)

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_svc", runners=[repo_runner, iris_clf_runner])

logger = logging.getLogger("bentoml")


@svc.api(
    input=JSON.from_sample(["pk1", "pk2", "pk3"]),
    output=JSON(),
)
async def classify(feature_keys: list[str]) -> Dict[str, list[float]]:
    # features: list[IrisFeature] = repository.filter(ids=feature_keys).get_many()
    # features: list[list[float]] = await repository.afilter(ids=feature_keys,_nokey=True).get_many()
    # feature = repository.get_all.run(page_size=10)
    features: list[IrisFeature] = repo_runner.get_many.run(pks=feature_keys)

    # input_arr = [[4.9 3.  1.4 0.2], [5.1 3.5 1.4 0.3], [5.5 2.5 4.  1.3]]
    input_arr = np.array(
        [[v for k, v in row.items() if k != "pk"] for row in features if row]
    )

    result: np.ndarray = await iris_clf_runner.predict.async_run(input_arr)
    return {"result": result.tolist()}


# INSERT INTO test.iris_features (PK, feature1, feature2, feature3, feature4) VALUES ('pk1', 4.9, 3.0, 1.4, 0.2);
# INSERT INTO test.iris_features (PK, feature1, feature2, feature3, feature4) VALUES ('pk2', 5.1, 3.5, 1.4, 0.3);
# INSERT INTO test.iris_features (PK, feature1, feature2, feature3, feature4) VALUES ('pk3', 5.5, 2.5, 4.0, 1.3);
