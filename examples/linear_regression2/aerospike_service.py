import logging

import bentoml
import numpy as np
import numpy.random
from pydantic import BaseModel, Field

import bentomlx

logger = logging.getLogger("bentoml")


db_settings = {
    "HOSTS": ["127.0.0.1:3000", ],
    "NAMESPACE": "test",
    "SET_NAME": "sample_feature",
    # "USERNAME": "user",
    # "PASSWORD": " password",
    "USE_SHARED_CONNECTION": False
}

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
fs_repo_runner = bentomlx.featurestore.runner.aerospike(db_settings=db_settings, embedded=True)

class Bento1Request(BaseModel):
    feature_key_list: list[str] = Field(description="pk 리스트", example=["pk1", "pk2"])


@bentoml.service(
    workers=1
)
class HelloBentoService:
    repo =  bentoml.depends(
            on=bentomlx.featurestore.aerospike.AsyncFeatureRepository
        )


    # input=JSON.from_sample(["pk1", "pk2", "pk3"]), output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]])
    @bentoml.api(route="/predict")
    async def predict(self, bento_request: Bento1Request) -> np.ndarray:
        _features: list[dict[str, float]] = await self.repo.get_many(pks=bento_request.feature_key_list)
        logger.info(f"Aerospike 데이터: {_features}")
        # _redis_features = await redis_fs_repo_runner.aget_many.async_run(pks=bento_request.feature_key_list,
        #                                                                  field_names=["feature1", "feature2"])

        # _redis_features = await _redis_features
        # features = np.array([[v for k, v in row.items() if k != "pk"] for row in _features], dtype=np.float32)
        #
        # redis_features = np.array([[v for k, v in row.items() if k != "pk"] for row in _redis_features],
        #                           dtype=np.float32)

        # result = await reg_runner.predict.async_run(features)
        #
        # result = await reg_runner.predict.async_run(redis_features)

        return numpy.random.randn(3)

# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk1', 0.9, 0.8);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk2', 0.7, 0.6);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk3', 0.6, 0.9);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk4', 0.8, 0.5);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk5', 0.4, 0.7);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk6', 0.6, 0.4);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk7', 0.9, 0.3);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk8', 0.4, 0.2);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk9', 0.9, 0.8);
# INSERT INTO test.sample_feature (PK, feature1, feature2) VALUES ('pk10', 0.5, 0.6);
