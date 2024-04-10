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
LinearRegService = bentoml.runner_service(runner=reg_runner)


class BentoRequest(BaseModel):
    feature_key_list: list[str] = Field(description="pk list", example=["pk1", "pk2"])


@bentoml.service(
    workers=1
)
class HelloBentoService:
    repo = bentoml.depends(
        # todo: inner SVC, there is no way to inject db_settings....
        on=bentomlx.featurestore.aerospike.AsyncFeatureRepository
    )
    linear_reg = bentoml.depends(
        on=LinearRegService
    )

    @bentoml.api(route="/predict")
    async def predict(self, bento_request: BentoRequest) -> np.ndarray:
        nd_arr: np.ndarray = await self.repo.get_many(pks=bento_request.feature_key_list, _np=True)
        logger.info(f"Aerospike data: {nd_arr.tolist()}")
        result = await reg_runner.predict.async_run(nd_arr)
        return result

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
