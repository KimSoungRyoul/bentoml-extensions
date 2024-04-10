import logging

import bentoml
from bentoml.io import NumpyNdarray, JSON

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

svc = bentoml.Service("linear_regression", runners=[reg_runner, fs_repo_runner])


@svc.api(input=JSON.from_sample(["pk1", "pk2", "pk3"]), output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]]))
async def predict(feature_key_list: list[str]):
    nd_arr: list[dict[str, float]] = await fs_repo_runner.get_many.async_run(pks=feature_key_list, numpy=True)
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


# HSET pk1 feature1 0.9 feature2 0.8
# HSET pk2 feature1 0.85 feature2 0.75
# HSET pk3 feature1 0.8 feature2 0.7
# HSET pk4 feature1 0.95 feature2 0.65
# HSET pk5 feature1 0.9 feature2 0.6
# HSET pk6 feature1 0.88 feature2 0.78
# HSET pk7 feature1 0.92 feature2 0.82
