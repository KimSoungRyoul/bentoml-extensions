import bentoml
import numpy as np

import bentomlx
from bentoml.io import NumpyNdarray, JSON

from bentomlx.featurestore import DBSettings

db_settings = DBSettings(HOSTS=["localhost:3000", ], NAMESPACE="test")

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
fs_repo_runner = bentomlx.featurestore.aerospike(db_settings).to_repo_runner(entity_name="sample_feature")

svc = bentoml.Service("linear_regression", runners=[reg_runner, fs_repo_runner])

input_spec = NumpyNdarray.from_sample([[1, 0.546456456]])


@svc.api(input=JSON.from_sample(["pk1", "pk2", "pk3"]), output=NumpyNdarray.from_sample([[0.111, 0.222, 0.333]]))
async def predict(feature_key_list: list[str]):
    _features: list[dict[str, float]] = await fs_repo_runner.get_many.async_run(pks=feature_key_list)
    features = np.array([[v for k, v in row.items() if k != "pk"] for row in _features], dtype=np.float32)

    result = await reg_runner.predict.async_run(features)

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
