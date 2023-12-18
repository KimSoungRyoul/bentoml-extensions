from __future__ import annotations

import time
import aerospike
import typing as t
from statistics import mean
from typing import TYPE_CHECKING

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

from bentoml_extensions.feature_store.aerospike import DBSettings

if TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod, Runner


    class RunnerImpl(bentoml.Runner):
        is_positive: RunnerMethod

R = t.TypeVar("R")


class AerospikeRepositoryRunnable(t.Generic[R], bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    db_settings: DBSettings
    client: aerospike.Client

    def __init__(self, db_settings: DBSettings):
        self.db_settings = db_settings
        client = aerospike.client({"hosts": self.db_settings.hosts, })
        if self.db_settings.username:
            self.client = client.connect(self.db_settings.username, self.db_settings.password)
        else:
            self.client = client.connect()

    @bentoml.Runnable.method(batchable=False)
    def get(self, set_: str, pk: t.Any) -> R:
        (_, _, pk, _), _meta, bins = self.client.get((self.db_settings.namespace, set_, pk))
        bins["pk"] = pk
        return bins

    def get_many(self, set_: str, pks: t.List[t.Any]) -> t.List[R]:
        ...

    def save(self, record: R):
        ...

    def save_all(self, records: t.List[R]):
        ...




def to_runner(db_settings: DBSettings,  embedded: bool = False,) -> Runner:
    return bentoml.Runner(
        AerospikeRepositoryRunnable,
        name="aerospike_repository",
        embedded= embedded,
        runnable_init_params={
            "db_settings": db_settings,
        }
    )



repository = bentoml.Runner(
     AerospikeRepositoryRunnable,
     name="aerospike_repository",
     runnable_init_params={
         "model_file": "./saved_model_1.pt",
     }
 )