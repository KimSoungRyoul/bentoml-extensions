import abc
from typing import Any

import bentoml
import typing as t
from .aerospike import AerospikeFeatureRepositoryRunnable
from .redis import RedisFeatureRepositoryRunnable

if t.TYPE_CHECKING:
    class Strategy:
        """
            bentoml/_internal/runner/strategy.py
        """

        @classmethod
        def get_worker_count(
                cls,
                runnable_class: type[bentoml.Runnable],
                resource_request: dict[str, t.Any] | None,
                workers_per_resource: int | float,
        ) -> int:
            ...

        @classmethod
        def get_worker_env(
                cls,
                runnable_class: type[bentoml.Runnable],
                resource_request: dict[str, t.Any] | None,
                workers_per_resource: int | float,
                worker_index: int,
        ) -> dict[str, t.Any]:
            """
            Args:
                runnable_class : The runnable class to be run.
                resource_request : The resource request of the runnable.
                worker_index : The index of the worker, start from 0.
            """
            ...


def aerospike(
        db_settings: dict[str, Any],
        name: str = "",
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, dict[str, int]] | None = None,
        embedded: bool = False,
) -> bentoml.Runner:
    return bentoml.Runner(
        runnable_class=AerospikeFeatureRepositoryRunnable,
        runnable_init_params={
            "db_settings": db_settings,
        },
        name=name if name != "" else "aerospike_featurestore_runner",
        max_batch_size=max_batch_size,
        max_latency_ms=max_latency_ms,
        method_configs=method_configs,
        embedded=embedded,
        # scheduling_strategy=scheduling_strategy,

    )


def redis(
        db_settings: dict[str, Any],
        record_class: t.Type[t.TypedDict],
        name: str = "",
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, dict[str, int]] | None = None,
        embedded: bool = False,
) -> bentoml.Runner:
    return bentoml.Runner(
        runnable_class=RedisFeatureRepositoryRunnable,
        runnable_init_params={
            "db_settings": db_settings,
            "record_class": record_class,
        },
        name=name if name != "" else "redis_featurestore_runner",
        max_batch_size=max_batch_size,
        max_latency_ms=max_latency_ms,
        method_configs=method_configs,
        embedded=embedded,
        # scheduling_strategy=scheduling_strategy,

    )


__all__ = [
    "aerospike",
    "redis",
]
