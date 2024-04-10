import typing as t
from typing import get_type_hints
import concurrent.futures
from concurrent.futures import wait
import bentoml
import orjson
import redis

from .repository import FeatureRepository

P = t.TypeVar("P", bound=t.Any, covariant=True)  # PK
R = t.TypeVar("R", bound=dict[str, t.Any] | None, covariant=True)  # Record


class RedisFeatureRepositoryRunnable(FeatureRepository[P, R], bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    _pool: redis.ConnectionPool

    database: int

    executor = concurrent.futures.ThreadPoolExecutor()

    # client: redis.Redis
    @property
    def client(self) -> redis.Redis:
        return redis.Redis.from_pool(self._pool)

    def __init__(self, db_settings: dict[str, t.Any], record_class: t.Type[t.TypedDict]) -> None:
        self._pool = redis.ConnectionPool(
            host=db_settings["HOST"],
            port=db_settings["PORT"],
            db=db_settings["DB"],
        )
        self._record_class = record_class

        self._record_metadata: dict[str, t.Type] = get_type_hints(self._record_class)
        self._field_names: list[str] = [k for k, v in self._record_metadata.items() if k != "pk"]
        self._field_types: list[t.Type] = [v for k, v in self._record_metadata.items() if k != "pk"]

    @bentoml.Runnable.method(batchable=False)
    def get_all(
            self, page_size: int = 10, page_num: int = 1, with_ttl=False
    ) -> t.List[R]:
        raise NotImplementedError("redis does not support paging")

    @bentoml.Runnable.method(batchable=False)
    def get_many(
            self, pks: t.List[P], with_ttl: bool = False, options: dict[str, t.Any] = None
    ) -> t.List[t.Optional[R]]:
        """
            :param pks:
            :param with_ttl:
            :param options:  todo: { "hashes": true } -> use hmget
            :return:
        """
        client = self.client

        records: list[dict[str, t.Any]] = []
        for pk, b_fields in zip(pks, client.mget(*pks)):
            fields = orjson.loads(b_fields)
            r = {"pk": pk}
            r.update(fields)
            records.append(r)

        return records

    #
    # def get(self, pk: P, with_ttl=False) -> R: ...
    #
    # async def asave(self, record: R, ttl: int | None = None): ...
    #
    # async def asave_all(self, records: t.List[R], ttl: int | None = None) -> None: ...
    #
    # async def adelete(self, pk: P): ...
    #
    # async def adelete_all(self, pks: t.List[P]): ...
    #
    # async def acount(self) -> int: ...
