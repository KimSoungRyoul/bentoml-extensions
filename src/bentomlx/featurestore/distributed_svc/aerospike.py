import math

import bentoml
import aerospike
import typing as t
from pydantic import Field, BaseModel

from .repository import AsyncFeatureRepository

P = t.TypeVar("P", bound=t.Any, covariant=True)  # PK
R = t.TypeVar("R", bound=dict[str, t.Any] | None, covariant=True)  # Record


class RepoResponse(BaseModel):
    records: t.List[dict[str, t.Any]] = Field(default_factory=list)


@bentoml.service(
    workers=1
)
class AsyncFeatureRepository:
    aerospike_namespace: str
    aerospike_set: str

    client: aerospike.Client
    read_policy = {"key": aerospike.POLICY_KEY_SEND}
    write_policy = {"key": aerospike.POLICY_KEY_SEND}
    remove_policy = {"durable_delete": False}

    default_meta = {"ttl": aerospike.TTL_NEVER_EXPIRE}

    def __init__(self, db_settings: dict[str, t.Any]= None):
        """

        :param db_settings:
            {
                "HOSTS": ["127.0.0.1:3000"],
                "NAMESPACE": "test",
                "SET_NAME": "sample_feature"
                "USERNAME": "user",
                "PASSWORD":" password",
                "USE_SHARED_CONNECTION": False
            }
        """
        if db_settings is None:
            db_settings = {
                "HOSTS": ["127.0.0.1:3000", ],
                "NAMESPACE": "test",
                "SET_NAME": "sample_feature",
                # "USERNAME": "user",
                # "PASSWORD": " password",
                "USE_SHARED_CONNECTION": False
            }

        config = {
            "hosts": [
                (host_str.split(":")[0], int(host_str.split(":")[1]))
                for host_str in db_settings["HOSTS"]
            ],
            "polices": {
                "read": self.read_policy,
                "write": self.write_policy,
                "remove": self.remove_policy,
            },
            "use_shared_connection": db_settings.get("USE_SHARED_CONNECTION", False),
        }
        self.client = aerospike.Client(config)
        self.aerospike_namespace = db_settings["NAMESPACE"]
        self.aerospike_set = db_settings["SET_NAME"]

    @bentoml.api
    async def get_all(
            self,
            page_size: int = Field(default=10),
            page_num: int = Field(default=1),
            with_ttl: bool = Field(default=False),
    ) -> RepoResponse:
        """
        page_size: 10
        page_num: 1
        """
        query = self.client.query(self.aerospike_namespace, self.aerospike_set)
        query.max_records = page_size
        query.paginate()

        records_: list[R] = []
        for _ in range(page_num):
            if query.is_done() is False:
                records_ = query.results(policy=self.read_policy)
            else:
                records_ = []
                break

        records: list[R] = []
        for (_ns, _set, pk, _), _meta, bins in records_:
            record = {"pk": pk} | bins
            if with_ttl:
                record = record | {"ttl": _meta["ttl"]}
            records.append(record)

        return RepoResponse(records=records)

    @bentoml.api
    async def get_many(
            self,
            pks: t.List[P] = Field(),
            with_ttl: bool = Field(default=False)
    ) -> RepoResponse:
        _records = self.client.get_many(
            [(self.aerospike_namespace, self.aerospike_set, pk) for pk in pks],
            policy=self.read_policy,
        )
        records: list[R] = []
        for (_ns, _set, pk, _), _meta, bins in _records:
            if bins:
                bins = {
                    k: v if not (isinstance(v, float) and math.isnan(v)) else None
                    for k, v in bins.items()
                }
                record = {"pk": pk} | bins
                if with_ttl:
                    record = record | {"ttl": _meta["ttl"]}
                records.append(record)
            else:
                records.append(None)
        return RepoResponse(records=records)
    #
    # async def get(self, pk: P, with_ttl=False) -> R: ...
    #
    # async def save(self, record: R, ttl: int | None = None): ...
    #
    # async def save_all(self, records: t.List[R], ttl: int | None = None) -> None: ...
    #
    # async def delete(self, pk: P): ...
    #
    # async def delete_all(self, pks: t.List[P]): ...
    #
    # async def count(self) -> int: ...
