from __future__ import annotations

import math
import re
import typing as t
from functools import lru_cache
from typing import TYPE_CHECKING

import aerospike
import bentoml
from aerospike_helpers.batch import records as br
from aerospike_helpers.operations import operations as op
from ..repository import FeatureRepository
from ...settings import DBSettings

if TYPE_CHECKING:
    from bentoml._internal.runner.runner import Runner, RunnerMethod

    class RunnerImpl(bentoml.Runner):
        is_positive: RunnerMethod


P = t.TypeVar("P", bound=t.Any, covariant=True)  # PK
R = t.TypeVar("R", bound=dict[str, t.Any] | None, covariant=True)  # Record


class AerospikeFeatureRepositoryRunnable(FeatureRepository[P, R], bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    aerospike_namespace: str
    aerospike_set: str

    client: aerospike.Client
    read_policy = {"key": aerospike.POLICY_KEY_SEND}
    write_policy = {"key": aerospike.POLICY_KEY_SEND}
    remove_policy = {"durable_delete": False}

    default_meta = {"ttl": aerospike.TTL_NEVER_EXPIRE}

    def __init__(self, db_settings: DBSettings, set_name: str = None):
        config = {
            "hosts": [
                (host_str.split(":")[0], int(host_str.split(":")[1]))
                for host_str in db_settings.HOSTS
            ],
            "polices": {
                "read": self.read_policy,
                "write": self.write_policy,
                "remove": self.remove_policy,
            },
            "use_shared_connection": db_settings.USE_SHARED_CONNECTION,
        }
        self.client = aerospike.Client(config)
        self.aerospike_namespace = db_settings.NAMESPACE
        self.aerospike_set = set_name

    @bentoml.Runnable.method(batchable=False)
    def get_all(
        self, page_size: int = 10, page_num: int = 1, with_ttl=False
    ) -> list[R]:
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

        return records

    @bentoml.Runnable.method(batchable=False)
    def get_many(self, pks: list[P], with_ttl=False) -> list[R]:
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
        return records

    @bentoml.Runnable.method(batchable=False)
    def get(self, pk: P, with_ttl=False) -> R:
        pk_tuple = (self.aerospike_namespace, self.aerospike_set, pk)
        (_ns, _set, pk, _), _meta, bins = self.client.get(
            pk_tuple, policy=self.read_policy
        )
        if bins:
            bins = {
                k: v if not (isinstance(v, float) and math.isnan(v)) else None
                for k, v in bins.items()
            }
            record = {"pk": pk} | bins
            if with_ttl:
                record = record | {"ttl": _meta["ttl"]}
            return record
        else:
            return None

    @bentoml.Runnable.method(batchable=False)
    def save(self, record: R, ttl: int | None = None):
        """
        ttl: seconds
        """
        pk = record["pk"]
        bins = {k: v for k, v in record.items() if k != "pk"}
        self.client.put(
            key=(self.aerospike_namespace, self.aerospike_set, pk),
            bins=bins,
            meta={"ttl": ttl},
            policy=self.write_policy,
        )

    @bentoml.Runnable.method(batchable=False)
    def save_all(self, records: list[R], ttl: int | None = None) -> None:
        """
        ttl: seconds
        """
        pk_list = []
        bins_list = []
        for record in records:
            pk_list.append(record.pop("pk"))
            for _bin_name, _bin_value in record.items():
                bins_list.append(record)

        op_list = [
            br.Write(
                key=(self.aerospike_namespace, self.aerospike_set, pk),
                ops=[
                    op.write(_bin_name, _bin_value)
                    for _bin_name, _bin_value in bins.items()
                ],
                meta={"ttl": ttl},
                policy=self.write_policy,
            )
            for pk, bins in zip(pk_list, bins_list)
        ]
        batch_records = br.BatchRecords(op_list)
        self.client.batch_write(batch_records)

    @bentoml.Runnable.method(batchable=False)
    def delete(self, pk: P):
        self.client.remove(
            (self.aerospike_namespace, self.aerospike_set, pk),
            policy=self.remove_policy,
        )

    @bentoml.Runnable.method(batchable=False)
    def delete_all(self, pks: list[P]):
        self.client.batch_remove(
            [(self.aerospike_namespace, self.aerospike_set, pk) for pk in pks],
            policy_batch_remove=self.remove_policy,
        )

    @bentoml.Runnable.method(batchable=False)
    def count(self) -> int:
        """
        특정 SET의 총 갯수만 count() 가능합니다.
        """
        conn: aerospike.Client = self.client.connect()
        set_info_list = [
            _set_info
            for _, _set_info in conn.info_all(
                f"sets/{self.aerospike_namespace}/{self.aerospike_set}"
            ).values()
        ]
        pattern = re.compile(r"objects=(\d+)")
        count_list_by_each_node = []
        for set_info_text in set_info_list:
            r: list[str] = pattern.findall(set_info_text)
            if len(r) != 0:
                count_list_by_each_node.append(int(r[0]))
            else:
                count_list_by_each_node.append(0)

        return sum(count_list_by_each_node)

    # @classmethod
    # def to_runner(cls, db_settings: DBSettings, embedded: bool = False) -> Runner:
    #     return bentoml.Runner(
    #         AerospikeRepositoryRunnable,
    #         name="aerospike_repository",
    #         embedded=embedded,
    #         runnable_init_params={
    #             "db_settings": db_settings,
    #         }
    #     )
