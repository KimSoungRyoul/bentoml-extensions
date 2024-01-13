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

# import re
# from typing import (
#     TypeVar,
#     Dict,
#     Any,
#     List,
#     Optional,
# )
#
# import aerospike
# from aerospike_helpers.batch import records as br
# from aerospike import predicates
# from aerospike_helpers.operations import operations as op
# from typing_extensions import Protocol
#
# P = TypeVar("P", bound=Any, covariant=True)  # PK
# R = TypeVar("R", bound=Dict[str, Any], covariant=True)  # Record
#
#
# class AerospikeRepository(Protocol[P, R]):
#     aerospike_namespace: str
#     aerospike_set: str
#
#     client: aerospike.Client
#     read_policy = {"key": aerospike.POLICY_KEY_SEND}
#     write_policy = {"key": aerospike.POLICY_KEY_SEND}
#     remove_policy = {"durable_delete": False}
#
#     default_meta = {"ttl": aerospike.TTL_NEVER_EXPIRE}
#
#     def __init__(self, client: aerospike.Client, aerospike_namespace: str, aerospike_set: str):
#         self.client = client
#         self.aerospike_namespace = aerospike_namespace
#         self.aerospike_set = aerospike_set
#
#     def get_all(self, page_size: int = 10, page_num: int = 1, with_ttl=False) -> List[R]:
#         """
#             paginate: {"page_size": 10 , "page_num":1 }
#         """
#         query = self.client.query(self.aerospike_namespace, self.aerospike_set)
#         query.max_records = page_size
#         query.paginate()
#
#         records_: list[R] = []
#         for _ in range(page_num):
#             if query.is_done() is False:
#                 records_ = query.results(policy=self.read_policy)
#             else:
#                 records_ = []
#                 break
#
#         records: List[R] = []
#         for (_ns, _set, pk, _), _meta, bins in records_:
#             record = {"pk": pk} | bins
#             if with_ttl:
#                 record = record | {"ttl": _meta["ttl"]}
#             records.append(record)
#
#         return records
#
#     def get_many(self, pks: List[P], with_ttl=False) -> List[Optional[R]]:
#         pk_tuples = [(self.aerospike_namespace, self.aerospike_set, pk) for pk in pks]
#         records_ = self.client.get_many(pk_tuples, policy=self.read_policy)
#
#         qq=self.client.query()
#         qq.where(predicates.between("pk",))
#
#         records: List[R] = []
#         for (_ns, _set, pk, _), _meta, bins in records_:
#             record = {"pk": pk} | bins
#             if with_ttl:
#                 record = record | {"ttl": _meta["ttl"]}
#             records.append(record)
#         return records
#
#     def get(self, pk: P, with_ttl=False) -> R:
#         pk_tuple = (self.aerospike_namespace, self.aerospike_set, pk)
#         (_ns, _set, pk, _), _meta, bins = self.client.get(pk_tuple, policy=self.read_policy)
#         record = {"pk": pk} | bins
#         if with_ttl:
#             record = record | {"ttl": _meta["ttl"]}
#         return record
#
#     def save(self, record: R, ttl: int | None = None):
#         """
#             ttl: seconds
#         """
#         pk = record["pk"]
#         bins = {k: v for k, v in record.items() if k != "pk"}
#         self.client.put(
#             key=(self.aerospike_namespace, self.aerospike_set, pk),
#             bins=bins,
#             meta={"ttl": ttl},
#             policy=self.write_policy,
#         )
#
#     def save_all(self, records: List[R], ttl: int | None = None) -> None:
#         """
#             ttl: seconds
#         """
#         pk_list = []
#         bins_list = []
#         for record in records:
#             pk_list.append(record.pop("pk"))
#             for _bin_name, _bin_value in record.items():
#                 bins_list.append(record)
#
#         op_list = [
#             br.Write(
#                 key=(self.aerospike_namespace, self.aerospike_set, pk),
#                 ops=[
#                     op.write(_bin_name, _bin_value)
#                     for _bin_name, _bin_value in bins.items()
#                 ],
#                 meta={"ttl": ttl},
#                 policy=self.write_policy,
#             )
#             for pk, bins in zip(pk_list, bins_list)
#         ]
#         batch_records = br.BatchRecords(op_list)
#         self.client.batch_write(batch_records)
#
#     def delete(self, pk: P):
#         self.client.remove((self.aerospike_namespace, self.aerospike_set, pk), policy=self.remove_policy)
#
#     def delete_all(self, pks: List[P]):
#         self.client.batch_remove([(self.aerospike_namespace, self.aerospike_set, pk) for pk in pks],
#                                  policy_batch_remove=self.remove_policy)
#
#     def count(
#             self,
#     ) -> int:
#         """
#            특정 SET의 총 갯수만 count() 가능합니다.
#         """
#         conn: aerospike.Client = self.client.connect()
#         set_info_list = [_set_info for _, _set_info in
#                          conn.info_all(f"sets/{self.aerospike_namespace}/{self.aerospike_set}").values()]
#         pattern = re.compile(r"objects=(\d+)")
#         count_list_by_each_node = []
#         for set_info_text in set_info_list:
#             r: list[str] = pattern.findall(set_info_text)
#             if len(r) != 0:
#                 count_list_by_each_node.append(int(r[0]))
#             else:
#                 count_list_by_each_node.append(0)
#
#         return sum(count_list_by_each_node)

def to_runner(db_settings: DBSettings, embedded: bool = False) -> Runner:
    return bentoml.Runner(
        AerospikeRepositoryRunnable,
        name="aerospike_repository",
        embedded=embedded,
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
