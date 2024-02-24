from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import aerospike

if TYPE_CHECKING:
    from ..settings import DBSettings


P = t.TypeVar("P")
R = t.TypeVar("R", bound=dict[str, t.Any] | None, covariant=True)


class FeatureRepository(t.Protocol[P, R]):  # bentoml.Runnable,
    db_settings: DBSettings
    client: aerospike.Client

    def get_all(
            self, page_size: int = 10, page_num: int = 1, with_ttl=False
    ) -> t.List[R]: ...

    def get_many(self, pks: t.List[P], with_ttl=False) -> t.List[t.Optional[R]]: ...

    def get(self, pk: P, with_ttl=False) -> R: ...

    def save(self, record: R, ttl: int | None = None): ...

    def save_all(self, records: t.List[R], ttl: int | None = None) -> None: ...

    def delete(self, pk: P): ...

    def delete_all(self, pks: t.List[P]): ...

    def count(self) -> int: ...
