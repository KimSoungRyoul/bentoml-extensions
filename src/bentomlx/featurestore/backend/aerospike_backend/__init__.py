import typing as t
from typing import Any

import bentoml

if t.TYPE_CHECKING:
    from .. import DBSettings


def aerospike(db_settings: "DBSettings") -> "FeatureStore":
    from ... import FeatureStore
    from .repository import AerospikeFeatureRepositoryRunnable

    return FeatureStore(
        db_settings=db_settings,
        runnable_class=AerospikeFeatureRepositoryRunnable,
    )


__all__ = [
    # "to_runner",
    "aerospike"
]
