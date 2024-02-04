import typing as t
from typing import Any

import bentoml

if t.TYPE_CHECKING:
    from .. import DBSettings


def aerospike_fs(db_settings: "DBSettings" = None) -> "FeatureStore":
    from ... import FeatureStore
    from .repository import AerospikeFeatureRepositoryRunnable

    if db_settings is None:
        db_settings = DBSettings()

    return FeatureStore(
        db_settings=db_settings,
        runnable_class=AerospikeFeatureRepositoryRunnable,
    )


__all__ = [
    # "to_runner",
    "aerospike_fs"
]
