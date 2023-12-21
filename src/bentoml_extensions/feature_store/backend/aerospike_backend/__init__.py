from typing import Any

from .settings import DBSettings
from .repository import to_runner


def aerospike(db_settings: dict[str, Any]) -> "FeatureStore":
    ...


__all__ = [
    "DBSettings",
    # "to_runner",
    "aerospike"
]
