import bentoml
from . import runner

_, major, minor = bentoml.__version__.split(".")

if int(major) >= 2:
    # distributed service is supported in 1.2.x and later
    from .distributed_svc import aerospike

# from .backend.redis_backend import redis
from .festurestore import FeatureStore
from .settings import DBSettings

__all__ = [
    "aerospike",
    "runner",
    DBSettings,

]
