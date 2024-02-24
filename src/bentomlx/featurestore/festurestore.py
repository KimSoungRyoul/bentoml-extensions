import typing as t

import bentoml

if t.TYPE_CHECKING:
    from ..featurestore import DBSettings


class FeatureStore:
    db_settings: "DBSettings"

    def __init__(
        self, db_settings: "DBSettings", runnable_class: t.Type[bentoml.Runnable]
    ):
        self.db_settings = db_settings
        self.runnable_class = runnable_class

    def to_runner(self, set_name: str = None, embedded: bool = False) -> bentoml.Runner:
        return bentoml.Runner(
            runnable_class=self.runnable_class,
            name="aerospike_repository_runner",
            runnable_init_params={
                "db_settings": self.db_settings,
                "set_name": set_name,
            },
            embedded=embedded,
        )

    def to_repo_runner(
        self, entity_name: str = None, embedded: bool = False
    ) -> bentoml.Runner:
        return bentoml.Runner(
            runnable_class=self.runnable_class,
            name="aerospike_feature_repository_runner",
            runnable_init_params={
                "db_settings": self.db_settings,
                "set_name": entity_name,
            },
            embedded=embedded,
        )
