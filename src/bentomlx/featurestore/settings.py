from typing import Any, List, Optional, Tuple, Type

from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class MyCustomSource(EnvSettingsSource):
    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        if field_name == "HOSTS" and isinstance(value,str):
            return [str(x) for x in value.split(",")]
        return value


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, env_prefix="BENTOML_REPO__")
    USERNAME: Optional[str] = None
    PASSWORD: Optional[str] = None
    HOSTS: List[str]  # = ["127.0.0.1:3000", ]
    NAMESPACE: str = "test"
    USE_SHARED_CONNECTION: bool = False

    # @classmethod
    # def settings_customise_sources(
    #     cls,
    #     settings_cls: Type[BaseSettings],
    #     init_settings: PydanticBaseSettingsSource,
    #     env_settings: PydanticBaseSettingsSource,
    #     dotenv_settings: PydanticBaseSettingsSource,
    #     file_secret_settings: PydanticBaseSettingsSource,
    # ) -> Tuple[PydanticBaseSettingsSource, ...]:
    #     return (MyCustomSource(settings_cls),)
