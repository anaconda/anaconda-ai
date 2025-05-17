from pathlib import Path
from typing import Type, Tuple, Optional, Dict
from typing_extensions import Self

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from .clients import get_default_client
from .clients.base import APIParams, LoadParams, InferParams
from .clients.base import VectorDbTableSchema

DEFAULT_SPEC_PATH = Path("anaconda-ai.toml")


class Inference(BaseModel, extra="allow"):
    model: str
    api_params: Optional[APIParams] = None
    load_params: Optional[LoadParams] = None
    infer_params: Optional[InferParams] = None


class VectorDB(BaseModel):
    table: str
    schema: VectorDbTableSchema


class AISpec(BaseSettings, extra="ignore"):
    inference: Dict[str, Inference] = {}
    vector_db: Optional[VectorDB] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)

    @classmethod
    def load(cls, path: Path = DEFAULT_SPEC_PATH) -> Self:
        source = TomlConfigSettingsSource(cls, path)
        return cls(**source())

    def up(self) -> None:
        client = get_default_client()

        for name, server_config in self.inference.items():
            server = client.servers.create(
                model=server_config.model,
                api_params=server_config.api_params,
                infer_params=server_config.infer_params,
                load_params=server_config.load_params,
            )
            server.start(show_progress=True, leave_running=True)

        if self.vector_db:
            vector_db = client.vector_db.create(show_progress=True, leave_running=True)
            if vector_db.running:
                client.vector_db.create_table(
                    self.vector_db.table, self.vector_db.schema
                )

    def down(self) -> None:
        client = get_default_client()

        for server_config in self.inference.values():
            server = client.servers.create(
                model=server_config.model,
                api_params=server_config.api_params,
                infer_params=server_config.infer_params,
                load_params=server_config.load_params,
            )
            if server._matched:
                server.stop()

        if self.vector_db:
            client.vector_db.drop_table(self.vector_db.table)
            client.vector_db.stop()
