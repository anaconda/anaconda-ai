from os.path import expandvars
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from anaconda_cli_base.config import AnacondaBaseSettings


class OllamaConfig(BaseModel):
    models_path: Path = Path("~/.ollama/models/blobs").expanduser()
    servers_path: Path = Path("~/.ollama/servers").expanduser()
    ollama_base_url: str = "http://localhost:11434"

    @field_validator("models_path")
    def expand_vars_models_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()

    @field_validator("servers_path")
    def expand_vars_servers_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()


class AICatalogConfig(BaseModel):
    domain: str = "anaconda.com"
    api_version: str = "2"
    models_path: Path = Path("~/.anaconda/models").expanduser()


class Backends(BaseModel):
    ai_catalog: AICatalogConfig = AICatalogConfig()
    ollama: OllamaConfig = OllamaConfig()


class AnacondaAIConfig(AnacondaBaseSettings, plugin_name="ai"):
    backends: Backends = Backends()
    backend: Literal["ollama", "ai-catalog"] = "ai-catalog"
    stop_server_on_exit: bool = True
