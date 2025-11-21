import json
from os.path import expandvars
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import BaseModel, computed_field, field_validator, Field

from anaconda_ai.exceptions import AnacondaDesktopConfigError
from anaconda_cli_base.config import AnacondaBaseSettings


class OllamaConfig(BaseModel):
    models_path: Path = Path("~/.ollama/models/blobs").expanduser()
    ollama_base_url: str = "http://localhost:11434"


class AICatalystConfig(BaseModel):
    api_version: str = "2"
    models_path: Path = Path("~/.anaconda/ai/models").expanduser()

    @field_validator("models_path")
    def expand_vars_models_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()


class AnacondaDesktopConfig(BaseModel):
    app_name: str = "anaconda-desktop"

    @computed_field
    @property
    def models_path(self) -> Path:
        path = self.get_config("downloadLocation")
        if path is None:
            raise AnacondaDesktopConfigError(
                "Unable to read model path from application config file"
            )
        return Path(path)

    @property
    def config_file(self) -> Path:
        # For Windows, use the roaming app data directory and do not include "author" in the path
        base_dir = Path(platformdirs.user_data_dir(self.app_name, False, roaming=True))
        return base_dir / "config.json"

    def get_config(self, key: str) -> Any:
        if not self.config_file.exists():
            return None

        with self.config_file.open("r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(
                    "There was a problem reading the Anaconda Desktop application config file"
                )
        return config.get(key)

    @property
    def port(self) -> int:
        port = self.get_config("aiNavApiServerPort")
        if port is None:
            raise AnacondaDesktopConfigError(
                "The API Port was not found in the application config file."
            )
        return port

    @property
    def api_key(self) -> str:
        key = self.get_config("aiNavApiKey")
        if key is None:
            raise AnacondaDesktopConfigError(
                "The API Key was not found in the application config file."
            )
        return key


class Backends(BaseModel):
    ai_catalyst: AICatalystConfig = Field(default_factory=AICatalystConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    anaconda_desktop: AnacondaDesktopConfig = Field(
        default_factory=AnacondaDesktopConfig
    )


class AnacondaAIConfig(AnacondaBaseSettings, plugin_name="ai"):
    backends: Backends = Field(default_factory=Backends)
    backend: Literal["ollama", "ai-catalyst", "anaconda-desktop"] = "ai-catalyst"
    stop_server_on_exit: bool = True
    server_operations_timeout: int = 30
