import json
from os.path import expandvars
from pathlib import Path
from typing import Any
from typing import Literal

import platformdirs
from pydantic import BaseModel, field_validator

from anaconda_cli_base.config import AnacondaBaseSettings
from .exceptions import APIKeyMissing


class AINavigatorConfig(BaseModel):
    app_name: str = "ai-navigator"
    port: int = 8001

    @property
    def config_file(self) -> Path:
        return Path(platformdirs.user_data_dir(self.app_name)) / "config.json"

    def get_config(self, key: str) -> Any:
        with self.config_file.open("r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(
                    "There was a problem reading the AINavigator application config file"
                )
        return config.get(key)

    @property
    def api_key(self) -> str:
        key = self.get_config("aiNavApiKey")
        if key is None:
            raise APIKeyMissing("Error: The API Key was not found in the config file.")
        return key


class OllamaConfig(BaseModel):
    models_path: Path = Path("~/.ollama/models/blobs").expanduser()
    servers_path: Path = Path("~/.ollama/servers").expanduser()
    kurator_domain: str = "kurator.anaconda.com"
    ollama_base_url: str = "http://localhost:11434"

    @field_validator("models_path")
    def expand_vars_models_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()

    @field_validator("servers_path")
    def expand_vars_servers_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()


class Backends(BaseModel):
    ai_navigator: AINavigatorConfig = AINavigatorConfig()
    ollama: OllamaConfig = OllamaConfig()


class AnacondaAIConfig(AnacondaBaseSettings, plugin_name="ai"):
    backends: Backends = Backends()
    default_backend: Literal["ai-navigator", "ollama"] = "ai-navigator"
    stop_server_on_exit: bool = True
