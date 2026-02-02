import json
from os.path import expandvars
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import BaseModel, field_validator, field_serializer, Field

from anaconda_ai.exceptions import AINavigatorConfigError
from anaconda_cli_base.config import AnacondaBaseSettings


class AICatalystConfig(BaseModel):
    api_version: str = "2"
    models_path: Path = Path("~/.anaconda/ai/models").expanduser()

    @field_serializer("models_path")
    def serialize_models_path(self, models_path: Path) -> str:
        return str(models_path)

    @field_validator("models_path")
    def expand_vars_models_path(cls, v: str) -> Path:
        return Path(expandvars(v)).expanduser()


class AINavigatorConfig(BaseModel):
    app_name: str = "ai-navigator"

    @property
    def models_path(self) -> Path:
        path = self.get_config("downloadLocation")
        if path is None:
            raise AINavigatorConfigError(
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
                    "There was a problem reading the AINavigator application config file"
                )
        return config.get(key)

    @property
    def port(self) -> int:
        port = self.get_config("aiNavApiServerPort")
        if port is None:
            raise AINavigatorConfigError(
                "The API Port was not found in the application config file."
            )
        return port

    @property
    def api_key(self) -> str:
        key = self.get_config("aiNavApiKey")
        if key is None:
            raise AINavigatorConfigError(
                "The API Key was not found in the application config file."
            )
        return key


class AnacondaDesktopConfig(AINavigatorConfig):
    app_name: str = "anaconda-desktop"


class Backends(BaseModel):
    ai_catalyst: AICatalystConfig = Field(default_factory=AICatalystConfig)
    ai_navigator: AINavigatorConfig = Field(default_factory=AINavigatorConfig)
    anaconda_desktop: AnacondaDesktopConfig = Field(
        default_factory=AnacondaDesktopConfig
    )


class AnacondaAIConfig(AnacondaBaseSettings, plugin_name="ai"):
    backends: Backends = Field(default_factory=Backends)
    backend: Literal["ai-catalyst", "ai-navigator", "anaconda-desktop"] = (
        "anaconda-desktop"
    )
    stop_server_on_exit: bool = True
    server_operations_timeout: int = 60
    show_blocked_models: bool = False
