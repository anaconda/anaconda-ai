import json
from pathlib import Path
from typing import Any
from typing import Literal

import platformdirs
from pydantic import BaseModel

from anaconda_cli_base.config import AnacondaBaseSettings
from .exceptions import AnacondaAIException


class AINavigatorConfigError(AnacondaAIException): ...


class AINavigatorConfig(BaseModel):
    app_name: str = "ai-navigator"

    @property
    def config_file(self) -> Path:
        # For Windows, use the roaming app data directory and do not include "author" in the path
        base_dir = Path(platformdirs.user_data_dir(self.app_name, False, roaming=True))

        return base_dir / "config.json"

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
    def port(self) -> int:
        port = self.get_config("aiNavApiServerPort")
        if port is None:
            raise AINavigatorConfigError(
                "The API Port was not found in the config file."
            )
        return port

    @property
    def api_key(self) -> str:
        key = self.get_config("aiNavApiKey")
        if key is None:
            raise AINavigatorConfigError(
                "The API Key was not found in the config file."
            )
        return key


class Backends(BaseModel):
    ai_navigator: AINavigatorConfig = AINavigatorConfig()


class AnacondaAIConfig(AnacondaBaseSettings, plugin_name="ai"):
    backends: Backends = Backends()
    backend: Literal["ai-navigator"] = "ai-navigator"
    stop_server_on_exit: bool = True
