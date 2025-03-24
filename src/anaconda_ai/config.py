import json
from pathlib import Path
from typing import Any
from typing import Literal

import platformdirs
from pydantic import BaseModel

from anaconda_cli_base.config import AnacondaBaseSettings
from anaconda_models.exceptions import APIKeyMissing


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


class Backends(BaseModel):
    ai_navigator: AINavigatorConfig = AINavigatorConfig()


class AnacondaModelsConfig(AnacondaBaseSettings, plugin_name="models"):
    backends: Backends = Backends()
    default_backend: Literal["ai-navigator"] = "ai-navigator"
    stop_server_on_exit: bool = False
