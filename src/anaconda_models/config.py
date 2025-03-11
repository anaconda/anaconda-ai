import os
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

import platformdirs
from pydantic import BaseModel
from pydantic import field_validator

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


class KuratorConfig(BaseModel):
    domain: str = "kurator.anaconda.com"
    ssl_verify: bool = True
    extra_headers: Optional[Union[Dict[str, str], str]] = None
    run_on: Literal["local"] = "local"
    local_servers_path: Path = Path("~/.ai-navigator/local-servers").expanduser()
    models_path: Path = Path("~/.ai-navigator/models").expanduser()

    @field_validator("local_servers_path")
    def expand_variables_servers_path(cls, v: Union[Path, str]) -> Path:
        return Path(os.path.expandvars(v)).expanduser()

    @field_validator("models_path")
    def expand_variables_models_path(cls, v: Union[Path, str]) -> Path:
        return Path(os.path.expandvars(v)).expanduser()


class Backends(BaseModel):
    ai_navigator: AINavigatorConfig = AINavigatorConfig()
    kurator: KuratorConfig = KuratorConfig()


class AnacondaModelsConfig(AnacondaBaseSettings, plugin_name="models"):
    default_backend: Literal["kurator", "ai-navigator"] = "ai-navigator"
    backends: Backends = Backends()
    stop_server_on_exit: bool = False
