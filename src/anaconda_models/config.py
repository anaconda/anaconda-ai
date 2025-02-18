import os
from shutil import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

import tomlkit
from pydantic import BaseModel
from pydantic import field_validator

from anaconda_cli_base.config import AnacondaBaseSettings, anaconda_config_path


def set_config(table: str, key: str, value: Any) -> None:
    expanded = table.split(".")

    # save a backup of the config.toml just to be safe
    config_toml = anaconda_config_path()
    if config_toml.exists():
        copy(config_toml, config_toml.with_suffix(".backup.toml"))
        with open(config_toml, "rb") as f:
            config = tomlkit.load(f)
    else:
        config = tomlkit.document()

    # Add table if it doesn't exist
    config_table = config
    for table_key in expanded:
        if table_key not in config_table:  # type: ignore
            config_table[table_key] = tomlkit.table()  # type: ignore
        config_table = config_table[table_key]  # type: ignore

    # config_table is still referenced in the config doc
    # we can edit the value here and then write the whole doc back
    config_table[key] = value  # type: ignore

    config_toml.parent.mkdir(parents=True, exist_ok=True)
    with open(config_toml, "w") as f:
        tomlkit.dump(config, f)


class AINavigatorConfig(BaseModel):
    port: int = 8001
    api_key: Optional[str] = None


class KuratorConfig(BaseModel):
    domain: str = "kurator.anaconda.com"
    ssl_verify: bool = True
    extra_headers: Optional[Union[Dict[str, str], str]] = None
    run_on: Literal["local"] = "local"
    local_servers_path: Path = Path("~/.ai-navigator/local-servers").expanduser()

    @field_validator("local_servers_path")
    def expand_variables_servers_path(cls, v: Union[Path, str]) -> Path:
        return Path(os.path.expandvars(v)).expanduser()


class Backends(BaseModel):
    ai_navigator: AINavigatorConfig = AINavigatorConfig()
    kurator: KuratorConfig = KuratorConfig()


class AnacondaModelsConfig(AnacondaBaseSettings, plugin_name="models"):
    models_path: Path = Path("~/.ai-navigator/models").expanduser()
    default_backend: Literal["kurator", "ai-navigator"] = "ai-navigator"
    backends: Backends = Backends()

    @field_validator("models_path")
    def expand_variables_models_path(cls, v: Union[Path, str]) -> Path:
        return Path(os.path.expandvars(v)).expanduser()
