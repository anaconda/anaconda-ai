import os
from pathlib import Path
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import field_validator

from anaconda_cli_base.config import AnacondaBaseSettings


class AINavigatorConfig(BaseModel):
    port: int = 8001


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
