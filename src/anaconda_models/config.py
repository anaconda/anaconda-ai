import os
from pathlib import Path
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import field_validator

from anaconda_cli_base.config import AnacondaBaseSettings


class AINavigator(BaseModel):
    port: int = 9999


class ModelsConfig(AnacondaBaseSettings, plugin_name="models"):
    cache_path: Path = Path("~/.ai-navigator/models").expanduser()
    domain: str = "kurator.anaconda.com"
    ssl_verify: bool = True
    extra_headers: Optional[Union[Dict[str, str], str]] = None
    ai_navigator: AINavigator = AINavigator()
    run_on: Literal["local", "ai-navigator"] = "local"

    @field_validator("cache_path")
    def expand_variables(cls, v: Union[Path, str]) -> Path:
        return Path(os.path.expandvars(v)).expanduser()


config = ModelsConfig()
