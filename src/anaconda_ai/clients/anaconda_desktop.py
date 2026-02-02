from typing import Optional, Any, Dict

from ..config import AnacondaAIConfig
from .base import GenericClient
from .ai_navigator import (
    AINavigatorModels,
    AINavigatorServers,
    AINavigatorVectorDbServer,
    AiNavigatorVersion,
)


class AnacondaDesktopClient(GenericClient):
    name: str = "anaconda-desktop"

    def __init__(self, app_name: Optional[str] = None, **kwargs: Any) -> None:
        ai_kwargs: Dict[str, Any] = {"backends": {"anaconda_desktop": {}}}
        if app_name is not None:
            ai_kwargs["backends"]["anaconda_desktop"]["app_name"] = app_name

        config = AnacondaAIConfig.model_validate(ai_kwargs)
        domain = f"localhost:{config.backends.anaconda_desktop.port}"
        api_key = config.backends.anaconda_desktop.api_key

        super().__init__(domain=domain, api_key=api_key)
        self._base_uri = f"http://{domain}"

        if app_name is not None:
            self.ai_config.backends.anaconda_desktop.app_name = app_name

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)
        self.vector_db = AINavigatorVectorDbServer(self)

    @property
    def online(self) -> bool:
        res = self.get("/api")
        return res.status_code < 400

    def get_version(self) -> Dict[str, str]:
        res = self.get("api")
        return AiNavigatorVersion(**res.json()["data"]).model_dump()
