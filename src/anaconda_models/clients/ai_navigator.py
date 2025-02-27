from typing import Optional, Any, Dict

from anaconda_cloud_auth.client import BearerAuth
from anaconda_cloud_auth.token import TokenInfo
from anaconda_cloud_auth.exceptions import TokenNotFoundError
from requests import Response
from rich.prompt import Prompt

from .. import __version__ as version
from ..config import AnacondaModelsConfig
from .base import (
    GenericClient,
    ModelSummary,
    BaseModels,
    BaseServers,
    ServerConfig,
    Server,
    ServerStatus,
)


class AINavigatorModels(BaseModels):
    def list(self) -> list[ModelSummary]:
        res = self._client.get("api/models", expire_after=100)
        res.raise_for_status()
        data = res.json()["data"]
        models = [ModelSummary(**m) for m in data]

        config = AnacondaModelsConfig()
        downloaded = [fn.name for fn in config.models_path.glob("**/*.gguf")]
        for model in models:
            for quant in model.metadata.quantizations:
                if quant.modelFileName in downloaded:
                    quant.isDownloaded = True
        return [
            m for m in models if any(q.isDownloaded for q in m.metadata.quantizations)
        ]


class AINavigatorServers(BaseServers):
    def _create(
        self, server_config: ServerConfig, start_immediately: bool = False
    ) -> Server:
        body = {
            "serverConfig": server_config.model_dump(exclude={"id"}),
            "startImmediately": start_immediately,
        }

        res = self._client.post("api/servers", json=body)
        res.raise_for_status()
        server = Server(**res.json())
        return server

    def _start(self, server_id: str) -> ServerStatus:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "start"})
        res.raise_for_status()
        return ServerStatus(**res.json())

    def _status(self, server_id: str) -> ServerStatus:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "start"})
        res.raise_for_status()
        return ServerStatus(**res.json())

    def _stop(self, server_id: str) -> ServerStatus:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "stop"})
        res.raise_for_status()
        return ServerStatus(**res.json())


class AINavigatorClient(GenericClient):
    _user_agent = f"anaconda-models/{version}"
    auth: BearerAuth

    def __init__(self, port: Optional[int] = None, api_key: Optional[str] = None):
        kwargs: Dict[str, Any] = {"ai_navigator": {}}
        if port is not None:
            kwargs["ai_navigator"]["port"] = port
        if api_key is not None:
            kwargs["ai_navigator"]["api_key"] = api_key

        self._config = AnacondaModelsConfig(**kwargs)

        try:
            token = TokenInfo.load(domain="ai-navigator")
        except TokenNotFoundError:
            token = self.ask_for_api_key()

        domain = f"localhost:{self._config.backends.ai_navigator.port}"

        super().__init__(domain=domain, ssl_verify=False, api_key=token.api_key)

        self._base_uri = f"http://{domain}"

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)

    def ask_for_api_key(self) -> TokenInfo:
        api_key = Prompt.ask("Paste the AI Navigator API key here", password=True)
        token = TokenInfo(domain="ai-navigator", api_key=api_key)
        token.save()
        print("AI Navigator API Key saved in keyring")
        return token

    def request(
        self,
        method: str,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        try:
            response = super().request(method, url, *args, **kwargs)
        except ConnectionError:
            raise RuntimeError(
                "Could not connect to AI Navigator. It may not be running. Or you may have the wrong port configured."
            )

        if response.status_code == 401:
            token = self.ask_for_api_key()
            self.auth.api_key = token.api_key
            response = super().request(method, url, *args, **kwargs)

        return response
