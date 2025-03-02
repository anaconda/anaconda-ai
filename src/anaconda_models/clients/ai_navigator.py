from typing import Optional, Any, Dict

from requests import PreparedRequest, Response
from requests_cache import DO_NOT_CACHE
from requests.auth import AuthBase

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
from ..exceptions import APIKeyMissing


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
    def list(self) -> list[Server]:
        res = self._client.get("api/servers")
        res.raise_for_status()
        servers = []
        for s in res.json():
            if "id" not in s:
                continue
            server = Server(**s)
            server._client = self._client
            servers.append(server)
        return servers

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

    def _status(self, server_id: str) -> str:
        servers: list[dict] = self._client.get("api/servers").json()
        matched = [s for s in servers if s["id"] == server_id]
        if not matched:
            raise RuntimeError(f"{server_id} not found")
        return matched[0]["status"]

    def _stop(self, server_id: str) -> None:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "stop"})
        res.raise_for_status()

    def _delete(self, server_id: str) -> None:
        res = self._client.delete(f"api/servers/{server_id}")
        res.raise_for_status()


class AINavigatorAPIKey(AuthBase):
    def __init__(self, config: AnacondaModelsConfig) -> None:
        self.config = config
        super().__init__()

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        api_key = self.config.backends.ai_navigator.get_config("aiNavApiKey")
        if api_key is None:
            raise APIKeyMissing(
                "The aiNavApiKey was not found. Try upgrading and restarting AI Navigator"
            )

        r.headers["Authorization"] = f"Bearer {api_key}"
        return r


class AINavigatorClient(GenericClient):
    _user_agent = f"anaconda-models/{version}"
    auth: AuthBase

    def __init__(self, port: Optional[int] = None):
        kwargs: Dict[str, Any] = {"ai_navigator": {}}
        if port is not None:
            kwargs["ai_navigator"]["port"] = port

        self._config = AnacondaModelsConfig(**kwargs)

        domain = f"localhost:{self._config.backends.ai_navigator.port}"

        super().__init__(domain=domain, ssl_verify=False, backend="memory")
        # Cache Settings
        # The cache is disabled by default, but can be enabled as needed by request
        # for a client session. New Client objects have an empty cache
        self.cache.clear()  # this is likely redundant for backend=memory, but safe
        self.expire_after = DO_NOT_CACHE

        self._base_uri = f"http://{domain}"

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)
        self.auth = AINavigatorAPIKey(self._config)

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
                "Could not connect to AI Navigator. It may not be running."
            )

        return response
