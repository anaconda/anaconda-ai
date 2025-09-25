import json
from typing import Any, Dict, Optional, Union, List

import requests
from requests.exceptions import ConnectionError
from rich.console import Console
from urllib.parse import urljoin, urlparse

from ..config import AnacondaAIConfig
from .base import (
    GenericClient,
    QuantizedFile,
    BaseServers,
    Server,
    ServerConfig,
    MODEL_NAME,
)
from .ai_catalog import AICatalogModels as _AICatalogModels
from .ai_catalog import AICatalogQuantizedFile
from .ai_catalog import AICatalogClient


class OllamaSession(requests.Session):
    def __init__(self, base_url: Optional[str] = None) -> None:
        super().__init__()

        kwargs = {}
        if base_url is not None:
            kwargs["base_url"] = base_url

        config = AnacondaAIConfig(**{"backends": {"ollama": kwargs}})  # type: ignore
        self.base_url = config.backends.ollama.ollama_base_url

    def request(
        self,
        method: Union[str, bytes],
        url: Union[str, bytes],
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        try:
            response = super().request(method, url, *args, **kwargs)
        except ConnectionError:
            raise RuntimeError("Could not connect to Ollama. It may not be running.")

        return response


class AICatalogModels(_AICatalogModels):
    def __init__(self, client: GenericClient, ollama_session: OllamaSession):
        super().__init__(client)
        self._ollama_session = ollama_session

    def download(
        self,
        model_quantization: Union[str, QuantizedFile],
        force: bool = False,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        super().download(model_quantization, force, show_progress, console)

        if isinstance(model_quantization, str):
            model_quantization = self._find_quantization(model_quantization)

        ollama_models_path = AnacondaAIConfig().backends.ollama.models_path
        ollama_model_path = ollama_models_path / f"sha256-{model_quantization.sha256}"
        if not ollama_model_path.exists():
            model_quantization.local_path.link_to(ollama_model_path)

        self._client.servers.create(model_quantization)

    def _delete(self, model_quantization: AICatalogQuantizedFile) -> None:
        res = self._ollama_session.delete(
            urljoin(self._ollama_session.base_url, "api/delete"),
            json={"model": model_quantization.identifier},
        )
        if res.status_code == 404 and model_quantization.local_path:
            model_quantization.local_path.unlink()
            return
        res.raise_for_status()


class OllamaServers(BaseServers):
    always_detach: bool = True

    def __init__(self, client: GenericClient, ollama_session: OllamaSession):
        super().__init__(client)
        self._ollama_session = ollama_session

    def list(self) -> List[Server]:
        config = AnacondaAIConfig()

        saved_server_configs: Dict[str, Server] = {}
        for fn in config.backends.ollama.servers_path.glob("*.json"):
            with fn.open() as f:
                server = Server(**json.load(f))
                server._client = self._client
                saved_server_configs[server.serverConfig.model_name] = server

        res = self._ollama_session.get(
            urljoin(self._ollama_session.base_url, "api/tags")
        )
        res.raise_for_status()
        data: List[Dict[str, str]] = res.json()["models"]

        servers: List[Server] = []
        for server_entry in data:
            model = server_entry["name"].rsplit(":", maxsplit=1)[0]
            if model not in saved_server_configs:
                continue

            matched = saved_server_configs[model]
            servers.append(matched)
        return servers

    def _create(self, server_config: ServerConfig) -> Server:
        match = MODEL_NAME.match(server_config.model_name)
        if match is None:
            raise ValueError(
                f"{server_config.model_name} is not a valid model quantization"
            )

        _, model, quantization, _ = match.groups()
        quant = self._client.models.get(model).get_quantization(quantization)

        body = {
            "model": server_config.model_name,
            "files": {server_config.model_name: f"sha256:{quant.sha256}"},
        }
        url = urljoin(self._ollama_session.base_url, "api/create")
        res = self._ollama_session.post(url, json=body)
        res.raise_for_status()

        parsed = urlparse(self._ollama_session.base_url)
        server_config.host = parsed.hostname or "localhost"
        server_config.port = parsed.port or 11434
        server_entry = Server(
            id=server_config.model_name,
            serverConfig=server_config,
            _client=self._client,
        )
        config = AnacondaAIConfig()
        config.backends.ollama.servers_path.mkdir(parents=True, exist_ok=True)

        server_file = (
            config.backends.ollama.servers_path / f"{server_config.model_name}.json"
        )
        server_file.write_text(server_entry.model_dump_json(indent=2))

        return server_entry

    def _status(self, server_id: str) -> str:
        config = AnacondaAIConfig()
        server_config = config.backends.ollama.servers_path / f"{server_id}.json"
        if server_config.exists():
            return "running"
        else:
            return "stopped"

    def _start(self, _: str) -> None:
        pass

    def _stop(self, server_id: str) -> None:
        config = AnacondaAIConfig()
        server_config = config.backends.ollama.servers_path / f"{server_id}.json"
        if not server_config.exists():
            return

        server = Server(**json.loads(server_config.read_text()))

        body = {"model": server.serverConfig.model_name, "keep_alive": 0}
        res = self._ollama_session.post(
            urljoin(self._ollama_session.base_url, "api/generate"), json=body
        )
        res.raise_for_status()
        server_config.unlink()

    def _delete(self, server_id: str) -> None:
        self._stop(server_id)


class OllamaClient(AICatalogClient):
    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        models_domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        super().__init__(
            domain=models_domain,
            api_key=api_key,
            user_agent=user_agent,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )

        ollama_session = OllamaSession(ollama_base_url)
        self.models = AICatalogModels(self, ollama_session)
        self.servers = OllamaServers(self, ollama_session)
