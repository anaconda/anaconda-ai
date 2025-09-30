from pathlib import Path
from typing import Any, Dict, Optional, Union, List, cast

import requests
from pydantic import computed_field
from requests.exceptions import ConnectionError
from rich.console import Console
from urllib.parse import urljoin

from ..config import AnacondaAIConfig
from ..exceptions import AssetNotFound
from .base import (
    GenericClient,
    BaseServers,
    Server,
    ServerConfig,
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
        model_quantization: Union[str, AICatalogQuantizedFile],
        path: Optional[Path] = None,
        force: bool = False,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        super().download(model_quantization, path, force, show_progress, console)

        if isinstance(model_quantization, str):
            model_quantization = cast(
                AICatalogQuantizedFile, self._find_quantization(model_quantization)
            )

        ollama_models_path = AnacondaAIConfig().backends.ollama.models_path
        ollama_model_path = ollama_models_path / f"sha256-{model_quantization.sha256}"
        ollama_model_path.unlink(missing_ok=True)
        model_quantization.local_path.link_to(ollama_model_path)

        self._client.servers.create(model_quantization)

    def _delete(self, model_quantization: AICatalogQuantizedFile) -> None:
        ollama_models_path = AnacondaAIConfig().backends.ollama.models_path
        ollama_model_path = ollama_models_path / f"sha256-{model_quantization.sha256}"

        res = self._ollama_session.delete(
            urljoin(self._ollama_session.base_url, "api/delete"),
            json={"model": model_quantization.identifier},
        )
        if res.status_code > 404:
            res.raise_for_status()

        ollama_model_path.unlink(missing_ok=True)
        model_quantization.local_path.unlink(missing_ok=True)


class OllamaServer(Server):
    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        return AnacondaAIConfig().backends.ollama.ollama_base_url

    def stop(
        self, show_progress: bool = True, console: Optional[Console] = None
    ) -> None:
        return


class OllamaServers(BaseServers):
    always_detach: bool = True

    def __init__(self, client: GenericClient, ollama_session: OllamaSession):
        super().__init__(client)
        self._ollama_session = ollama_session

    def _get_available_ollama_models(self) -> List[str]:
        res = self._ollama_session.get(
            urljoin(self._ollama_session.base_url, "api/tags")
        )
        res.raise_for_status()
        data: List[Dict[str, str]] = res.json()["models"]
        return [model["name"].rsplit(":", maxsplit=1)[0] for model in data]

    def _get_running_ollama_models(self) -> List[str]:
        res = self._ollama_session.get(urljoin(self._ollama_session.base_url, "api/ps"))
        res.raise_for_status()
        data: List[Dict[str, str]] = res.json()["models"]
        return [model["name"].rsplit(":", maxsplit=1)[0] for model in data]

    def list(self) -> List[OllamaServer]:
        active_models = self._get_available_ollama_models()

        servers = []
        for model in active_models:
            try:
                quant = self._client.models._find_quantization(model)
                assert quant.identifier == model
                server_config = OllamaServer(
                    id=quant.identifier,
                    serverConfig=ServerConfig(model_name=quant.identifier),
                )
                server_config._client = self._client
                servers.append(server_config)
            except AssetNotFound:
                continue

        return servers

    def match(self, server_config: ServerConfig) -> Union[Server, None]:
        config_dump = server_config.model_dump()
        print(config_dump)
        servers = self.list()
        for server in servers:
            server_dump = server.serverConfig.model_dump()
            print(server_dump)
            if server.is_running and (config_dump == server_dump):
                server._matched = True
                return server
        else:
            return None

    def _create(
        self,
        model_quantization: AICatalogQuantizedFile,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> OllamaServer:
        body = {
            "model": model_quantization.identifier,
            "files": {
                model_quantization.identifier: f"sha256:{model_quantization.sha256}"
            },
        }
        url = urljoin(self._ollama_session.base_url, "api/create")
        res = self._ollama_session.post(url, json=body)
        res.raise_for_status()

        server_entry = OllamaServer(
            id=model_quantization.identifier,
            serverConfig=ServerConfig(model_name=model_quantization.identifier),
        )
        server_entry._client = self._client

        return server_entry

    def _status(self, server_id: str) -> str:
        running_models = self._get_available_ollama_models()
        if server_id in running_models:
            return "running"
        else:
            return "stopped"

    def _start(self, _: str) -> None:
        pass

    def _stop(self, server_id: str) -> None:
        return
        # body = {"model": server_id, "keep_alive": 0}
        # res = self._ollama_session.post(
        #     urljoin(self._ollama_session.base_url, "api/generate"), json=body
        # )
        # res.raise_for_status()

    def _delete(self, server_id: str) -> None:
        self._stop(server_id)


class OllamaClient(AICatalogClient):
    def __init__(
        self,
        site: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        models_domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        super().__init__(
            site=site,
            domain=models_domain,
            api_key=api_key,
            user_agent=user_agent,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )

        ollama_session = OllamaSession(ollama_base_url)
        self.models = AICatalogModels(self, ollama_session)
        self.servers = OllamaServers(self, ollama_session)
