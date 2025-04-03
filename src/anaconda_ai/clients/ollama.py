import json
import os
from typing import Any, Dict, Optional, Union, List
from uuid import uuid4

import requests
import rich.progress
from requests.exceptions import ConnectionError
from rich.console import Console
from urllib.parse import urljoin

from .. import __version__ as version
from ..exceptions import QuantizedFileNotFound
from ..config import AnacondaAIConfig
from .base import (
    GenericClient,
    BaseModels,
    ModelSummary,
    ModelQuantization,
    BaseServers,
    Server,
    ServerConfig,
    MODEL_NAME,
)

OLLAMA_URL = "http://localhost:11434"


class OllamaSession(requests.Session):
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


class KuratorModels(BaseModels):
    def __init__(self, client: GenericClient):
        super().__init__(client)
        self._ollama_session = OllamaSession()

    def list(self) -> List[ModelSummary]:
        response = self._client.get("/api/models")
        response.raise_for_status()
        data = response.json()["result"]["data"]

        models_path = AnacondaAIConfig().backends.ollama.models_path

        models: List[ModelSummary] = []
        for model in data:
            summary = dict(
                id=model["id"],
                name=model["name"],
                metadata=dict(
                    numParameters=model["numParameters"],
                    contextWindowSize=model["contextWindowSize"],
                    trainedFor=model["trainedFor"],
                    description=model["description"],
                    files=[],
                ),
            )
            for quant in model["quantizedFiles"]:
                filename = (
                    f"{model['name']}_{quant['quantMethod']}.{quant['format'].lower()}"
                )
                path = models_path / f"sha256-{quant['sha256']}"
                file = dict(
                    sha256checksum=quant["sha256"],
                    name=filename,
                    quantization=quant["quantMethod"],
                    sizeBytes=quant["sizeBytes"],
                    maxRamUsage=quant["maxRamUsage"],
                    localPath=path,
                )

                if path.exists() and (os.stat(path).st_size == quant["sizeBytes"]):
                    file["isDownloaded"] = True

                summary["metadata"]["files"].append(file)

            models.append(ModelSummary(**summary))
        return models

    def _download(
        self,
        model_summary: ModelSummary,
        quantization: ModelQuantization,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        response = self._client.get(f"api/models/{model_summary.id}")
        response.raise_for_status()
        for quant in response.json()["quantizedFiles"]:
            if quant["quantMethod"].lower() == quantization.method.lower():
                download_url = quant["downloadUrl"]
                break
        else:
            raise QuantizedFileNotFound(quantization.modelFileName)

        response = self._client.get(download_url, stream=True)
        response.raise_for_status()

        console = Console() if console is None else console
        stream_progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
            rich.progress.TimeRemainingColumn(elapsed_when_finished=True),
            console=console,
            refresh_per_second=10,
        )
        description = f"Downloading {quantization.modelFileName}"
        task = stream_progress.add_task(
            description=description,
            total=int(quantization.sizeBytes),
            visible=show_progress,
        )

        assert quantization.localPath
        quantization.localPath.parent.mkdir(parents=True, exist_ok=True)
        with open(quantization.localPath, "wb") as f:
            with stream_progress as s:
                for chunk in response.iter_content(1024**2):
                    f.write(chunk)
                    s.update(task, advance=len(chunk))

    def _delete(self, _: ModelSummary, quantization: ModelQuantization) -> None:
        model = f"anaconda/{quantization.modelFileName}"
        res = self._ollama_session.delete(
            urljoin(OLLAMA_URL, "api/delete"), json={"model": model}
        )
        if res.status_code == 404 and quantization.localPath:
            os.remove(quantization.localPath)
            return
        res.raise_for_status()


class OllamaServers(BaseServers):
    def __init__(self, client: GenericClient):
        super().__init__(client)
        self._ollama_session = OllamaSession()

    def list(self) -> List[Server]:
        config = AnacondaAIConfig()

        saved_server_configs: Dict[str, Server] = {}
        for fn in config.backends.ollama.servers_path.glob("*.json"):
            with fn.open() as f:
                server = Server(**json.load(f))
                server._client = self._client
                saved_server_configs[server.serverConfig.modelFileName] = server

        res = self._ollama_session.get(urljoin(OLLAMA_URL, "api/tags"))
        res.raise_for_status()
        data = res.json()["models"]

        servers = []
        for server in data:
            model = server["name"].rsplit(":", maxsplit=1)[0]
            if model not in saved_server_configs:
                continue

            matched = saved_server_configs[model]
            matched.serverConfig.modelFileName = (
                matched.serverConfig.modelFileName.replace("anaconda/", "")
            )
            servers.append(saved_server_configs[model])
        return servers

    def _create(self, server_config: ServerConfig) -> Server:
        match = MODEL_NAME.match(server_config.modelFileName)
        if match is None:
            raise ValueError(
                f"{server_config.modelFileName} is not a valid model quantization"
            )

        _, model, quantization, _ = match.groups()
        quant = self._client.models.get(model).get_quantization(quantization)

        body = {
            "model": f"anaconda/{server_config.modelFileName}",
            "files": {server_config.modelFileName: f"sha256:{quant.id}"},
        }
        url = urljoin(OLLAMA_URL, "api/create")
        res = self._ollama_session.post(url, json=body)
        res.raise_for_status()

        uuid = uuid4()
        server_config.modelFileName = f"anaconda/{server_config.modelFileName}"
        server_config.apiParams.host = "localhost"
        server_config.apiParams.port = 11434
        server_entry = Server(id=uuid, serverConfig=server_config, _client=self._client)
        config = AnacondaAIConfig()
        config.backends.ollama.servers_path.mkdir(parents=True, exist_ok=True)
        with open(config.backends.ollama.servers_path / f"{uuid}.json", "w") as f:
            f.write(server_entry.model_dump_json(indent=2))

        return server_entry

    def _status(self, _: str) -> str:
        return "running"

    def _start(self, _: str) -> None:
        pass

    def _stop(self, server_id: str) -> None:
        config = AnacondaAIConfig()
        server_config = config.backends.ollama.servers_path / f"{server_id}.json"
        os.remove(server_config)

    def _delete(self, _: str) -> None:
        pass


class OllamaClient(GenericClient):
    _user_agent = f"anaconda-ai/{version}"

    def __init__(
        self,
        domain: Optional[str] = None,
        auth_domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if domain is not None:
            kwargs["ollama"]["domain"] = domain

        kwargs_top = {"backends": {"ollama": kwargs}}
        self._config = AnacondaAIConfig(**kwargs_top)  # type: ignore

        super().__init__(
            user_agent=user_agent,
            api_key=api_key,
            domain=auth_domain,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )
        self._base_uri = f"https://{self._config.backends.ollama.domain}"

        self.models = KuratorModels(self)
        self.servers = OllamaServers(self)
