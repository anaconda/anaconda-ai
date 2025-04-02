import os
from typing import Any, Dict, Optional, Union, List
from uuid import uuid4

import requests
import rich.progress
from rich.console import Console

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
)


class KuratorModels(BaseModels):
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
        assert quantization.localPath
        os.remove(quantization.localPath)


class OllamaServer(BaseServers):
    def ollama_running(self) -> bool:
        response = requests.get("http://localhost:11434")
        return response.ok

    def _create(self, server_config: ServerConfig) -> Server:
        uuid = uuid4()
        server_entry = dict(
            id=uuid,
            serverConfig=server_config,
        )
        server_file = self._client._config.backends.ollama.servers_path / f"{uuid}.json"
        server_file.parent.mkdir(parents=True, exist_ok=True)

        server = Server(**server_entry)  # type: ignore
        with server_file.open("w") as f:
            f.write(server.model_dump_json(indent=2))

        return server


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
        self._config = AnacondaAIConfig(**kwargs_top)

        super().__init__(
            user_agent=user_agent,
            api_key=api_key,
            domain=auth_domain,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )
        self._base_uri = f"https://{self._config.backends.ollama.domain}"

        self.models = KuratorModels(self)
