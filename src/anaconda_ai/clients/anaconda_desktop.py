from pathlib import Path
from time import time, sleep
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote

from pydantic import Field, computed_field, ConfigDict
from rich.console import Console
import rich.progress

from ..exceptions import ModelDownloadCancelledError
from ..config import AnacondaAIConfig
from .base import (
    Model,
    QuantizedFile,
    BaseModels,
    GenericClient,
    Server,
    BaseServers,
    ServerConfig,
)
from ..utils import find_free_port

DOWNLOAD_START_DELAY = 8


class AnacondaDesktopQuantizedFile(QuantizedFile):
    sha256: str = Field(alias="id")
    size_bytes: int = Field(alias="sizeBytes")
    quant_method: str = Field(alias="quantization")
    max_ram_usage: int = Field(alias="maxRamUsage")
    format: str = "gguf"
    _model: "AnacondaDesktopModel"

    @computed_field
    @property
    def local_path(self) -> Path:
        return (
            AnacondaAIConfig().backends.anaconda_desktop.models_path
            / self._model.id
            / self.identifier
        )

    @property
    def _url(self) -> str:
        model_id = quote(self._model.id, safe="")
        url = f"/api/models/{model_id}/files/{self.sha256}"
        return url

    @property
    def is_downloaded(self) -> bool:
        res = self._model._client.get(self._url)
        res.raise_for_status()
        return res.json()["data"]["isDownloaded"]


class AnacondaDesktopModel(Model):
    id: str
    num_parameters: int = Field(alias="numParameters")
    context_window_size: int = Field(alias="contextWindowSize")
    trained_for: str = Field(alias="trainedFor")
    quantized_files: List[AnacondaDesktopQuantizedFile] = Field(alias="files")


class AnacondaDesktopModels(BaseModels):
    def list(self) -> List[AnacondaDesktopModel]:
        res = self._client.get("api/models")
        res.raise_for_status()

        data = res.json().get("data", [])

        models = []
        for entry in data:
            revised = {"id": entry["id"], "name": entry["name"], **entry["metadata"]}
            model = AnacondaDesktopModel(**revised)
            model._client = self._client
            models.append(model)

        return models

    def _download(
        self,
        model_quantization: AnacondaDesktopQuantizedFile,
        path: Optional[Path] = None,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        size = model_quantization.size_bytes
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
        description = f"Downloading {model_quantization.identifier}"
        task = stream_progress.add_task(
            description=description,
            total=int(size),
            visible=show_progress,
        )

        res = self._client.patch(model_quantization._url, json={"action": "start"})
        res.raise_for_status()
        status = res.json()["data"]
        status_msg = status["status"]
        if status.get("progress", {}).get("paused", False):
            res = self._client.patch(model_quantization._url, json={"action": "resume"})
            res.raise_for_status()
            status = res.json()["data"]
            status_msg = status["status"]

        if status_msg != "in_progress":
            raise RuntimeError(
                f"Cannot initiate download of {model_quantization.identifier}"
            )

        with stream_progress as progress_bar:
            t0 = time()
            res = self._client.get(model_quantization._url)
            res.raise_for_status()
            status = res.json()["data"]
            # Must wait until the download officially
            # starts then we can poll for progress
            elapsed = time() - t0
            while "downloadStatus" not in status and elapsed <= DOWNLOAD_START_DELAY:
                res = self._client.get(model_quantization._url)
                res.raise_for_status()
                status = res.json()["data"]
                elapsed = time() - t0

            while True:
                res = self._client.get(model_quantization._url)
                res.raise_for_status()
                status = res.json()["data"]

                download_status = status.get("downloadStatus", {})
                if download_status.get("status", "") == "in_progress":
                    downloaded = download_status.get("progress", {}).get(
                        "transferredBytes", 0
                    )
                    progress_bar.update(task, completed=downloaded)
                    sleep(0.1)
                else:
                    if not status["isDownloaded"]:
                        raise ModelDownloadCancelledError(
                            "The download process stopped."
                        )
                    else:
                        break

        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            model_quantization.local_path.link_to(path)

    def _delete(self, model_quantization: AnacondaDesktopQuantizedFile) -> None:
        res = self._client.delete(model_quantization._url)
        res.raise_for_status()


class AnacondaDesktopServerConfig(ServerConfig):
    model_name: str = Field(alias="modelFileName")
    id: Optional[str] = None
    server_params: dict[str, Any] = Field(default={}, alias="serverParams")
    load_params: dict[str, Any] = Field(default={}, alias="loadParams")
    infer_params: dict[str, Any] = Field(default={}, alias="inferParams")
    logs_dir: str = Field(default="./logs", alias="logsDir")
    start_server_on_create: bool = Field(default=True, alias="startServerOnCreate")

    model_config = ConfigDict(serialize_by_alias=True)  # , validate_by_alias=True)


class AnacondaDesktopServer(Server):
    serverConfig: AnacondaDesktopServerConfig

    @computed_field
    @property
    def url(self) -> str:
        return f"http://{self.serverConfig.server_params['host']}:{self.serverConfig.server_params['port']}"


class AnacondaDesktopServers(BaseServers):
    def list(self) -> List[AnacondaDesktopServer]:
        res = self._client.get("api/servers")
        res.raise_for_status()
        servers = []
        for s in res.json()["data"]:
            if "id" not in s:
                continue
            server = AnacondaDesktopServer(**s)
            server._client = self._client
            if not server.is_running:
                continue
            servers.append(server)
        return servers

    def match(
        self,
        server_config: AnacondaDesktopServerConfig,
    ) -> Union[AnacondaDesktopServer, None]:
        match_excludes = {
            "id": True,
            "start_server_on_create": True,
            "logs_dir": True,
            "server_params": {"port": True, "host": True},
        }

        config_dump = server_config.model_dump(exclude=match_excludes)
        print(config_dump)
        servers = self.list()
        for server in servers:
            server_dump = server.serverConfig.model_dump(exclude=match_excludes)
            print(server_dump)
            if server.is_running and (config_dump == server_dump):
                server._matched = True
                return server
        else:
            return None

    def _create(
        self,
        model_quantization: AnacondaDesktopQuantizedFile,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> AnacondaDesktopServer:
        server_config = AnacondaDesktopServerConfig(
            modelFileName=model_quantization.identifier
        )

        requested_port = server_config.server_params.get("port", 0)
        if not requested_port:
            port = find_free_port()
            server_config.server_params["port"] = port

        server_config.server_params["host"] = "127.0.0.1"

        if model_quantization._model.trained_for == "sentence-similarity":
            server_config.load_params["embedding"] = True

        matched = self.match(
            server_config,
        )
        if matched is not None:
            return matched

        body = {"serverConfig": server_config.model_dump(exclude={"id"})}

        res = self._client.post("api/servers", json=body)
        res.raise_for_status()
        server = AnacondaDesktopServer(**res.json()["data"])
        return server

    def _status(self, server_id: str) -> str:
        res = self._client.get(f"api/servers/{server_id}")
        res.raise_for_status()
        return res.json()["data"]["status"]

    def _start(self, server_id: str) -> None:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "start"})
        res.raise_for_status()

    def _stop(self, server_id: str) -> None:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "stop"})
        res.raise_for_status()


class AnacondaDesktopClient(GenericClient):
    def __init__(self, site: Optional[str] = None) -> None:
        _ = site  # just ignore it
        self._ai_config = AnacondaAIConfig()
        domain = f"localhost:{self._ai_config.backends.anaconda_desktop.port}"
        super().__init__(
            domain=domain, api_key=self._ai_config.backends.anaconda_desktop.api_key
        )
        self._base_uri = f"http://{domain}"

        self.models = AnacondaDesktopModels(self)
        self.servers = AnacondaDesktopServers(self)
