from time import sleep
from typing import Optional, Any, Dict

from requests import PreparedRequest, Response
from requests.auth import AuthBase
from requests.exceptions import ConnectionError
from rich.console import Console
import rich.progress
from urllib.parse import quote

from .. import __version__ as version
from ..config import AnacondaModelsConfig
from .base import (
    GenericClient,
    ModelSummary,
    ModelQuantization,
    BaseModels,
    BaseServers,
    ServerConfig,
    Server,
)


class AINavigatorModels(BaseModels):
    def list(self) -> list[ModelSummary]:
        res = self._client.get("api/models")
        res.raise_for_status()
        model_catalog = res.json()["data"]

        models = []
        for model in model_catalog:
            quoted = quote(model["id"], safe="")
            res = self._client.get(f"api/models/{quoted}/files")
            res.raise_for_status()
            files = res.json()["data"]
            model["metadata"]["files"] = files

            model_summary = ModelSummary(**model)
            models.append(model_summary)
        return models

    def _download(
        self,
        model_summary: ModelSummary,
        quantization: ModelQuantization,
        show_progress: bool = True,
        console: Console | None = None,
    ) -> None:
        model_id = quote(model_summary.id, safe="")
        url = f"api/models/{model_id}/files/{quantization.id}"

        size = quantization.sizeBytes
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
            total=int(size),
            visible=show_progress,
        )

        res = self._client.patch(url, json={"action": "start"})
        res.raise_for_status()
        status = res.json()["data"]
        status_msg = status["status"]
        if status.get("progress", {}).get("paused", False):
            res = self._client.patch(url, json={"action": "resume"})
            res.raise_for_status()
            status = res.json()["data"]
            status_msg = status["status"]

        if status_msg != "in_progress":
            raise RuntimeError(
                f"Cannot initiate download of {quantization.modelFileName}"
            )

        with stream_progress as progress_bar:
            res = self._client.get(url)
            res.raise_for_status()
            status = res.json()["data"]
            # Must wait until the download officially
            # starts then we can poll for progress
            while "downloadStatus" not in status:
                res = self._client.get(url)
                res.raise_for_status()
                status = res.json()["data"]

            while True:
                res = self._client.get(url)
                res.raise_for_status()
                status = res.json()["data"]
                if status["isDownloaded"]:
                    break

                download_status = status.get("downloadStatus", {})
                if download_status.get("status", "") == "in_progress":
                    downloaded = download_status.get("progress", {}).get(
                        "transferredBytes", 0
                    )
                    progress_bar.update(task, completed=downloaded)
                    sleep(0.1)
                else:
                    break


class AINavigatorServers(BaseServers):
    def list(self) -> list[Server]:
        res = self._client.get("api/servers")
        res.raise_for_status()
        servers = []
        for s in res.json()["data"]:
            if "id" not in s:
                continue
            server = Server(**s)
            server._client = self._client
            servers.append(server)
        return servers

    def _create(
        self,
        server_config: ServerConfig,
    ) -> Server:
        body = {
            "serverConfig": server_config.model_dump(exclude={"id"}),
        }

        res = self._client.post("api/servers", json=body)
        res.raise_for_status()
        server = Server(**res.json()["data"])
        return server

    def _start(self, server_id: str) -> None:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "start"})
        res.raise_for_status()

    def _status(self, server_id: str) -> str:
        res = self._client.get(f"api/servers/{server_id}")
        res.raise_for_status()
        status = res.json()["data"]["status"]
        return status

    def _stop(self, server_id: str) -> None:
        res = self._client.patch(f"api/servers/{server_id}", json={"action": "stop"})
        if not res.ok:
            if (
                res.status_code == 400
                and res.json().get("error", {}).get("code", "") == "SERVER_NOT_RUNNING"
            ):
                return
            else:
                res.raise_for_status()

    def _delete(self, server_id: str) -> None:
        res = self._client.delete(f"api/servers/{server_id}")
        res.raise_for_status()


class AINavigatorAPIKey(AuthBase):
    def __init__(self, config: AnacondaModelsConfig) -> None:
        self.config = config
        super().__init__()

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        api_key = self.config.backends.ai_navigator.api_key
        r.headers["Authorization"] = f"Bearer {api_key}"
        return r


class AINavigatorClient(GenericClient):
    _user_agent = f"anaconda-models/{version}"
    auth: AuthBase

    def __init__(self, port: Optional[int] = None, app_name: Optional[str] = None):
        kwargs: Dict[str, Any] = {"backends": {"ai_navigator": {}}}
        if port is not None:
            kwargs["backends"]["ai_navigator"]["port"] = port
        if app_name is not None:
            kwargs["backends"]["ai_navigator"]["app_name"] = app_name

        self._config = AnacondaModelsConfig(**kwargs)

        domain = f"localhost:{self._config.backends.ai_navigator.port}"

        super().__init__(domain=domain, ssl_verify=False)

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
