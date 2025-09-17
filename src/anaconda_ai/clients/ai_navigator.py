from time import sleep, time
from typing import Optional, Any, Union
from anaconda_cli_base import console
from packaging.version import parse
from requests import PreparedRequest, Response
from requests.auth import AuthBase
from requests.exceptions import ConnectionError
from rich.console import Console
import rich.progress
from rich.status import Status
from urllib.parse import quote

from .. import __version__ as version
from ..config import AnacondaAIConfig
from .base import (
    AiNavigatorVersion,
    BaseVectorDb,
    IncompatibleVersionError,
    VectorDbServerResponse,
    TableInfo,
    GenericClient,
    ModelSummary,
    ModelQuantization,
    BaseModels,
    BaseServers,
    ServerConfig,
    Server,
    VectorDbTableSchema,
)
from ..exceptions import AnacondaAIException
from ..utils import find_free_port

DOWNLOAD_START_DELAY = 8
MIN_AI_NAV_VERSION = "1.14.2"

class ModelDownloadCancelledError(AnacondaAIException): ...


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
        console: Optional[Console] = None,
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
            t0 = time()
            res = self._client.get(url)
            res.raise_for_status()
            status = res.json()["data"]
            # Must wait until the download officially
            # starts then we can poll for progress
            elapsed = time() - t0
            while "downloadStatus" not in status and elapsed <= DOWNLOAD_START_DELAY:
                res = self._client.get(url)
                res.raise_for_status()
                status = res.json()["data"]
                elapsed = time() - t0

            while True:
                res = self._client.get(url)
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

    def _delete(
        self, model_summary: ModelSummary, quantization: ModelQuantization
    ) -> None:
        model_id = quote(model_summary.id, safe="")
        url = f"api/models/{model_id}/files/{quantization.id}"
        res = self._client.delete(url)
        res.raise_for_status()


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
        if not server_config.apiParams.port or server_config.apiParams.port == 0:
            port = find_free_port()
            server_config.apiParams.port = port

        if not server_config.apiParams.host:
            server_config.apiParams.host = "127.0.0.1"

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

class AINavigatorVectorDbServer(BaseVectorDb):

    def create(self,
        show_progress: bool = True,
        leave_running: Optional[bool] = None, # TODO: Implement this
        console: Optional[Console] = None,
        ) -> VectorDbServerResponse:
        """Create a vector database service.
        
        Returns:
            dict: The vector database service information.
        """
        
        text = f"Starting pg vector database"
        console = Console() if console is None else console
        console.quiet = not show_progress
        with Status(text, console=console) as display:
            res = self._client.post("api/vector-db")
            text = "pg vector database started"
            display.update(text)

        console.print(f"[bold green]✓[/] {text}", highlight=False)

        return VectorDbServerResponse(**res.json()["data"])
    
    def delete(self) -> None:
        self._client.delete("api/vector-db")

    def stop(self) -> VectorDbServerResponse:
        res = self._client.patch("api/vector-db", json={"running": False})
        return VectorDbServerResponse(**res.json()["data"])

    def get_tables(self) -> list[TableInfo]:
        res = self._client.get("api/vector-db/tables")
        return [TableInfo(**t) for t in res.json()["data"]]
    
    def drop_table(self, table: str) -> None:
        self._client.delete(f"api/vector-db/tables/{table}")

    
    def create_table(self, table: str, schema: VectorDbTableSchema) -> None:
        res = self._client.post(f"api/vector-db/tables", json={
            "schema": schema.model_dump(),
            "name": table
        })
     

class AINavigatorAPIKey(AuthBase):
    def __init__(self, config: AnacondaAIConfig) -> None:
        self.config = config
        super().__init__()

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        api_key = self.config.backends.ai_navigator.api_key
        r.headers["Authorization"] = f"Bearer {api_key}"
        return r


class AINavigatorClient(GenericClient):
    _user_agent = f"anaconda-ai/{version}"
    auth: AuthBase

    def __init__(self, port: Optional[int] = None, app_name: Optional[str] = None):
        kwargs: dict[str, Any] = {"backends": {"ai_navigator": {}}}
        if port is not None:
            kwargs["backends"]["ai_navigator"]["port"] = port
        if app_name is not None:
            kwargs["backends"]["ai_navigator"]["app_name"] = app_name

        self._config = AnacondaAIConfig(**kwargs)

        domain = f"localhost:{self._config.backends.ai_navigator.port}"

        super().__init__(domain=domain, ssl_verify=False)

        self._base_uri = f"http://{domain}"

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)
        self.vector_db = AINavigatorVectorDbServer(self)
        self.auth = AINavigatorAPIKey(self._config)

    def request(
        self,
        method: Union[str, bytes],
        url: Union[str, bytes],
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        try:
            # to avoid recursive calls to the version check
            if url != "api":
                self.version_check()

            response = super().request(method, url, *args, **kwargs)
            self.raise_for_status(response)

        except ConnectionError:
            raise RuntimeError(
                f"Could not connect to AI Navigator. It may not be running. Please ensure you have at least version {MIN_AI_NAV_VERSION} installed."
            )

        return response
    
    def version_check(self) -> None:
        # ignore version check for ai-navigator-alpha
        if self._config.backends.ai_navigator.app_name == "ai-navigator-alpha":
            return
        
        ai_navigator_versions = self.get_ai_navigator_version()
        if parse(ai_navigator_versions.version) < parse(MIN_AI_NAV_VERSION):
            raise IncompatibleVersionError(f"Version {MIN_AI_NAV_VERSION} of AI Navigator is required, you have version {ai_navigator_versions.version}")   

    def get_ai_navigator_version(self) -> AiNavigatorVersion:
        res = self.get("api")
        return AiNavigatorVersion(**res.json()["data"])
    
    def get_version(self) -> str:
        ai_navigator_versions = self.get_ai_navigator_version()

        warning = ""
        try:
            self.version_check()
        except IncompatibleVersionError as e:
            warning = f"Warning: {e}"

        version_str = f"AI Navigator: {ai_navigator_versions.version}\n"\
                f"Mamba Version: {ai_navigator_versions.mambaVersion}\n"\
                f"LlamaCpp Version: {ai_navigator_versions.llamaCppVersion}\n"\
                f"{warning}"
        return version_str

    def raise_for_status(self, response: Response) -> None:
        if response.ok:
            return
        
        error = None
        try:
            error = response.json()['error']
        except:
            response.raise_for_status()
        
        raise AnacondaAIException(error)