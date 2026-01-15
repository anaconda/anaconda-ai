from pathlib import Path
from time import time, sleep
from typing import Dict, List, Optional, Any, Union, Generator, Sequence, Set
from typing_extensions import Self
from urllib.parse import quote

from pydantic import Field, computed_field, ConfigDict, model_validator, BaseModel
from rich.console import Console
from rich.status import Status

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
    VectorDbServerResponse,
    BaseVectorDb,
    VectorDbTableSchema,
    TableInfo,
)
from ..utils import find_free_port

DOWNLOAD_START_DELAY = 8


class AINavigatorQuantizedFile(QuantizedFile):
    sha256: str = Field(alias="id")
    size_bytes: int = Field(alias="sizeBytes")
    quant_method: str = Field(alias="quantization")
    max_ram_usage: int = Field(alias="maxRamUsage")
    format: str = "gguf"
    _model: "AINavigatorModel"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def local_path(self) -> Path:
        return (
            AnacondaAIConfig().backends.ai_navigator.models_path
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


class AINavigatorModel(Model):
    id: str
    num_parameters: int = Field(alias="numParameters")
    context_window_size: int = Field(alias="contextWindowSize")
    trained_for: str = Field(alias="trainedFor")
    quantized_files: Sequence[AINavigatorQuantizedFile] = Field(alias="files")

    def __init__(self, client: GenericClient, **data: Any) -> None:
        super().__init__(client, **data)


class AINavigatorModels(BaseModels):
    def list(self) -> Sequence[AINavigatorModel]:
        res = self.client.get("api/models")
        res.raise_for_status()

        data = res.json().get("data", [])

        models = []
        for entry in data:
            revised = {"id": entry["id"], "name": entry["name"], **entry["metadata"]}
            model = AINavigatorModel(client=self.client, **revised)
            models.append(model)

        return models

    def _download(
        self,
        model_quantization: AINavigatorQuantizedFile,  # type: ignore
        path: Optional[Path] = None,
    ) -> Generator[int, None, None]:
        res = self.client.patch(model_quantization._url, json={"action": "start"})
        res.raise_for_status()
        status = res.json()["data"]
        status_msg = status["status"]
        if status.get("progress", {}).get("paused", False):
            res = self.client.patch(model_quantization._url, json={"action": "resume"})
            res.raise_for_status()
            status = res.json()["data"]
            status_msg = status["status"]

        if status_msg != "in_progress":
            raise RuntimeError(
                f"Cannot initiate download of {model_quantization.identifier}"
            )

        t0 = time()
        res = self.client.get(model_quantization._url)
        res.raise_for_status()
        status = res.json()["data"]
        # Must wait until the download officially
        # starts then we can poll for progress
        elapsed = time() - t0
        while "downloadStatus" not in status and elapsed <= DOWNLOAD_START_DELAY:
            res = self.client.get(model_quantization._url)
            res.raise_for_status()
            status = res.json()["data"]
            elapsed = time() - t0

        while True:
            res = self.client.get(model_quantization._url)
            res.raise_for_status()
            status = res.json()["data"]

            download_status = status.get("downloadStatus", {})
            if download_status.get("status", "") == "in_progress":
                downloaded = download_status.get("progress", {}).get(
                    "transferredBytes", 0
                )
                yield downloaded
                sleep(0.1)
            else:
                if not status["isDownloaded"]:
                    raise ModelDownloadCancelledError("The download process stopped.")
                else:
                    break

        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            path.hardlink_to(model_quantization.local_path)

    def _delete(self, model_quantization: AINavigatorQuantizedFile) -> None:  # type: ignore
        res = self.client.delete(model_quantization._url)
        res.raise_for_status()


class AINavigatorServerParams(BaseModel, extra="allow"):
    host: Optional[str] = None
    port: Optional[int] = None
    mmproj: Optional[str] = None
    timeout: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    main_gpu: Optional[int] = None
    metrics: Optional[bool] = None
    batch_size: Optional[int] = None
    jinja: Optional[bool] = None
    cont_batching: Optional[bool] = None
    ctx_size: Optional[int] = None
    memory_f32: Optional[bool] = None
    mlock: Optional[bool] = None
    rope_freq_base: Optional[int] = None
    rope_freq_scale: Optional[int] = None
    seed: Optional[int] = None
    tensor_split: Optional[List[Union[int, float]]] = None
    use_mmap: Optional[bool] = None
    embedding: Optional[bool] = None
    threads: Optional[int] = None
    n_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repeat_last: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temp: Optional[float] = None
    parallel: Optional[int] = None


class AINavigatorServerConfig(ServerConfig):
    model_name: str = Field(alias="modelFileName")
    id: Optional[str] = None
    api_params: AINavigatorServerParams = Field(
        default_factory=AINavigatorServerParams, alias="apiParams"
    )
    load_params: AINavigatorServerParams = Field(
        default_factory=AINavigatorServerParams, alias="loadParams"
    )
    infer_params: AINavigatorServerParams = Field(
        default_factory=AINavigatorServerParams, alias="inferParams"
    )
    logs_dir: str = Field(default="./logs", alias="logsDir")
    start_server_on_create: bool = Field(default=True, alias="startServerOnCreate")

    model_config = ConfigDict(serialize_by_alias=True)

    _params_dump: Set[str] = {"api_params", "load_params", "infer_params"}


class AINavigatorServer(Server):
    config: AINavigatorServerConfig = Field(alias="serverConfig")
    url: str = ""

    def __init__(self, client: GenericClient, **data: Any) -> None:
        super().__init__(client, **data)

    @model_validator(mode="after")
    def generate_url(self) -> Self:
        if not self.url:
            host = self.config.api_params.host
            port = self.config.api_params.port
            if host and port:
                self.url = f"http://{host}:{port}"
        return self


class AINavigatorServers(BaseServers):
    def list(self) -> Sequence[AINavigatorServer]:
        res = self.client.get("api/servers")
        res.raise_for_status()
        servers = []
        for s in res.json()["data"]:
            if "id" not in s:
                continue
            server = AINavigatorServer(**s, client=self.client)
            server._client = self.client
            if not server.is_running:
                continue
            servers.append(server)
        return servers

    def get(self, server: str) -> AINavigatorServer:
        res = self.client.get(f"api/servers/{server}")
        res.raise_for_status()

        data: dict = res.json()["data"]
        s = AINavigatorServer(**data, client=self.client)
        return s

    def match(
        self,
        server_config: AINavigatorServerConfig,
    ) -> Union[AINavigatorServer, None]:
        match_excludes: Dict[str, Any] = {
            "id": True,
            "start_server_on_create": True,
            "logs_dir": True,
            "api_params": {"port": True, "host": True},
        }

        config_dump = server_config.model_dump(exclude=match_excludes)
        servers = self.list()
        for server in servers:
            server_dump = server.config.model_dump(exclude=match_excludes)
            if server.is_running and (config_dump == server_dump):
                server._matched = True
                return server
        else:
            return None

    def _create(
        self,
        model_quantization: AINavigatorQuantizedFile,  # type: ignore
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> AINavigatorServer:
        server_config = AINavigatorServerConfig(
            modelFileName=model_quantization.identifier,
            loadParams=AINavigatorServerParams(**(extra_options or {})),
        )

        requested_port = server_config.api_params.port or 0
        if not requested_port:
            port = find_free_port()
            server_config.api_params.port = port

        server_config.api_params.host = "127.0.0.1"

        if model_quantization._model.trained_for == "sentence-similarity":
            server_config.load_params.embedding = True

        matched = self.match(
            server_config,
        )
        if matched is not None:
            return matched

        body = {"serverConfig": server_config.model_dump(exclude={"id"})}

        res = self.client.post("api/servers", json=body)
        res.raise_for_status()
        server = AINavigatorServer(**res.json()["data"], client=self.client)
        return server

    def _status(self, server_id: str) -> str:
        res = self.client.get(f"api/servers/{server_id}")
        res.raise_for_status()
        return res.json()["data"]["status"]

    def _start(self, server_id: str) -> None:
        res = self.client.patch(f"api/servers/{server_id}", json={"action": "start"})
        res.raise_for_status()

    def _stop(self, server_id: str) -> None:
        res = self.client.patch(f"api/servers/{server_id}", json={"action": "stop"})
        res.raise_for_status()

    def _delete(self, server_id: str) -> None:
        return


class AINavigatorVectorDbServer(BaseVectorDb):
    def create(
        self,
        show_progress: bool = True,
        leave_running: Optional[bool] = None,  # TODO: Implement this
        console: Optional[Console] = None,
    ) -> VectorDbServerResponse:
        """Create a vector database service.

        Returns:
            dict: The vector database service information.
        """

        text = "Starting pg vector database"
        console = Console() if console is None else console
        console.quiet = not show_progress
        with Status(text, console=console) as display:
            res = self.client.post("api/vector-db")
            text = "pg vector database started"
            display.update(text)

        console.print(f"[bold green]✓[/] {text}", highlight=False)

        data = res.json()["data"]
        vectordb = VectorDbServerResponse(**data)
        return vectordb

    def delete(self) -> None:
        self.client.delete("api/vector-db")

    def stop(self) -> VectorDbServerResponse:
        res = self.client.patch("api/vector-db", json={"running": False})
        return VectorDbServerResponse(**res.json()["data"])

    def get_tables(self) -> list[TableInfo]:
        res = self.client.get("api/vector-db/tables")
        return [TableInfo(**t) for t in res.json()["data"]]

    def drop_table(self, table: str) -> None:
        self.client.delete(f"api/vector-db/tables/{table}")

    def create_table(self, table: str, schema: VectorDbTableSchema) -> None:
        res = self.client.post(
            "api/vector-db/tables", json={"schema": schema.model_dump(), "name": table}
        )
        res.raise_for_status()


class AINavigatorClient(GenericClient):
    def __init__(
        self,
        domain: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._ai_config = AnacondaAIConfig()
        domain = domain or f"localhost:{self._ai_config.backends.ai_navigator.port}"
        api_key = api_key or self._ai_config.backends.ai_navigator.api_key

        super().__init__(domain=domain, api_key=api_key)
        self._base_uri = f"http://{domain}"

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)
        self.vector_db = AINavigatorVectorDbServer(self)

    @property
    def online(self) -> bool:
        res = self.get("/api")
        return res.status_code < 400
