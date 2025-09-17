import atexit
import re
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from requests import Response
from typing_extensions import Self
from urllib.parse import urljoin
from uuid import UUID

import openai
from pydantic import BaseModel, computed_field, field_validator, Field
from pydantic.types import UUID4
from rich.status import Status
from rich.console import Console

from anaconda_cloud_auth.client import BaseClient
from ..config import AnacondaAIConfig
from ..exceptions import (
    AnacondaAIException,
    ModelNotFound,
    QuantizedFileNotFound,
    ModelNotDownloadedError,
)

MODEL_NAME = re.compile(
    r"^"
    r"(?:(?P<author>[^/]+)[/])??"
    r"(?P<model>[^/]+?)"
    r"(?:(?:[_/])(?P<quantization>Q4_K_M|Q5_K_M|Q6_K|Q8_0)(?:[.](?P<format>gguf))?)?"
    r"$",
    flags=re.IGNORECASE,
)

class AiNavigatorVersion(BaseModel):
    name: str
    version: str
    mambaVersion: str
    llamaCppVersion: str

class GenericClient(BaseClient):
    models: "BaseModels"
    servers: "BaseServers"
    vector_db: "BaseVectorDb"
    _config: AnacondaAIConfig

    def get_version(self) -> str:
        raise NotImplementedError


class ModelQuantization(BaseModel):
    id: str = Field(alias="sha256checksum")
    modelFileName: str = Field(alias="name")
    method: str = Field(alias="quantization")
    sizeBytes: int
    maxRamUsage: int
    isDownloaded: bool = False
    localPath: Optional[Path] = None
    _client: GenericClient

    def download(
        self, show_progress: bool = True, console: Optional[Console] = None
    ) -> None:
        self._client.models.download(self, show_progress=show_progress, console=console)

    def delete(self) -> None:
        self._client.models.delete(self)


class TrainedFor(str, Enum):
    sentence_similarity = "sentence-similarity"
    text_generation = "text-generation"


class ModelMetadata(BaseModel):
    numParameters: int
    contextWindowSize: int
    trainedFor: TrainedFor
    description: str
    files: List[ModelQuantization]

    @field_validator("files", mode="after")
    @classmethod
    def sort_quantizations(
        cls, value: List[ModelQuantization]
    ) -> List[ModelQuantization]:
        return sorted(value, key=lambda q: q.method)


class ModelSummary(BaseModel):
    id: str
    name: str
    metadata: ModelMetadata
    _client: GenericClient

    def get_quantization(self, method: str) -> ModelQuantization:
        for quant in self.metadata.files:
            if quant.method.lower() == method.lower():
                quant._client = self._client
                return quant
        else:
            raise QuantizedFileNotFound(
                f"Quantization {method} not found for {self.name}."
            )


class BaseModels:
    def __init__(self, client: GenericClient):
        self._client = client

    def list(self) -> List[ModelSummary]:
        raise NotImplementedError

    def get(self, model: str) -> ModelSummary:
        match = MODEL_NAME.match(model)
        if match is None:
            raise ValueError(f"{model} does not look like a model name.")

        _, model_name, _, _ = match.groups()

        models = self.list()
        for entry in models:
            if entry.name.lower() == model_name.lower():
                model_info = entry
                break
            elif entry.id.lower().endswith(model_name.lower()):
                model_info = entry
                break
        else:
            raise ModelNotFound(f"{model} was not found")

        model_info._client = self._client
        return model_info

    def _download(
        self,
        model_summary: ModelSummary,
        quantization: ModelQuantization,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        raise NotImplementedError(
            "Downloading models is not available with this client"
        )

    def _model_quant(
        self, model: Union[str, ModelQuantization]
    ) -> Tuple[ModelSummary, ModelQuantization]:
        if isinstance(model, str):
            match = MODEL_NAME.match(model)
            if match is None:
                raise ValueError(f"{model} does not look like a model name.")

            _, model_name, quant_method, _ = match.groups()

            if quant_method is None:
                raise ValueError(
                    "You must include the quantization method in the model as <model>/<quantization>"
                )

            model_info = self.get(model_name)
            quantization = model_info.get_quantization(quant_method)
        else:
            model_info = self.get(model.modelFileName)
            quantization = model

        return model_info, quantization

    def download(
        self,
        model: Union[str, ModelQuantization],
        force: bool = False,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        model_info, quantization = self._model_quant(model)

        if quantization.isDownloaded and not force:
            return

        if force:
            self.delete(model)

        self._download(
            model_summary=model_info,
            quantization=quantization,
            show_progress=show_progress,
            console=console,
        )

    def _delete(
        self, model_summary: ModelSummary, quantization: ModelQuantization
    ) -> None:
        raise NotImplementedError

    def delete(self, model: Union[str, ModelQuantization]) -> None:
        model_info, quantization = self._model_quant(model)

        self._delete(model_info, quantization)


class APIParams(BaseModel, extra="forbid"):
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    log_disable: Optional[bool] = None
    mmproj: Optional[str] = None
    timeout: Optional[int] = None
    verbose: Optional[bool] = None
    n_gpu_layers: Optional[int] = None
    main_gpu: Optional[int] = None
    metrics: Optional[bool] = None


class LoadParams(BaseModel, extra="forbid"):
    batch_size: Optional[int] = None
    cont_batching: Optional[bool] = None
    ctx_size: Optional[int] = None
    main_gpu: Optional[int] = None
    memory_f32: Optional[bool] = None
    mlock: Optional[bool] = None
    n_gpu_layers: Optional[int] = None
    rope_freq_base: Optional[int] = None
    rope_freq_scale: Optional[int] = None
    seed: Optional[int] = None
    tensor_split: Optional[List[Union[int, float]]] = None
    use_mmap: Optional[bool] = None
    embedding: Optional[bool] = None


class InferParams(BaseModel, extra="forbid"):
    threads: Optional[int] = None
    n_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repeat_last: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temp: Optional[float] = None
    parallel: Optional[int] = None


class ServerConfig(BaseModel):
    modelFileName: str
    apiParams: APIParams = APIParams()
    loadParams: LoadParams = LoadParams()
    inferParams: InferParams = InferParams()
    logsDir: str = "./logs"


class Server(BaseModel):
    id: UUID4
    serverConfig: ServerConfig
    api_key: Optional[str] = "empty"
    _client: GenericClient
    _matched: bool = False

    @property
    def status(self) -> str:
        return self._client.servers.status(self.id)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.stop()
        return exc_type is None

    def start(
        self,
        show_progress: bool = True,
        leave_running: Optional[bool] = None,
        console: Optional[Console] = None,
    ) -> None:
        text = f"{self.serverConfig.modelFileName} (creating)"
        console = Console() if console is None else console
        console.quiet = not show_progress
        with Status(text, console=console) as display:
            self._client.servers.start(self)
            status = "starting"
            text = f"{self.serverConfig.modelFileName} ({status})"
            display.update(text)

            while status != "running":
                status = self._client.servers.status(self)
                text = f"{self.serverConfig.modelFileName} ({status})"
                display.update(text)
        console.print(f"[bold green]✓[/] {text}", highlight=False)

        if not self._matched:
            kwargs = {}
            if leave_running is not None:
                kwargs["stop_server_on_exit"] = leave_running

            config = AnacondaAIConfig(**kwargs)  # type: ignore
            if config.stop_server_on_exit:
                atexit.register(self.stop, console=console)

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def stop(
        self, show_progress: bool = True, console: Optional[Console] = None
    ) -> None:
        console = Console() if console is None else console
        console.quiet = not show_progress
        text = f"{self.serverConfig.modelFileName} (stopping)"
        with Status(text, console=console) as display:
            status = "stopping"
            self._client.servers.stop(self.id)
            while status != "stopped":
                status = self._client.servers.status(self)
                text = f"{self.serverConfig.modelFileName} ({status})"
                display.update(text)
        console.print(f"[bold green]✓[/] {text}", highlight=False)

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        return f"http://{self.serverConfig.apiParams.host}:{self.serverConfig.apiParams.port}"

    @computed_field  # type: ignore[misc]
    @property
    def openai_url(self) -> str:
        return urljoin(self.url, "/v1")

    def openai_client(self, **kwargs: Any) -> openai.OpenAI:
        client = openai.OpenAI(base_url=self.openai_url, api_key=self.api_key, **kwargs)
        return client

    def openai_async_client(self, **kwargs: Any) -> openai.AsyncOpenAI:
        client = openai.AsyncOpenAI(
            base_url=self.openai_url, api_key=self.api_key, **kwargs
        )
        return client


class BaseServers:
    def __init__(self, client: GenericClient):
        self._client = client

    def _get_server_id(self, server: Union[UUID4, Server, str]) -> str:
        if isinstance(server, Server):
            server_id = str(server.id)
        elif isinstance(server, UUID):
            server_id = str(server)
        elif isinstance(server, str):
            server_id = server
        else:
            raise ValueError(f"{server} is not a valid Server identifier")

        return server_id

    def list(self) -> List[Server]:
        raise NotImplementedError

    def match(self, server_config: ServerConfig) -> Union[Server, None]:
        exclude = {"apiParams": {"host", "port", "api_key"}}
        servers = self.list()
        for server in servers:
            config_dump = server_config.model_dump(exclude=exclude)
            server_dump = server.serverConfig.model_dump(exclude=exclude)
            if server.is_running and (config_dump == server_dump):
                server._matched = True
                return server
        else:
            return None

    def _create(self, server_config: ServerConfig) -> Server:
        raise NotImplementedError

    def create(
        self,
        model: Union[str, ModelQuantization],
        api_params: Optional[Union[APIParams, Dict[str, Any]]] = None,
        load_params: Optional[Union[LoadParams, Dict[str, Any]]] = None,
        infer_params: Optional[Union[InferParams, Dict[str, Any]]] = None,
        download_if_needed: bool = True,
    ) -> Server:
        if isinstance(model, str):
            match = MODEL_NAME.match(model)
            if match is None:
                raise ValueError(
                    f"{model} does not look like a quantized model name in the format <model>/<quant>"
                )

            _, model_name, quantization, _ = match.groups()
            quantization = quantization.upper()

            if not quantization:
                raise ValueError(
                    "You must provide a quantization level in the model name as <model>/<quant>"
                )

            model_summary = self._client.models.get(model_name)
            model = model_summary.get_quantization(quantization)
        elif isinstance(model, ModelQuantization):
            pass
        else:
            raise ValueError(
                f"model={model} of type {type(model)} is not a supported way to specify a model."
            )

        if not model.isDownloaded:
            if not download_if_needed:
                raise ModelNotDownloadedError(f"{model} has not been downloaded")
            else:
                self._client.models.download(model)

        apiParams = api_params if api_params else APIParams()
        loadParams = load_params if load_params else LoadParams()
        inferParams = infer_params if infer_params else InferParams()

        server_config = ServerConfig(
            modelFileName=model.modelFileName,
            apiParams=apiParams,  # type: ignore
            loadParams=loadParams,  # type: ignore
            inferParams=inferParams,  # type: ignore
        )

        matched = self.match(server_config)
        if matched is None:
            server = self._create(server_config=server_config)
            server._client = self._client
            return server
        else:
            return matched

    def _start(self, server_id: str) -> None:
        raise NotImplementedError

    def start(self, server: Union[UUID4, Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._start(server_id)

    def _status(self, server_id: str) -> str:
        raise NotImplementedError

    def status(self, server: Union[UUID4, Server, str]) -> str:
        server_id = self._get_server_id(server)
        status = self._status(server_id)
        return status

    def _stop(self, server_id: str) -> None:
        raise NotImplementedError

    def stop(self, server: Union[UUID4, Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._stop(server_id)

    def _delete(self, server_id: str) -> None:
        raise NotImplementedError

    def delete(self, server: Union[UUID4, Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._delete(server_id)

class VectorDbServerResponse(BaseModel):
    running: bool
    host: str
    port: int
    database: str
    user: str
    password: str

class VectorDbTableColumn(BaseModel):
    name: str
    type: str
    constraints: Optional[List[str]] = None

class VectorDbTableSchema(BaseModel):
    columns: List[VectorDbTableColumn]

class TableInfo(BaseModel):
    name: str
    table_schema: VectorDbTableSchema = Field(alias="schema")
    numRows: int

class BaseVectorDb:
    def __init__(self, client: GenericClient) -> None:
        self._client = client

    def create(self,
        show_progress: bool = True,
        leave_running: Optional[bool] = None,
        console: Optional[Console] = None,
    ) -> VectorDbServerResponse:
        raise NotImplementedError()
    
    def delete(self) -> None:
        raise NotImplementedError
    
    def stop(self) -> VectorDbServerResponse:
        raise NotImplementedError
    
    def create_table(self, table: str, schema: VectorDbTableSchema) -> None:
        raise NotImplementedError
    
    def get_tables(self) -> List[TableInfo]:
        raise NotImplementedError

    def drop_table(self, table: str) -> None:
        raise NotImplementedError
    

class IncompatibleVersionError(Exception):
    pass
