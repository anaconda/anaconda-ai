import re
from pathlib import Path
from typing import Any
from typing_extensions import Self
from typing import cast
from urllib.parse import urljoin
from uuid import UUID

import openai
from pydantic import BaseModel, computed_field, field_validator, Field
from pydantic.types import UUID4
from requests_cache import CacheMixin
from rich.status import Status
from rich.console import Console

from anaconda_cloud_auth.client import BaseClient
from anaconda_models.config import AnacondaModelsConfig
from anaconda_models.exceptions import ModelNotFound, QuantizedFileNotFound
from anaconda_models.utils import find_free_port

MODEL_NAME = re.compile(
    r"^"
    r"(?:(?P<author>[^/]+)[/])??"
    r"(?P<model>[^/]+?)"
    r"(?:(?:[_/])(?P<quantization>Q4_K_M|Q5_K_M|Q6_K|Q8_0)(?:[.](?P<format>gguf))?)?"
    r"$",
    flags=re.IGNORECASE,
)


class ModelQuantization(BaseModel):
    id: str = Field(alias="sha256checksum")
    modelFileName: str = Field(alias="name")
    method: str = Field(alias="quantization")
    sizeBytes: int
    maxRamUsage: int
    isDownloaded: bool = False
    localPath: str | None = None


class ModelMetadata(BaseModel):
    numParameters: int
    contextWindowSize: int
    trainedFor: str
    description: str
    files: list[ModelQuantization]

    @field_validator("files", mode="after")
    @classmethod
    def sort_quantizations(
        cls, value: list[ModelQuantization]
    ) -> list[ModelQuantization]:
        return sorted(value, key=lambda q: q.method)


class ModelSummary(BaseModel):
    id: str
    name: str
    metadata: ModelMetadata

    def get_quantization(self, method: str) -> ModelQuantization:
        for quant in self.metadata.files:
            if quant.method.lower() == method.lower():
                return quant
        else:
            raise QuantizedFileNotFound(
                f"Quantization {method} not found for {self.name}."
            )


class GenericClient(CacheMixin, BaseClient):
    models: "BaseModels"
    servers: "BaseServers"
    _config: AnacondaModelsConfig


class BaseModels(BaseClient):
    def __init__(self, client: GenericClient):
        self._client = client

    def list(self) -> list[ModelSummary]:
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

        return model_info

    def _download(
        self,
        model_summary: ModelSummary,
        quantization: ModelQuantization,
        show_progress: bool = True,
        console: Console | None = None,
    ) -> Path:
        raise NotImplementedError(
            "Downloading models is not available with this client"
        )

    def download(
        self,
        model: str | ModelQuantization,
        force: bool = False,
        show_progress: bool = True,
        console: Console | None = None,
    ) -> None:
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

        if quantization.isDownloaded and not force:
            return

        self._download(
            model_summary=model_info,
            quantization=quantization,
            show_progress=show_progress,
            console=console,
        )


class APIParams(BaseModel, extra="forbid"):
    host: str = "127.0.0.1"
    port: int = 0
    api_key: str | None = None
    log_disable: bool | None = None
    mmproj: str | None = None
    timeout: int | None = None
    verbose: bool | None = None
    n_gpu_layers: int | None = None
    main_gpu: int | None = None
    metrics: bool | None = None


class LoadParams(BaseModel, extra="forbid"):
    batch_size: int | None = None
    cont_batchin: bool | None = None
    ctx_size: int | None = None
    main_gpu: int | None = None
    memory_f32: bool | None = None
    mlock: bool | None = None
    n_gpu_layers: int | None = None
    rope_freq_base: int | None = None
    rope_freq_scale: int | None = None
    seed: int | None = None
    tensor_split: list[int] | None = None
    use_mmap: bool | None = None
    embedding: bool | None = None


class InferParams(BaseModel, extra="forbid"):
    threads: int | None = None
    n_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_last: int | None = None
    repeat_penalty: float | None = None
    temp: float | None = None
    parallel: int | None = None


class ServerConfig(BaseModel):
    modelFileName: Path | str
    apiParams: APIParams = APIParams()
    loadParams: LoadParams = LoadParams()
    inferParams: InferParams = InferParams()
    logsDir: str = "./logs"


class Server(BaseModel):
    id: UUID4
    serverConfig: ServerConfig
    api_key: str | None = "empty"
    _client: GenericClient
    _matched: bool = False

    @property
    def status(self):
        return self._client.servers.status(self.id)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return exc_type is None

    def start(self, show_progress: bool = True, console: Console | None = None) -> None:
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

    @property
    def is_running(self):
        return self.status == "running"

    def stop(self, show_progress: bool = True, console: Console | None = None) -> None:
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

    @computed_field
    @property
    def url(self) -> str:
        return f"http://{self.serverConfig.apiParams.host}:{self.serverConfig.apiParams.port}"

    @computed_field
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


class BaseServers(BaseClient):
    def __init__(self, client: GenericClient):
        self._client = client

    def _get_server_id(self, server: UUID4 | Server | str) -> str:
        if isinstance(server, Server):
            server_id = str(server.id)
        elif isinstance(server, UUID):
            server_id = str(server)
        elif isinstance(server, str):
            server_id = server
        else:
            raise ValueError(f"{server} is not a valid Server identifier")

        return server_id

    def list(self) -> list[Server]:
        raise NotImplementedError

    def match(self, server_config: ServerConfig) -> Server | None:
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
        model: str | ModelQuantization,
        api_params: APIParams | dict[str, Any] | None = None,
        load_params: LoadParams | dict[str, Any] | None = None,
        infer_params: InferParams | dict[str, Any] | None = None,
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

            model = cast(
                ModelQuantization,
                self._client.models.get(model_name).get_quantization(quantization),
            )
        elif isinstance(model, ModelQuantization):
            pass
        else:
            raise ValueError(
                f"model={model} of type {type(model)} is not a supported way to specify a model."
            )

        if not model.isDownloaded:
            if not download_if_needed:
                raise RuntimeError
            # else:
            #     model.download()

        apiParams = api_params if api_params else APIParams()
        loadParams = load_params if load_params else LoadParams()
        inferParams = infer_params if infer_params else InferParams()
        server_config = ServerConfig(
            modelFileName=model.modelFileName,
            apiParams=apiParams,  # type: ignore
            loadParams=loadParams,  # type: ignore
            inferParams=inferParams,  # type: ignore
        )

        if server_config.apiParams.port == 0:
            port = find_free_port()
            server_config.apiParams.port = port

        matched = self.match(server_config)
        if matched is None:
            server = self._create(server_config=server_config)
            server._client = self._client
            return server
        else:
            return matched

    def _start(self, server_id: str) -> None:
        raise NotImplementedError

    def start(self, server: UUID4 | Server | str) -> None:
        server_id = self._get_server_id(server)
        self._start(server_id)

    def _status(self, server_id: str) -> str:
        raise NotImplementedError

    def status(self, server: UUID4 | Server | str) -> str:
        server_id = self._get_server_id(server)
        status = self._status(server_id)
        return status

    def _stop(self, server_id: str) -> None:
        raise NotImplementedError

    def stop(self, server: UUID4 | Server | str) -> None:
        server_id = self._get_server_id(server)
        status = self._stop(server_id)
        return status

    def _delete(self, server_id: str) -> None:
        raise NotImplementedError

    def delete(self, server: UUID4 | Server | str) -> None:
        server_id = self._get_server_id(server)
        status = self._delete(server_id)
        return status
