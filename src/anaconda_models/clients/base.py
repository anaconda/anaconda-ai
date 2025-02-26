import re
import datetime as dt
from pathlib import Path
from typing import Any
from typing import cast
from urllib.parse import urljoin
from uuid import UUID

import openai
from pydantic import BaseModel, computed_field, Field
from pydantic.types import UUID4
from requests_cache import CacheMixin

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
    modelFileName: str
    method: str
    sizeBytes: int
    maxRamUsage: int
    isDownloaded: bool | None = None


class ModelMetadata(BaseModel):
    numParameters: str
    trainedFor: str
    description: str
    quantizations: list[ModelQuantization]


class ModelSummary(BaseModel):
    id: str
    name: str
    metadata: ModelMetadata

    def get_quantization(self, method: str) -> ModelQuantization:
        for quant in self.metadata.quantizations:
            if quant.method.lower() == method.lower():
                return quant
        else:
            raise QuantizedFileNotFound(
                f"Quantization {method} not found for {self.name}."
            )


class GenericClient(BaseClient, CacheMixin):
    models: "BaseModels"
    servers: "BaseServers"
    _config: AnacondaModelsConfig


class BaseModels(BaseClient):
    def __init__(self, client: GenericClient):
        self._client = client

    def list(self, downloaded_only: bool = False) -> list[ModelSummary]:
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

    def get_quantized_model(self, model: str) -> ModelQuantization:
        match = MODEL_NAME.match(model)
        if match is None:
            raise ValueError(f"{model} does not look like a model name.")

        _, model_name, quant_method, _ = match.groups()

        if quant_method is None:
            raise ValueError(
                "You must include the quantization method in the model as <model>/<quantization>"
            )

        model_info = self.get(model_name)
        quantfile = model_info.get_quantization(quant_method)
        return quantfile

    def _download(self, model: ModelQuantization, show_progress: bool = True) -> Path:
        raise NotImplementedError

    def download(
        self,
        model: str | ModelQuantization,
        force: bool = False,
        show_progress: bool = True,
    ) -> None:
        if isinstance(model, str):
            model = self.get_quantized_model(model)

        if model.isDownloaded and not force:
            return

        _ = self._download(model=model, show_progress=show_progress)


class APIParams(BaseModel):
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


class LoadParams(BaseModel):
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


class InferParams(BaseModel):
    threads: int | None = None
    n_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_last: int | None = None
    repeat_penalty: float | None = None
    temp: float | None = Field(default=None, alias="temperature")
    parallel: int | None = None


class ServerConfig(BaseModel):
    modelFileName: Path | str
    apiParams: APIParams = APIParams()
    loadParams: LoadParams = LoadParams()
    inferParams: InferParams = InferParams()
    logsDir: str = "./logs"


class ServerStatus(BaseModel):
    id: UUID4
    host: str | None = None
    port: int | None = None
    status: str
    startedAt: dt.datetime | None = None
    stoppedAt: dt.datetime | None = None


class Server(BaseModel):
    id: UUID4
    createdAt: dt.datetime
    startImmediately: bool
    serverConfig: ServerConfig
    api_key: str = "empty"

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

    def _create(
        self, server_config: ServerConfig, start_immediately: bool = False
    ) -> Server:
        raise NotImplementedError

    def create(
        self,
        model: str | ModelQuantization,
        start_immediately: bool = True,
        api_params: APIParams | dict[str, Any] | None = None,
        load_params: LoadParams | dict[str, Any] | None = None,
        infer_params: InferParams | dict[str, Any] | None = None,
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

        server = self._create(
            server_config=server_config, start_immediately=start_immediately
        )
        return server

    def _start(self, server_id: str) -> ServerStatus:
        raise NotImplementedError

    def start(self, server: UUID4 | Server | str) -> ServerStatus:
        server_id = self._get_server_id(server)
        status = self._start(server_id)
        return status

    def _status(self, server_id: str) -> ServerStatus:
        raise NotImplementedError

    def status(self, server: UUID4 | Server | str) -> ServerStatus:
        server_id = self._get_server_id(server)
        status = self._status(server_id)
        return status

    def _stop(self, server_id: str) -> ServerStatus:
        raise NotImplementedError

    def stop(self, server: UUID4 | Server | str) -> ServerStatus:
        server_id = self._get_server_id(server)
        status = self._stop(server_id)
        return status

    def _delete(self, server_id: str) -> ServerStatus:
        raise NotImplementedError

    def delete(self, server: UUID4 | Server | str) -> ServerStatus:
        server_id = self._get_server_id(server)
        status = self._delete(server_id)
        return status
