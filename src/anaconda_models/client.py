import json
import os
import re
import datetime as dt
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import cast
from typing_extensions import Self
from uuid import UUID, uuid4
from urllib.parse import urljoin

import openai
from intake.readers.readers import LlamaServerReader
from intake.readers.datatypes import LlamaCPPService
from pydantic import BaseModel, HttpUrl, computed_field, Field, model_validator
from pydantic.types import UUID4
from requests import Response
from requests_cache import CacheMixin, DO_NOT_CACHE
from requests.exceptions import ConnectionError

from anaconda_cloud_auth.client import BaseClient
from anaconda_cloud_auth.client import BearerAuth
from anaconda_cloud_auth.config import AnacondaCloudConfig
from anaconda_models import __version__ as version
from anaconda_models.config import AnacondaModelsConfig
from anaconda_models.exceptions import ModelNotFound, APIKeyMissing
from anaconda_models.utils import find_free_port
from anaconda_cli_base.config import anaconda_config_path

MODEL_NAME = re.compile(
    r"^"
    r"(?:(?P<author>[^/]+)[/])??"
    r"(?P<model>[^/]+?)"
    r"(?:(?:[_/])(?P<quantization>Q4_K_M|Q5_K_M|Q6_K|Q8_0)(?:[.](?P<format>gguf))?)?"
    r"$",
    flags=re.IGNORECASE,
)


class Benchmark(BaseModel):
    name: str
    value: float | int


class Evaluation(BaseModel):
    name: str
    value: float | int


class QuantizedFile(BaseModel):
    benchmarks: list[Benchmark]
    downloadUrl: HttpUrl
    evaluations: list[Evaluation]
    format: str
    generatedOn: dt.datetime
    id: str
    maxRamUsage: int
    quantEngine: str
    quantMethod: str
    sha256: str
    sizeBytes: int
    path: Path | None = None

    @property
    def is_downloaded(self) -> bool:
        assert self.path
        if not self.path.exists():
            return False

        return self.sizeBytes == os.stat(self.path).st_size


class Source(BaseModel):
    id: str
    name: str
    url: HttpUrl


class Group(BaseModel):
    id: int
    name: str


class ConvertedFile(BaseModel):
    conversionEngine: str
    downloadUrl: HttpUrl
    evaluations: list[Evaluation]
    format: str
    generatedOn: dt.datetime
    maxRamUsage: int
    sha256: str
    sizeBytes: int


class Model(BaseModel):
    baseModel: str
    baseModels: list[Any]
    contextWindowSize: int
    convertedFiles: list[ConvertedFile]
    datasets: list[Any]
    description: str
    firstPublished: dt.datetime
    groups: list[Group]
    id: str
    infoUrl: HttpUrl | str
    knowledgeCutOff: Any
    languages: list[str]
    libraryName: str
    license: str
    modelId: str
    modelType: str
    name: str
    numParameters: int
    paperUrl: HttpUrl | str
    quantizedFiles: list[QuantizedFile]
    source: Source
    sourceUrl: HttpUrl
    tags: list[str]
    trainedFor: str

    @computed_field
    @property
    def quantizations(self) -> dict[str, QuantizedFile]:
        q = {}
        for quant in self.quantizedFiles:
            q[quant.quantMethod] = quant

        return q

    @model_validator(mode="after")
    def update_quantized_files(self) -> Self:
        for quantization in self.quantizedFiles:
            models_path = AnacondaModelsConfig().models_path
            quantization.path = (
                models_path
                / self.modelId
                / f"{self.name}_{quantization.quantMethod}.gguf"
            )

        self.quantizedFiles = sorted(self.quantizedFiles, key=lambda q: q.quantMethod)

        return self


class Models(BaseClient):
    def __init__(self, client: BaseClient):
        self._client = client

    def list(self) -> list[Model]:
        response = self._client.get("/api/models", expire_after=60)
        response.raise_for_status()
        data = response.json()["result"]["data"]
        models = [Model(**m) for m in data]
        return models

    def get(self, model: str) -> Model | QuantizedFile:
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


class APIParams(BaseModel):
    host: str = "127.0.0.1"
    port: int = 0


class ServerConfig(BaseModel):
    modelFileName: Path
    apiParams: APIParams = APIParams()
    loadParams: dict = Field(default_factory=dict)
    inferParams: dict = Field(default_factory=dict)
    logsDir: str = "./logs"


class ServerStatus(BaseModel):
    id: UUID4
    host: str
    port: int
    status: str
    startedAt: dt.datetime


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
    def __init__(self, client: BaseClient):
        self._client = client

    def list(self) -> list[Server]:
        raise NotImplementedError

    def _create(
        self, server_config: ServerConfig, start_immediately: bool = False
    ) -> Server:
        raise NotImplementedError

    def create(
        self, model: str | QuantizedFile, start_immediately: bool = False
    ) -> Server:
        if isinstance(model, str):
            match = MODEL_NAME.match(model)
            if match is None:
                raise ValueError(f"{model} does not like a quantized model name")

            _, model_name, quantization, _ = match.groups()
            quantization = quantization.upper()

            if not quantization:
                raise ValueError("You must provide a quantization level")

            model = cast(
                QuantizedFile,
                self._client.models.get(model_name).quantizations[quantization],
            )
        elif isinstance(model, QuantizedFile):
            pass
        else:
            raise ValueError(
                f"model={model} of type {type(model)} is not a supported way to specify a model."
            )

        server_config = ServerConfig(modelFileName=model.path)

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
        if isinstance(server, Server):
            server_id = str(server.id)
        elif isinstance(server, UUID):
            server_id = str(server)
        elif isinstance(server, str):
            server_id = server
        else:
            raise ValueError(f"{server}")

        status = self._start(server_id)
        return status

    def delete(self, server: UUID4 | Server | str) -> ServerStatus:
        raise NotImplementedError


class AINavigatorServers(BaseServers):
    def _create(
        self, server_config: ServerConfig, start_immediately: bool = False
    ) -> Server:
        body = {
            "serverConfig": server_config.model_dump(exclude={"id"}),
            "startImmediately": start_immediately,
        }
        body["serverConfig"]["modelFileName"] = body["serverConfig"][
            "modelFileName"
        ].name

        res = self._client.post("/api/servers", json=body)
        res.raise_for_status()
        server = Server(**res.json())
        return server

    def _start(self, server_id: str) -> ServerStatus:
        res = self._client.post(f"/api/servers/{server_id}/start")
        res.raise_for_status()
        return ServerStatus(**res.json())


class LocalServers(BaseServers):
    def _create(
        self, server_config: ServerConfig, start_immediately: bool = False
    ) -> Server:
        uuid = uuid4()
        server_entry = Server(
            id=uuid,
            createdAt=dt.datetime.now(tz=dt.timezone.utc),
            startImmediately=start_immediately,
            serverConfig=server_config,
        )
        server_file = (
            self._client._config.backends.kurator.local_servers_path / f"{uuid}.json"
        )
        server_file.parent.mkdir(parents=True, exist_ok=True)

        with server_file.open("w") as f:
            f.write(server_entry.model_dump_json(indent=2))

        return server_entry

    def _start(self, server_id: str) -> ServerStatus:
        server_file = (
            self._client._config.backends.kurator.local_servers_path
            / f"{server_id}.json"
        )
        with server_file.open("r") as f:
            data = json.load(f)

        server = Server(**data)
        llama_cpp = LlamaServerReader(str(server.serverConfig.modelFileName))

        llama_cpp_kwargs = {
            **server.serverConfig.loadParams,
            **server.serverConfig.inferParams,
            **{
                "port": server.serverConfig.apiParams.port,
                "ctx-size": 0,
            },
        }
        _: LlamaCPPService = llama_cpp.read(**llama_cpp_kwargs)

        status = ServerStatus(
            id=server.id,
            host=server.serverConfig.apiParams.host,
            port=server.serverConfig.apiParams.port,
            status="running",
            startedAt=dt.datetime.now(tz=dt.timezone.utc),
        )
        return status


class KuratorClient(CacheMixin, BaseClient):  # type: ignore
    _user_agent = f"anaconda-models/{version}"

    def __init__(
        self,
        domain: Optional[str] = None,
        auth_domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        kwargs: Dict[str, Any] = {"kurator": {}}
        if domain is not None:
            kwargs["kurator"]["domain"] = domain
        if ssl_verify is not None:
            kwargs["kurator"]["ssl_verify"] = ssl_verify
        if extra_headers is not None:
            kwargs["kurator"]["extra_headers"] = extra_headers

        self._config = AnacondaModelsConfig(**kwargs)

        super().__init__(
            domain=self._config.backends.kurator.domain,
            user_agent=user_agent,
            extra_headers=self._config.backends.kurator.extra_headers,
            ssl_verify=self._config.backends.kurator.ssl_verify,
            backend="memory",
        )

        auth_kwargs: Dict[str, Any] = {}
        if auth_domain is not None:
            auth_kwargs["domain"] = auth_domain
        if api_key is not None:
            auth_kwargs["api_key"] = api_key
        auth_config = AnacondaCloudConfig(**auth_kwargs)
        self.auth = BearerAuth(domain=auth_config.domain, api_key=auth_config.api_key)

        # Cache Settings
        # The cache is disabled by default, but can be enabled as needed by request
        # for a client session. New Client objects have an empty cache
        self.cache.clear()  # this is likely redundant for backend=memory, but safe
        self.expire_after = DO_NOT_CACHE

        self.models = Models(self)
        self.servers = LocalServers(self)


class AINavigatorClient(BaseClient):
    _user_agent = f"anaconda-models/{version}"

    def __init__(self, port: Optional[int] = None, api_key: Optional[str] = None):
        kwargs: Dict[str, Any] = {"ai_navigator": {}}
        if port is not None:
            kwargs["ai_navigator"]["port"] = port
        if api_key is not None:
            kwargs["ai_navigator"]["api_key"] = api_key

        self._config = AnacondaModelsConfig(**kwargs)

        if self._config.backends.ai_navigator.api_key is None:
            raise APIKeyMissing(
                f"The AI Navigator API Key was not found in {anaconda_config_path()}"
            )

        domain = f"localhost:{self._config.backends.ai_navigator.port}"

        super().__init__(
            domain=domain,
            ssl_verify=False,
            api_key=self._config.backends.ai_navigator.api_key,
        )

        self._base_uri = f"http://{domain}"

        _kurator = KuratorClient()
        self.models = Models(_kurator)
        self.servers = AINavigatorServers(self)

    def request(
        self,
        method: Union[str, bytes],
        url: Union[str, bytes],
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        try:
            return super().request(method, url, *args, **kwargs)
        except ConnectionError:
            raise RuntimeError(
                "Could not connect to AI Navigator. It may not be running. Or you may have the wrong port configured."
            )


def get_default_client() -> AINavigatorClient | KuratorClient:
    config = AnacondaModelsConfig()
    if config.default_backend == "kurator":
        return KuratorClient()
    elif config.default_backend == "ai-navigator":
        return AINavigatorClient()
    else:
        raise NotImplementedError(f"Backend {config.default_backend} not implemented")
