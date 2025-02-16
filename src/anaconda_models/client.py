import re
import datetime as dt
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from pydantic import BaseModel, HttpUrl
from requests import Response
from requests_cache import CacheMixin, DO_NOT_CACHE
from requests.exceptions import ConnectionError

from anaconda_cloud_auth.client import BaseClient
from anaconda_cloud_auth.client import BearerAuth
from anaconda_cloud_auth.config import AnacondaCloudConfig
from anaconda_models import __version__ as version
from anaconda_models.config import ModelsConfig
from anaconda_models.exceptions import ModelNotFound

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


class Models(BaseClient):
    def __init__(self, client: BaseClient):
        self._client = client

    def list(self) -> list[Model]:
        response = self._client.get("/api/models", expire_after=60)
        response.raise_for_status()
        data = response.json()["result"]["data"]
        models = [Model(**m) for m in data]
        return models

    def get(self, model) -> Model:
        match = MODEL_NAME.match(model)
        if match is None:
            raise ValueError(f"{model} does not look like a model name.")

        _, model_name, _, _ = match.groups()

        models = self.list()
        for entry in models:
            if entry.name.lower() == model_name.lower():
                return entry
            elif entry.id.lower().endswith(model_name.lower()):
                return entry
        else:
            raise ModelNotFound(f"{model} was not found")


class Server(BaseModel): ...


class Servers(BaseClient):
    def list(self): ...

    def create(self, model, kwargs): ...

    def get(self, server_id): ...

    def delete(self, server_id): ...


class Client(CacheMixin, BaseClient):  # type: ignore
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
        kwargs: Dict[str, Any] = {}
        if domain is not None:
            kwargs["domain"] = domain
        if ssl_verify is not None:
            kwargs["ssl_verify"] = ssl_verify
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers

        self._config = ModelsConfig(**kwargs)

        super().__init__(
            domain=self._config.domain,
            user_agent=user_agent,
            extra_headers=self._config.extra_headers,
            ssl_verify=self._config.ssl_verify,
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


class AINavigatorClient(BaseClient):
    _user_agent = f"anaconda-models/{version}"

    def __init__(
        self,
        port: Optional[int] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if port is not None:
            kwargs["ai_navigator"] = {"port": port}

        self._config = ModelsConfig(**kwargs)

        domain = f"localhost:{self._config.ai_navigator.port}"

        super().__init__(
            domain=domain, ssl_verify=False, api_key=self._config.ai_navigator.api_key
        )

        self._base_uri = f"http://{domain}"

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
