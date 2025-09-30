import atexit
import re
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from typing_extensions import Self
from urllib.parse import urljoin
from uuid import UUID

import openai
from pydantic import BaseModel, computed_field, model_validator, Field, PrivateAttr
from pydantic.types import UUID4
from rich.status import Status
from rich.console import Console

from anaconda_auth.client import BaseClient
from ..config import AnacondaAIConfig
from ..exceptions import (
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


def raises(ex: Exception):
    """Raises and exception"""
    raise ex


class GenericClient(BaseClient):
    models: "BaseModels"
    servers: "BaseServers"
    vector_db: "BaseVectorDb"
    _config: AnacondaAIConfig = PrivateAttr()

    def get_version(self) -> str:
        raise NotImplementedError


class QuantizedFile(BaseModel):
    sha256: str
    size_bytes: int
    quant_method: str
    format: str
    max_ram_usage: int
    _model: "Model" = PrivateAttr()

    @computed_field
    @property
    def identifier(self) -> str:
        return f"{self._model.name}_{self.quant_method}.{self.format.lower()}"

    @property
    def is_downloaded(self) -> bool:
        raise NotImplementedError

    def download(
        self,
        show_progress: bool = True,
        console: Optional[Console] = None,
        path: Optional[Union[Path, str]] = None,
    ) -> None:
        self._model._client.models.download(
            self, show_progress=show_progress, console=console, path=path
        )

    def delete(self) -> None:
        self._model._client.models.delete(self)


class Model(BaseModel):
    name: str
    description: str
    num_parameters: int
    trained_for: str
    context_window_size: int
    quantized_files: List[QuantizedFile]
    _client: GenericClient = PrivateAttr()

    @model_validator(mode="after")
    def refined_quantized_list(self) -> Self:
        for quant in self.quantized_files:
            quant._model = self
        self.quantized_files = sorted(
            self.quantized_files, key=lambda q: q.quant_method
        )
        return self

    def get_quantization(self, method: str) -> QuantizedFile:
        for quant in self.quantized_files:
            if quant.quant_method.lower() == method.lower():
                return quant
        else:
            raise QuantizedFileNotFound(
                f"Quantization {method} not found for {self.name}."
            )

    def download(
        self,
        method: str,
        show_progress: bool = True,
        console: Optional[Console] = None,
        path: Optional[Union[Path, str]] = None,
    ) -> None:
        quant = self.get_quantization(method)
        quant.download(show_progress=show_progress, console=console, path=path)


class BaseModels:
    def __init__(self, client: GenericClient):
        self._client = client

    def list(self) -> List[Model]:
        raise NotImplementedError

    def get(self, model: str) -> Model:
        match = MODEL_NAME.match(model)
        if match is None:
            raise ValueError(f"{model} does not look like a model name.")

        _, model_name, _, _ = match.groups()

        models = self.list()
        for entry in models:
            if entry.name.lower() == model_name.lower():
                model_info = entry
                break
        else:
            raise ModelNotFound(f"{model} was not found")

        model_info._client = self._client
        return model_info

    def _download(
        self,
        model_quantization: QuantizedFile,
        path: Optional[Path] = None,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        raise NotImplementedError(
            "Downloading models is not available with this client"
        )

    def _find_quantization(self, model_quant_identifier: str) -> QuantizedFile:
        match = MODEL_NAME.match(model_quant_identifier)
        if match is None:
            raise ValueError(
                f"{model_quant_identifier} does not look like a model quantization identifier."
            )

        _, model_name, quant_method, _ = match.groups()

        if quant_method is None:
            raise ValueError(
                "You must include the quantization method in the model as <model>/<quantization>"
            )

        model_info = self.get(model_name)
        quantization = model_info.get_quantization(quant_method)
        return quantization

    def download(
        self,
        model_quantization: Union[str, QuantizedFile],
        path: Optional[Union[Path, str]] = None,
        force: bool = False,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        if isinstance(model_quantization, str):
            model_quantization = self._find_quantization(model_quantization)

        if force:
            self.delete(model_quantization)

        path = path if path is None else Path(path)

        if not model_quantization.is_downloaded:
            self._download(
                model_quantization=model_quantization,
                path=path,
                show_progress=show_progress,
                console=console,
            )

    def _delete(self, model_quantization: QuantizedFile) -> None:
        raise NotImplementedError

    def delete(self, model_quantization: Union[str, QuantizedFile]) -> None:
        if isinstance(model_quantization, str):
            model_quantization = self._find_quantization(model_quantization)

        self._delete(model_quantization)


class ServerConfig(BaseModel):
    model_name: str


class Server(BaseModel):
    id: Union[UUID4, str]
    serverConfig: ServerConfig
    api_key: Optional[str] = "empty"
    _client: GenericClient = PrivateAttr()
    _matched: bool = PrivateAttr(default=False)

    @computed_field
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
        text = f"{self.serverConfig.model_name} (creating)"
        console = Console() if console is None else console
        console.quiet = not show_progress
        with Status(text, console=console) as display:
            self._client.servers.start(self)
            status = "starting"
            text = f"{self.serverConfig.model_name} ({status})"
            display.update(text)

            while status != "running":
                status = self._client.servers.status(self)
                text = f"{self.serverConfig.model_name} ({status})"
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
        text = f"{self.serverConfig.model_name} (stopping)"
        with Status(text, console=console) as display:
            status = "stopping"
            self._client.servers.stop(self.id)
            while status != "stopped":
                status = self._client.servers.status(self)
                text = f"{self.serverConfig.model_name} ({status})"
                display.update(text)
        console.print(f"[bold green]✓[/] {text}", highlight=False)

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        raise NotImplementedError()

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
    always_detach: bool = False
    download_required: bool = True
    match_field_excludes: set[str] = set()

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
        servers = self.list()
        for server in servers:
            config_dump = server_config.model_dump(exclude=self.match_field_excludes)
            server_dump = server.serverConfig.model_dump(
                exclude=self.match_field_excludes
            )
            if server.is_running and (config_dump == server_dump):
                server._matched = True
                return server
        else:
            return None

    def _create(self, server_config: ServerConfig) -> Server:
        raise NotImplementedError

    def create(
        self,
        model: Union[str, QuantizedFile],
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

            model = self._client.models.get(model_name).get_quantization(quantization)
        elif isinstance(model, QuantizedFile):
            pass
        else:
            raise ValueError(
                f"model={model} of type {type(model)} is not a supported way to specify a model."
            )

        if self.download_required:
            if not model.is_downloaded:
                if not download_if_needed:
                    raise ModelNotDownloadedError(f"{model} has not been downloaded")
                else:
                    self._client.models.download(model)

        server_config = ServerConfig(
            model_name=model.identifier,
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

    def create(
        self,
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
