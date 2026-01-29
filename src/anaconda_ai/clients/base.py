import atexit
import re
from pathlib import Path
from time import time
from types import TracebackType
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from typing import Set
from typing import Sequence
from typing import MutableMapping
from typing_extensions import Self
from urllib.parse import urljoin

import openai
import rich.progress
from pydantic import BaseModel, computed_field, model_validator, Field, PrivateAttr
from rich.status import Status
from rich.console import Console

from anaconda_auth.client import BaseClient
from anaconda_auth.config import AnacondaAuthSite
from .. import __version__ as version
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


class GenericClient(BaseClient):
    _user_agent = f"anaconda-ai/{version}"
    models: "BaseModels"
    servers: "BaseServers"
    vector_db: "BaseVectorDb"

    def __init__(
        self,
        site: Optional[Union[str, AnacondaAuthSite]] = None,
        base_uri: Optional[str] = None,
        domain: Optional[str] = None,
        auth_domain_override: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_version: Optional[str] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
        extra_headers: Optional[Union[str, dict]] = None,
        hash_hostname: Optional[bool] = None,
        proxy_servers: Optional[MutableMapping[str, str]] = None,
        client_cert: Optional[str] = None,
        client_cert_key: Optional[str] = None,
        stop_server_on_exit: Optional[bool] = None,
        server_operations_timeout: Optional[int] = None,
    ):
        super().__init__(
            site=site,
            base_uri=base_uri,
            domain=domain,
            auth_domain_override=auth_domain_override,
            api_key=api_key,
            user_agent=user_agent,
            api_version=api_version,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
            hash_hostname=hash_hostname,
            proxy_servers=proxy_servers,
            client_cert=client_cert,
            client_cert_key=client_cert_key,
        )

        ai_kwargs = {}
        if stop_server_on_exit is not None:
            ai_kwargs["stop_server_on_exit"] = stop_server_on_exit
        if server_operations_timeout is not None:
            ai_kwargs["server_operations_timeout"] = server_operations_timeout
        self.ai_config = AnacondaAIConfig().model_copy(update=ai_kwargs, deep=True)

    def get_version(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def online(self) -> bool:
        raise NotImplementedError


class QuantizedFile(BaseModel):
    sha256: str
    size_bytes: int
    quant_method: str
    format: str
    max_ram_usage: int
    _model: "Model" = PrivateAttr()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def local_path(self) -> Path:
        raise NotImplementedError

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_allowed(self) -> bool:
        return True

    @computed_field  # type: ignore[prop-decorator]
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
    quantized_files: Sequence[QuantizedFile]
    _client: GenericClient = PrivateAttr()

    def __init__(self, client: GenericClient, **data: Any) -> None:
        super().__init__(**data)
        self._client = client

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

    def delete(
        self,
        method: str,
    ) -> None:
        quant = self.get_quantization(method)
        quant.delete()


class BaseModels:
    client: GenericClient

    def __init__(self, client: GenericClient):
        self.client = client

    def list(self) -> Sequence[Model]:
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

        return model_info

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

    def _download(
        self,
        model_quantization: QuantizedFile,
        path: Optional[Path] = None,
    ) -> Generator[int, None, None]:
        raise NotImplementedError(
            "Downloading models is not available with this client"
        )

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

        if model_quantization.is_downloaded:
            return

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

        with stream_progress as progress_bar:
            for downloaded_bytes in self._download(model_quantization, path):
                progress_bar.update(task, completed=downloaded_bytes)

    def _delete(self, model_quantization: QuantizedFile) -> None:
        raise NotImplementedError

    def delete(self, model_quantization: Union[str, QuantizedFile]) -> None:
        if isinstance(model_quantization, str):
            model_quantization = self._find_quantization(model_quantization)

        self._delete(model_quantization)


class ServerConfig(BaseModel):
    model_name: str

    _params_dump: Set[str] = set()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def params(self) -> dict:
        return self.model_dump(
            include=self._params_dump,
            exclude_none=True,
            exclude_defaults=True,
        )


class Server(BaseModel):
    id: str
    url: str
    config: ServerConfig
    api_key: str = "empty"
    _client: GenericClient = PrivateAttr()
    _matched: bool = PrivateAttr(default=False)

    def __init__(self, client: GenericClient, **data: Any) -> None:
        super().__init__(**data)
        self._client = client

    @computed_field  # type: ignore[prop-decorator]
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
        text = f"{self.config.model_name} (creating)"
        console = Console() if console is None else console
        console.quiet = not show_progress
        with Status(text, console=console) as display:
            self._client.servers.start(self)
            status = "starting"
            text = f"{self.config.model_name} ({status})"
            display.update(text)

            t0 = time()
            start_timeout = self._client.ai_config.server_operations_timeout
            while status != "running":
                status = self._client.servers.status(self)
                if status == "errored":
                    raise AnacondaAIException("Server failed to start")
                text = f"{self.config.model_name} ({status})"
                display.update(text)
                t1 = time()
                if (t1 - t0) > start_timeout:
                    raise AnacondaAIException(
                        f"Server start timed out after {start_timeout} seconds"
                    )
        console.print(f"[bold green]✓[/] {text}", highlight=False)

        if not self._matched:
            kwargs = {}
            if leave_running is not None:
                kwargs["stop_server_on_exit"] = not leave_running

            config = self._client.ai_config.model_copy(update=kwargs)
            if config.stop_server_on_exit:

                def safe_stop(console: Console) -> None:
                    try:
                        self.stop(console=console)
                    except Exception:
                        pass

                atexit.register(safe_stop, console=console)

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def stop(
        self, show_progress: bool = True, console: Optional[Console] = None
    ) -> None:
        console = Console() if console is None else console
        console.quiet = not show_progress
        text = f"{self.config.model_name} (stopping)"
        with Status(text, console=console) as display:
            status = "stopping"
            self._client.servers.stop(self.id)
            while status != "stopped":
                status = self._client.servers.status(self)
                text = f"{self.config.model_name} ({status})"
                display.update(text)
        console.print(f"[bold green]✓[/] {text}", highlight=False)

    def delete(
        self, show_progress: bool = True, console: Optional[Console] = None
    ) -> None:
        console = Console() if console is None else console
        console.quiet = not show_progress
        text = f"{self.config.model_name} (stopping)"
        with Status(text, console=console) as display:
            self._client.servers.delete(self.id)
            text = f"{self.config.model_name} (deleted)"
            display.update(text)
        console.print(f"[bold green]✓[/] {text}", highlight=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def openai_url(self) -> str:
        base_url = self.url if self.url.endswith("/") else f"{self.url}/"
        return urljoin(base_url, "v1")

    def openai_client(self, **kwargs: Any) -> openai.OpenAI:
        client = openai.OpenAI(base_url=self.openai_url, api_key=self.api_key, **kwargs)
        return client

    def async_openai_client(self, **kwargs: Any) -> openai.AsyncOpenAI:
        client = openai.AsyncOpenAI(
            base_url=self.openai_url, api_key=self.api_key, **kwargs
        )
        return client


class BaseServers:
    always_detach: bool = False
    download_required: bool = True
    client: GenericClient

    def __init__(self, client: GenericClient):
        self.client = client

    def _get_server_id(self, server: Union[Server, str]) -> str:
        if isinstance(server, Server):
            server_id = str(server.id)
        elif isinstance(server, str):
            server_id = server
        else:
            raise ValueError(f"{server} is not a valid Server identifier")

        return server_id

    def list(self) -> Sequence[Server]:
        raise NotImplementedError

    def get(self, server: str) -> Server:
        raise NotImplementedError

    def _create(
        self,
        model_quantization: QuantizedFile,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Server:
        raise NotImplementedError

    def create(
        self,
        model: Union[str, QuantizedFile],
        download_if_needed: bool = True,
        extra_options: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> Server:
        if isinstance(model, str):
            match = MODEL_NAME.match(model)
            if match is None:
                raise ValueError(
                    f"{model} does not look like a quantized model name in the format <model>/<quant>"
                )

            _, model_name, quantization, _ = match.groups()
            if model_name is None or quantization is None:
                raise ValueError(
                    f"{model} does not look like a quantized model name in the format <model>/<quant>"
                )
            quantization = quantization.upper()

            if not quantization:
                raise ValueError(
                    "You must provide a quantization level in the model name as <model>/<quant>"
                )

            model = self.client.models.get(model_name).get_quantization(quantization)
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
                    self.client.models.download(model)

        console = Console() if console is None else console
        console.quiet = not show_progress
        text = f"{model.identifier} (creating)"
        with Status(text, console=console) as display:
            server = self._create(model, extra_options=extra_options)
            text = f"{model.identifier} (created)"
            display.update(text)

        return server

    def _start(self, server_id: str) -> None:
        raise NotImplementedError

    def start(self, server: Union[Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._start(server_id)

    def _status(self, server_id: str) -> str:
        raise NotImplementedError

    def status(self, server: Union[Server, str]) -> str:
        server_id = self._get_server_id(server)
        status = self._status(server_id)
        return status

    def _stop(self, server_id: str) -> None:
        raise NotImplementedError

    def stop(self, server: Union[Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._stop(server_id)

    def _delete(self, server_id: str) -> None:
        raise NotImplementedError

    def delete(self, server: Union[Server, str]) -> None:
        server_id = self._get_server_id(server)
        self._delete(server_id)


class VectorDbServerResponse(BaseModel):
    running: bool
    host: str
    port: int
    database: str
    user: str
    password: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uri(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class VectorDbTableColumn(BaseModel):
    name: str
    type: str
    constraints: List[str] = Field(default_factory=list)


class VectorDbTableSchema(BaseModel):
    columns: List[VectorDbTableColumn]


class TableInfo(BaseModel):
    name: str
    table_schema: VectorDbTableSchema = Field(alias="schema")
    numRows: int


class BaseVectorDb:
    client: GenericClient

    def __init__(self, client: GenericClient) -> None:
        self.client = client

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
