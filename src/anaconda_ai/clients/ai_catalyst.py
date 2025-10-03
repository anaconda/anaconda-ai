import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Generator
from uuid import UUID

import json
import requests
from pydantic import ValidationError, computed_field, BaseModel
from requests.exceptions import HTTPError

from anaconda_auth.cli import _login_required_message, _continue_with_login
from anaconda_cli_base.console import console
from anaconda_cli_base.exceptions import register_error_handler

from .. import __version__ as version
from ..config import AnacondaAIConfig
from .base import (
    GenericClient,
    Model,
    BaseModels,
    QuantizedFile,
    BaseServers,
    Server,
    ServerConfig,
)


def catalyst_login_required(
    response: requests.Response, *args: Any, **kwargs: Any
) -> requests.Response:
    has_auth_header = response.request.headers.get("Authorization", False)

    if response.status_code in [401, 403]:
        try:
            error_code = response.json().get("detail", "")
        except requests.JSONDecodeError:
            error_code = ""

        if error_code == "Not authenticated":
            if has_auth_header:
                response.reason = "Your API key or login token is invalid."
            else:
                response.reason = (
                    "You must login before using this API endpoint"
                    " or provide an api_key to your client."
                )

    return response


@register_error_handler(HTTPError)
def http_error(e: HTTPError) -> int:
    try:
        error_code = e.response.json().get("detail", "")
    except json.JSONDecodeError:
        error_code = ""

    if error_code == "Not authenticated":
        if "Authorization" in e.request.headers:
            console.print(
                "[bold][red]InvalidAuthentication:[/red][/bold] Your provided API Key or login token is invalid"
            )
        else:
            _login_required_message("AuthenticationMissingError")
        return _continue_with_login()
    else:
        console.print(f"[bold][red]{e.__class__.__name__}:[/red][/bold] {e}")
        return 1


class Policy(BaseModel):
    is_blocked: bool
    allowed_groups: List[Any]


class AICatalystQuantizedFile(QuantizedFile):
    file_uuid: UUID
    model_uuid: UUID
    generated_on: dt.datetime
    quant_engine: str
    published: bool
    file_type_id: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    updated_at: Optional[dt.datetime] = None
    context_window_size: Optional[int]
    estimated_n_cpus_req: Optional[int] = None
    download_url: Optional[str] = None

    @computed_field
    @property
    def identifier(self) -> str:
        return f"{self._model.name}_{self.quant_method}.{self.format.lower()}"

    @property
    def local_path(self) -> Path:
        models_path = AnacondaAIConfig().backends.ai_catalyst.models_path
        return models_path / self.identifier

    @property
    def is_downloaded(self) -> bool:
        return (
            self.local_path.exists()
            and self.local_path.stat().st_size == self.size_bytes
        )


class Tag(BaseModel):
    id: int
    name: str


class Group(BaseModel):
    id: int
    name: str


class AICatalystModel(Model):
    model_uuid: UUID
    model_type: str
    base_model: str
    license: str
    languages: List[str]
    first_published: dt.datetime
    knowledge_cut_off: str
    groups: List[Group]
    tags: List[Tag]
    quantized_files: List[AICatalystQuantizedFile]


class AICatalystModels(BaseModels):
    def __init__(self, client: GenericClient):
        super().__init__(client)

    @lru_cache
    def list(self) -> List[AICatalystModel]:
        response = self._client.get("/api/ai/model/org/models/model-data")
        # response = self._client.get("/api/ai/model/models")
        response.raise_for_status()
        data = response.json()["result"]["data"]

        models = []
        for model in data:
            try:
                entry = AICatalystModel(**model)
                entry._client = self._client
                models.append(entry)
            except ValidationError as e:
                raise ValueError(
                    f"Could not process {model['name']} ({model['model_uuid']})\n{e}"
                )
        return models

    def _get_model_by_uuid(self, model_uuid: UUID) -> AICatalystModel:
        res = self._client.get(f"/api/ai/models/{model_uuid}")
        res.raise_for_status()
        return AICatalystModel(**res.json())

    def _get_quantization_by_uuid(
        self, model_uuid: UUID, file_uuid: UUID
    ) -> AICatalystQuantizedFile:
        res = self._client.get(f"/api/ai/models/{model_uuid}/quants/{file_uuid}")
        res.raise_for_status()
        return AICatalystQuantizedFile(**res.json())

    def _download(
        self,
        model_quantization: AICatalystQuantizedFile,
        path: Optional[Path] = None,
    ) -> Generator[int, None, None]:
        if not model_quantization.published:
            raise RuntimeError(f"{model_quantization.identifier} is not published")

        if not model_quantization.download_url:
            raise RuntimeError(
                f"Cannot find download url for {model_quantization.identifier}"
            )

        response = self._client.get(model_quantization.download_url, stream=True)
        # download_url = f"api/ai/model/models/{model_quantization._model.model_uuid}/files/{model_quantization.file_uuid}/download"
        # response = self._client.get(download_url, params={"redirect": False}, stream=True)

        response.raise_for_status()

        model_quantization.local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_quantization.local_path, "wb") as f:
            downloaded_bytes = 0
            for chunk in response.iter_content(1024**2):
                f.write(chunk)
                downloaded_bytes += len(chunk)
                yield downloaded_bytes

        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            model_quantization.local_path.link_to(path)

    def _delete(self, model_quantization: AICatalystQuantizedFile) -> None:
        model_quantization.local_path.unlink()


class AICatalystServerConfig(ServerConfig):
    # model: AICatalystModel
    name: str
    model_uuid: UUID
    file_uuid: UUID
    address: str
    port: int
    # start_ts: dt.datetime


class AICatalystServer(Server):
    serverConfig: AICatalystServerConfig

    @property
    def api_key(self) -> str:
        return (
            self._client.auth.api_key
            or self._client.auth._token_info.get_access_token()
        )  # type: ignore

    @computed_field
    @property
    def url(self) -> str:
        return f"{self.serverConfig.address}:{self.serverConfig.port}"


class AICatalystServers(BaseServers):
    @lru_cache
    def list(self) -> List[AICatalystServer]:
        res = self._client.get("api/ai/model/org/servers")
        res.raise_for_status()
        discovered = res.json().get("result", {}).get("data", [])

        servers = []
        for server in discovered:
            model = AICatalystModel(**server["model"])
            server_entry = AICatalystServer(
                id=server["id"],
                serverConfig=AICatalystServerConfig(
                    model_name=model.quantized_files[0].identifier, **server
                ),
            )
            server_entry._client = self._client
            servers.append(server_entry)

        return servers

    def _status(self, server_id: str) -> str:
        res = self._client.get(f"api/ai/model/org/servers/server/{server_id}")
        res.raise_for_status()
        return res.json()["status"]


class AICatalystClient(GenericClient):
    _user_agent = f"anaconda-ai/{version}"

    def __init__(
        self,
        site: Optional[str] = None,
        domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        api_version: Optional[str] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if domain is not None:
            kwargs["domain"] = domain

        if api_version is not None:
            kwargs["api_version"] = api_version

        kwargs_top = {"backends": {"ai_catalyst": kwargs}}
        self._ai_config = AnacondaAIConfig(**kwargs_top)  # type: ignore

        super().__init__(
            site=site,
            user_agent=user_agent,
            api_key=api_key,
            domain=domain,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )

        if self._ai_config.backends.ai_catalyst.api_version is not None:
            self.headers["X-Anaconda-Api-Version"] = (
                self._ai_config.backends.ai_catalyst.api_version
            )

        self.models = AICatalystModels(self)
        self.servers = AICatalystServers(self)
        self.hooks["response"].insert(0, catalyst_login_required)
