import datetime as dt
from functools import cached_property, lru_cache
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Generator, MutableMapping
from uuid import UUID

import json
import requests
from pydantic import PrivateAttr, ValidationError, computed_field, BaseModel
from requests.exceptions import HTTPError

from anaconda_auth.cli import _login_required_message, _continue_with_login
from anaconda_auth.token import TokenInfo
from anaconda_cli_base.console import console
from anaconda_cli_base.exceptions import register_error_handler

from anaconda_ai.exceptions import AnacondaAIException, QuantizedFileNotFound

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


class ServerNotFoundError(AnacondaAIException): ...


class ModelNotAvailableError(AnacondaAIException): ...


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


class AICatalystQuantizedFilePolicy(BaseModel):
    is_blocked: bool
    missing_license_acceptance: bool
    allowed_groups: List[str]
    download_status: str


class AICatalystQuantizedFile(QuantizedFile):
    file_uuid: UUID
    model_uuid: UUID
    generated_on: dt.datetime
    quant_engine: str
    published: bool
    created_at: Optional[dt.datetime] = None
    updated_at: Optional[dt.datetime] = None
    context_window_size: Optional[int]
    estimated_n_cpus_req: Optional[int] = None
    policy: Optional[AICatalystQuantizedFilePolicy] = None
    _model: "AICatalystModel" = PrivateAttr()

    @computed_field
    @property
    def is_allowed(self) -> bool:
        if self.policy is None:
            return True

        if self.policy.is_blocked:
            return False
        elif any([g in self.policy.allowed_groups for g in self._model._client.groups]):
            if self.policy.missing_license_acceptance:
                return False
            elif self.policy.download_status != "downloaded":
                return False
            else:
                return True
        else:
            return False

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

    @property
    def download_url(self) -> str:
        return f"/api/ai/model/models/{self.model_uuid}/files/{self.file_uuid}/download"


class Tag(BaseModel):
    id: int
    name: str


class Group(BaseModel):
    id: int
    name: str


class AICatalystConvertedFiles(BaseModel):
    generated_on: dt.datetime
    published: bool


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
    converted_files: List[AICatalystConvertedFiles]
    _client: "AICatalystClient" = PrivateAttr()


class AICatalystModels(BaseModels):
    @lru_cache
    def list(self) -> List[AICatalystModel]:
        response = self.client.get("/api/ai/model/org/models/model-data")
        response.raise_for_status()
        data = response.json()["result"]["data"]

        models = []
        for model in data:
            try:
                entry = AICatalystModel(client=self.client, **model)
                models.append(entry)
            except ValidationError as e:
                raise ValueError(
                    f"Could not process {model['name']} ({model['model_uuid']})\n{e}"
                )
        return models

        # Save only the most recently generated model for each unique name
        # sorted_models = sorted(models, key=lambda m: m.converted_files[0].generated_on, reverse=False)
        # by_name = {model.name: model for model in sorted_models}
        # deduped_models = list[AICatalystModel](by_name.values())
        # return deduped_models

    def _get_model_by_uuid(self, model_uuid: Union[UUID, str]) -> AICatalystModel:
        res = self.client.get(f"/api/ai/model/models/{model_uuid}")
        res.raise_for_status()
        return AICatalystModel(client=self.client, **res.json()["data"])

    def _get_quantization_by_uuid(
        self, model_uuid: Union[UUID, str], file_uuid: Union[str, UUID]
    ) -> AICatalystQuantizedFile:
        if isinstance(file_uuid, str):
            file_uuid = UUID(file_uuid)

        model = self._get_model_by_uuid(model_uuid)
        for quant in model.quantized_files:
            if quant.file_uuid == file_uuid:
                return quant

        raise QuantizedFileNotFound(
            f"Could not find quantized file UUID {file_uuid} for model {model_uuid}"
        )
        # res = self._client.get(f"/api/ai/model/models/{model_uuid}/files/{file_uuid}")
        # res.raise_for_status()
        # return AICatalystQuantizedFile(**res.json()["data"])

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

        if not model_quantization.is_allowed:
            policy = model_quantization.policy.model_dump_json(indent=2)
            raise ModelNotAvailableError(
                f"{model_quantization.identifier} cannot be downloaded\nPolicy:\n{policy}"
            )

        res = self.client.get(model_quantization.download_url)
        res.raise_for_status()
        signed_url = res.json()["download_url"]
        response = requests.get(signed_url, stream=True)

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
            path.hardlink_to(model_quantization.local_path)

    def _delete(self, model_quantization: AICatalystQuantizedFile) -> None:
        model_quantization.local_path.unlink()


class AICatalystServerConfig(ServerConfig):
    # model: AICatalystModel
    # id: UUID
    # owner: str
    # uuid: UUID
    model_uuid: UUID
    file_uuid: UUID
    # address: str
    # status: str


class AICatalystServer(Server):
    id: str
    uuid: UUID
    owner: str
    config: AICatalystServerConfig
    _client: "AICatalystClient" = PrivateAttr()


class AICatalystServers(BaseServers):
    client: "AICatalystClient"
    download_required: bool = False

    @lru_cache
    def list(self) -> List[AICatalystServer]:
        res = self.client.get("api/ai/inference/servers")
        res.raise_for_status()
        discovered = res.json().get("data", {}).get("servers", [])

        servers = []
        for server in discovered:
            model = self.client.models._get_quantization_by_uuid(
                server["model_uuid"], server["file_uuid"]
            )
            model_name = model.identifier
            server_entry = AICatalystServer(
                id=server["name"],
                uuid=server["id"],
                url=server["address"],
                owner=server["owner"],
                client=self.client,
                api_key=self.client.api_key,
                config=AICatalystServerConfig(
                    model_name=model_name,
                    model_uuid=server["model_uuid"],
                    file_uuid=server["file_uuid"],
                ),
            )
            servers.append(server_entry)

        return servers

    def get(self, server: Union[str, UUID]) -> AICatalystServer:
        servers = self.list()

        try:
            uuid = UUID(server)
        except ValueError:
            uuid = None

        for found_server in servers:
            if uuid and found_server.uuid == uuid:
                return found_server
            elif server == found_server.id:
                return found_server
        raise ServerNotFoundError(f"Server {server} was not found.")

    def _create(
        self,
        model_quantization: AICatalystQuantizedFile,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Server:
        file_uuid = model_quantization.file_uuid
        model_uuid = model_quantization.model_uuid

        existing_servers = self.client.servers.list()
        for server in existing_servers:
            config = server.config.model_dump(exclude={"model_name"})
            if {"file_uuid": file_uuid, "model_uuid": model_uuid} == config:
                return server

        extra_options = extra_options or {}
        body = {
            "name": extra_options.get("name", "") or model_quantization.identifier,
            "file_uuid": str(file_uuid),
            "model_uuid": str(model_uuid),
        }

        res = self.client.post("api/ai/inference/servers/server", json=body)
        res.raise_for_status()
        data = res.json()["data"]

        server_config = AICatalystServerConfig(
            model_name=model_quantization.identifier,
            model_uuid=data["model_uuid"],
            file_uuid=data["file_uuid"],
        )
        server_entry = AICatalystServer(
            uuid=data["id"],
            id=data["name"],
            owner=data["owner"],
            url=data["address"],
            config=server_config,
            client=self.client,
            api_key=self.client.api_key,
        )
        from time import sleep

        sleep(5)

        return server_entry

    def _get_server_id(self, server: Union[AICatalystServer, UUID, str]) -> str:
        if isinstance(server, AICatalystServer):
            server_id = str(server.uuid)
        elif isinstance(server, str):
            server_entry = self.get(server)
            server_id = server_entry.uuid

        return server_id
        # return super()._get_server_id(server)

    def _start(self, server_id: str) -> None:
        res = self.client.post(f"api/ai/inference/servers/server/{server_id}/start")
        res.raise_for_status()

    def _stop(self, server_id: str) -> None:
        res = self.client.post(f"api/ai/inference/servers/server/{server_id}/stop")
        res.raise_for_status()

    def _status(self, server_id: str) -> str:
        res = self.client.get("api/ai/inference/servers")
        res.raise_for_status()
        for server in res.json()["data"]["servers"]:
            if server["id"] == str(server_id):
                break
        else:
            return "<unknown>"

        if server["status"] == "stopped":
            return "stopped"

        res = self.client.get(f"{server['address']}/v1/models")
        if res.ok:
            return "running"
        else:
            return "starting"

        # not currently working
        # try:
        #     res = self._client.get(f"api/ai/inference/servers/server/{server_id}")
        #     res.raise_for_status()
        #     status = res.json()["data"]["status"]
        # except Exception:
        #     status = "<unknown>"
        # return status

    def _delete(self, server_id: str) -> None:
        res = self.client.get(f"api/ai/inference/servers/server/{server_id}")
        res.raise_for_status()


class AICatalystClient(GenericClient):
    models: AICatalystModels
    servers: AICatalystServers

    def __init__(
        self,
        site: Optional[str] = None,
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
    ):
        ai_kwargs: Dict[str, Any] = {}
        if api_version is not None:
            ai_kwargs["api_version"] = api_version

        kwargs_top = {"backends": {"ai_catalyst": ai_kwargs}}
        self._ai_config = AnacondaAIConfig(**kwargs_top)  # type: ignore

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

        if self._ai_config.backends.ai_catalyst.api_version is not None:
            self.headers["X-Anaconda-Api-Version"] = (
                self._ai_config.backends.ai_catalyst.api_version
            )

        self.models = AICatalystModels(self)
        self.servers = AICatalystServers(self)
        self.hooks["response"].insert(0, catalyst_login_required)

    @cached_property
    def groups(self) -> list[str]:
        res = self.get("api/auth/sessions/whoami")
        res.raise_for_status()

        orgs = res.json()["passport"]["organizations"]
        for org in orgs:
            if org["org_id"] == "repo":
                break
        groups = [g["id"] for g in org["groups"]]
        return groups

    @cached_property
    def api_key(self) -> str:
        key = self.config.api_key or TokenInfo.load(domain=self.config.domain).api_key
        return key
