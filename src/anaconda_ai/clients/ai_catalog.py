from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import json
import requests
import rich.progress
from pydantic import ValidationError
from rich.console import Console
from requests.exceptions import HTTPError

from anaconda_auth.cli import _login_required_message, _continue_with_login
from anaconda_cli_base.console import console
from anaconda_cli_base.exceptions import register_error_handler

from .. import __version__ as version
from ..config import AnacondaAIConfig
from .base import GenericClient, Model, BaseModels, QuantizedFile


def catalog_login_required(
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


class AICatalogQuantizedFile(QuantizedFile):
    download_url: Optional[str] = None

    @property
    def local_path(self) -> Path:
        models_path = AnacondaAIConfig().backends.ai_catalog.models_path
        return models_path / self.identifier

    @property
    def is_downloaded(self) -> bool:
        return (
            self.local_path.exists()
            and self.local_path.stat().st_size == self.size_bytes
        )


class AICatalogModel(Model):
    quantized_files: List[AICatalogQuantizedFile]


class AICatalogModels(BaseModels):
    def __init__(self, client: GenericClient):
        super().__init__(client)

    @lru_cache
    def list(self) -> List[AICatalogModel]:
        response = self._client.get("/api/ai/model/org/models/model-data")
        # response = self._client.get("/api/ai/model/models")
        response.raise_for_status()
        data = response.json()["result"]["data"]

        models = []
        for model in data:
            try:
                entry = AICatalogModel(**model)
                entry._client = self._client
                models.append(entry)
            except ValidationError as e:
                raise ValueError(
                    f"Could not process {model['name']} ({model['model_uuid']})\n{e}"
                )
        return models

    def _download(
        self,
        model_quantization: AICatalogQuantizedFile,
        show_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
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
            total=int(model_quantization.size_bytes),
            visible=show_progress,
        )

        model_quantization.local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_quantization.local_path, "wb") as f:
            with stream_progress as s:
                for chunk in response.iter_content(1024**2):
                    f.write(chunk)
                    s.update(task, advance=len(chunk))

    def _delete(self, model_quantization: AICatalogQuantizedFile) -> None:
        model_quantization.local_path.unlink()


class AICatalogClient(GenericClient):
    _user_agent = f"anaconda-ai/{version}"

    def __init__(
        self,
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

        kwargs_top = {"backends": {"ai_catalog": kwargs}}
        self._ai_config = AnacondaAIConfig(**kwargs_top)  # type: ignore

        super().__init__(
            user_agent=user_agent,
            api_key=api_key,
            domain=self._ai_config.backends.ai_catalog.domain,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
        )

        if self._ai_config.backends.ai_catalog.api_version is not None:
            self.headers["X-Anaconda-Api-Version"] = (
                self._ai_config.backends.ai_catalog.api_version
            )

        self.models = AICatalogModels(self)
        self.hooks["response"].insert(0, catalog_login_required)
