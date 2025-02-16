import atexit
import os
import sys
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
from urllib.parse import urljoin

import openai
import rich.progress
from intake.readers.datatypes import LlamaCPPService
from intake.readers.readers import LlamaServerReader
from rich.console import Console

from anaconda_models.client import AINavigatorClient, MODEL_NAME, Model
from anaconda_models.client import Client
from anaconda_models.config import config
from anaconda_models.exceptions import ModelNotFound
from anaconda_models.exceptions import QuantizedFileNotFound
from anaconda_models.utils import find_free_port


def get_models(client: Optional[Client] = None, expire_after: int = 60) -> Any:
    """Metadata for all models"""
    if client is None:
        client = Client()
    response = client.get("/api/models", expire_after=expire_after)
    response.raise_for_status()
    return [Model(**m) for m in response.json()["result"]["data"]]


def model_info(model: str, client: Optional[Client] = None) -> Any:
    """Metadata for a single model"""
    match = MODEL_NAME.match(model)
    if match is None:
        raise ValueError(f"{model} does not look like a model name.")

    _, model_name, _, _ = match.groups()

    models = get_models(client=client)
    for entry in models:
        if entry["name"].lower() == model_name.lower():
            return entry
        elif entry["id"].lower().endswith(model_name.lower()):
            return entry


def quantized_model_info(
    model: str,
    quantization: Optional[str] = None,
    format: Optional[str] = None,
    client: Optional[Client] = None,
) -> dict:
    """Metadata about a single quantized model file"""
    match = MODEL_NAME.match(model)
    if match is None:
        raise ValueError(f"{model} does not look like a quantized model name.")

    _, model, quant_method, quant_format = match.groups()

    if quant_method is None and quantization is None:
        raise ValueError(
            "You must supply the quantization argument or provide quantization method in the model argument"
        )

    if quant_method and quantization:
        raise ValueError(
            "You cannot provide both the quantization argument and quantization in model"
        )

    if quant_format and format:
        raise ValueError(
            "You cannot provide both the format argument and format in model"
        )

    info = model_info(model, client=client)
    if info is None:
        raise ModelNotFound(model)

    q = quant_method if quantization is None else quantization
    model_id = f"{model}/{q}".lower()

    fmt = quant_format if format is None else format
    fmt = "gguf" if fmt is None else fmt

    quantized_files = info.pop("quantizedFiles")
    for quant in quantized_files:
        if (quant["quantMethod"].lower() == q.lower()) and (
            quant["format"].lower() == fmt.lower()
        ):
            return {**info, **quant}
    raise QuantizedFileNotFound(model_id)


class AnacondaQuantizedModelService(LlamaCPPService):
    def __init__(
        self,
        url: str,
        api_key: str = "none",
        options: Optional[Any] = None,
        metadata: Optional[Any] = None,
    ):
        super().__init__(url, options, metadata)
        self.api_key = api_key

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


class AnacondaQuantizedModelCache:
    _cache: Path

    def __init__(
        self,
        name: str,
        quantization: Optional[str] = None,
        format: Optional[str] = None,
        client: Optional[Client] = None,
        cache_path: Optional[Union[Path, str]] = None,
        directory: Optional[Union[Path, str]] = None,
    ) -> None:
        config_kwargs = {}
        if cache_path is not None:
            config_kwargs["cache_path"] = Path(cache_path)

        self.name = name
        self.client = Client() if client is None else client

        self.metadata = quantized_model_info(
            model=name, quantization=quantization, format=format, client=self.client
        )

        suffix = self.metadata["format"].lower()
        fn = f"{self.metadata['name']}_{self.metadata['quantMethod']}.{suffix}"

        if directory is not None:
            self._cache = Path(directory) / fn
        else:
            self._cache = config.cache_path / self.metadata["modelId"] / fn

    @property
    def is_cached(self) -> bool:
        if not self._cache.exists():
            return False

        size = self.metadata["sizeBytes"]
        return size == os.stat(self._cache).st_size

    @property
    def path(self) -> Path:
        return self.download()

    def download(self, force: bool = False, to_stderr: bool = True) -> Path:
        if self.is_cached and not force:
            return self._cache

        if to_stderr:
            console = Console(file=sys.stderr)
        else:
            console = Console()

        self._cache.parent.mkdir(parents=True, exist_ok=True)

        response = self.client.get(self.metadata["downloadUrl"], stream=True)
        response.raise_for_status()

        size = response.headers["Content-Length"]
        stream_progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
            rich.progress.TimeRemainingColumn(elapsed_when_finished=True),
            console=console,
            refresh_per_second=10,
        )
        task = stream_progress.add_task(f"Downloading {self.name}", total=int(size))

        with open(self._cache, "wb") as f:
            with stream_progress as s:
                for chunk in response.iter_content(1024**2):
                    f.write(chunk)
                    s.update(task, advance=len(chunk))

        return self._cache

    def start(
        self, run_on: Optional[str] = None, **kwargs: Any
    ) -> AnacondaQuantizedModelService:
        if run_on is None:
            run_on = config.run_on

        if run_on == "local":
            port = kwargs.pop("port", 0)
            if port == 0:
                port = find_free_port()

            ctx_size = kwargs.pop("ctx_size", 0)

            gguf = self.download()
            server = LlamaServerReader(gguf)

            kind = "embedding" if "embedding" in kwargs else "inference"
            log_file = f"llama-cpp.{self.metadata['name']}_{self.metadata['quantMethod']}.{kind}.log"
            llama_cpp_kwargs = {
                **kwargs,
                **{
                    "port": port,
                    "ctx-size": ctx_size,
                    "log-file": log_file,
                    "log_file": log_file,
                },
            }
            _service: LlamaCPPService = server.read(**llama_cpp_kwargs)
            service = AnacondaQuantizedModelService(
                url=_service.url, options=_service.options
            )
            return service
        elif run_on == "ai-navigator":
            from rich.console import Console
            from rich.status import Status

            ai_nav = AINavigatorClient()

            model_id = self.metadata["modelId"]
            body = {
                "modelId": model_id,
                "quant": self.metadata["quantMethod"],
                "temperature": kwargs.get("temperature", 0),
            }
            res = ai_nav.post("inference-server", json=body)
            res.raise_for_status()
            server = res.json()

            status = server.get("data", {}).get("status", "")
            text = f"{model_id} ({status})"

            console = Console(stderr=True)
            with Status(text, console=console) as display:
                while status != "running":
                    res = ai_nav.post("inference-server", json=body)
                    # res2 = client.get(f"inference-server/{server['data']['id']}")
                    res.raise_for_status()
                    server = res.json()
                    status = server.get("data", {}).get("status", "")

                    text = f"{model_id} ({status})"
                    display.update(text)
            console.print(f"[bold green]✓[/] {text}", highlight=False)
            server_id = server["data"]["id"]

            def terminate(client: AINavigatorClient, server_id: str) -> None:
                client.delete(f"inference-server/{server_id}")

            atexit.register(terminate, client=ai_nav, server_id=server_id)

            url = f"http://localhost:{server_id}"
            service = AnacondaQuantizedModelService(url=url, options=kwargs)
            return service
        else:
            raise NotImplementedError(f"run_on='{run_on}' is not supported")
