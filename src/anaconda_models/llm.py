from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from urllib.parse import urljoin

import click
import llm
import openai
from intake.readers.datatypes import LlamaCPPService
from llm import hookimpl
from llm.default_plugins.openai_models import Chat
from llm.default_plugins.openai_models import OpenAIEmbeddingModel

from anaconda_models.core import AnacondaQuantizedModelCache
from anaconda_models.core import get_models


class AnacondaModelMixin:
    model_id: str
    client_options: dict | None = None
    anaconda_model: AnacondaQuantizedModelCache | None = None
    llama_cpp_options: dict | None = None
    llama_cpp_service: LlamaCPPService | None = None

    def _get_or_create_service(self, embedding: bool = False) -> None:
        if self.anaconda_model is None:
            client_kwargs = {} if self.client_options is None else self.client_options
            _, model_name = self.model_id.split("anaconda:")
            self.anaconda_model = AnacondaQuantizedModelCache(
                model_name, **client_kwargs
            )

        if (self.llama_cpp_service is None) or (
            self.llama_cpp_service.options["Process"].poll() is not None
        ):
            llama_cpp_kwargs = (
                {} if self.llama_cpp_options is None else self.llama_cpp_options
            )
            if embedding:
                llama_cpp_kwargs["embedding"] = None
                llama_cpp_kwargs["pooling"] = "mean"

            self.llama_cpp_service = self.anaconda_model.start(**llama_cpp_kwargs)

        self.api_base = urljoin(self.llama_cpp_service.url, "/v1")


class AnacondaQuantizedChat(Chat, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""

    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            key="none",
            model_name=model_id,
        )

    def execute(self, prompt, stream, response, conversation=None):  # type: ignore
        self._get_or_create_service(embedding=False)
        return super().execute(prompt, stream, response, conversation)

    def __str__(self) -> str:
        return f"Anaconda Quantized Chat: {self.model_id}"


class AnacondaQuantizedEmbedding(OpenAIEmbeddingModel, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""
    batch_size: int = 100

    def __init__(self, model_id: str, dimensions: Optional[Any] = None) -> None:
        super().__init__(model_id, openai_model_id=model_id, dimensions=dimensions)

    def embed_batch(self, items: Iterable[str | bytes]) -> Iterator[List[float]]:
        self._get_or_create_service(embedding=True)
        kwargs = {
            "input": items,
            "model": self.openai_model_id,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        client = openai.OpenAI(api_key="none", base_url=self.api_base)
        results = client.embeddings.create(**kwargs).data
        return ([float(r) for r in result.embedding] for result in results)


def _accepted_model_name_variants(
    model_id: str, model_name: str, method: str, format: str
) -> List[str]:
    variants = [
        f"{model_id}_{method}.{format.lower()}",
        f"{model_id}_{method.lower()}.{format.lower()}",
        f"{model_id}_{method}.{format}".lower(),
        f"{model_id}/{method}.{format.lower()}",
        f"{model_id}/{method.lower()}.{format.lower()}",
        f"{model_id}/{method}.{format}".lower(),
        f"{model_name}_{method}.{format.lower()}",
        f"{model_name}_{method.lower()}.{format.lower()}",
        f"{model_name}_{method}.{format}".lower(),
        f"{model_name}/{method}.{format.lower()}",
        f"{model_name}/{method.lower()}.{format.lower()}",
        f"{model_name}/{method}.{format}".lower(),
    ]
    return variants


@llm.hookimpl
def register_models(register: Callable) -> None:
    for model in get_models():
        model_id = model["modelId"]
        model_name = model["name"]
        for quant in sorted(model["quantizedFiles"], key=lambda q: q["quantMethod"]):
            method = quant["quantMethod"]
            format = quant["format"]

            file_ids = _accepted_model_name_variants(
                model_id, model_name, method, format
            )
            for file_id in file_ids:
                quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
                register(quant_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    for model in get_models():
        model_id = model["modelId"]
        model_name = model["name"]
        for quant in sorted(model["quantizedFiles"], key=lambda q: q["quantMethod"]):
            method = quant["quantMethod"]
            format = quant["format"]

            file_ids = _accepted_model_name_variants(
                model_id, model_name, method, format
            )
            for file_id in file_ids:
                embed = AnacondaQuantizedEmbedding(model_id=f"anaconda:{file_id}")
                register(embed)


@hookimpl
def register_commands(cli: click.Group) -> None:
    @cli.group(name="anaconda")
    def anaconda_() -> None:
        "Commands for working directly with Anaconda models"

    @anaconda_.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    @click.option("--key", help="Anaconda.cloud API key")
    def models(json_: bool, key: str) -> None:
        from llm.cli import get_key
        from rich.table import Table, Column
        from rich.console import Console

        from anaconda_models.client import Client

        api_key = get_key(key, "anaconda", "ANACONDA_CLOUD_API_KEY")

        client = Client(api_key=api_key)
        models = get_models(client=client)

        console = Console()
        table = Table(
            Column("Model", no_wrap=True),
            "Method\n(downloaded in bold)",
            "Max Ram (GB)",
            "Size (GB)",
            header_style="bold green",
        )

        quantized = []
        for model in models:
            model_id = model["modelId"]
            quantized_files = model.pop("quantizedFiles")
            for qf in sorted(quantized_files, key=lambda q: q["quantMethod"]):
                method = qf["quantMethod"]
                format = qf["format"]
                file_id = f"anaconda:{model_id}_{method}.{format.lower()}"
                qf["model_id"] = file_id
                quant = {**qf, **model}
                quantized.append(quant)

                cacher = AnacondaQuantizedModelCache(
                    f"{model_id}/{method}", client=client
                )
                if cacher.is_cached:
                    method = f"[bold green]{method}[/bold green]"

                ram = f"{quant['maxRamUsage'] / 1024 / 1024 / 1024:.2f}"
                size = f"{quant['sizeBytes'] / 1024 / 1024 / 1024:.2f}"
                table.add_row(file_id, method, ram, size)
            table.add_section()

        if json_:
            console.print_json(data=quantized)
        else:
            console.print(table)
