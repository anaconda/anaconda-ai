from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import cast
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
                llama_cpp_kwargs["embedding"] = ""

            self.llama_cpp_service = self.anaconda_model.start()

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


@llm.hookimpl
def register_models(register: Callable) -> None:
    for model in get_models():
        model_id = model["modelId"]
        model_name = model["name"]
        for quant in model["quantizedFiles"]:
            method = quant["quantMethod"]
            format = quant["format"]

            file_id = f"{model_id}_{method}.{format}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}/{method}.{format}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}_{method}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}/{method}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}/{method}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}_{method}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}/{method}.{format}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}_{method}.{format}"
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)

            file_id = f"{model_id}_{method}.{format}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}/{method}.{format}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}_{method}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_id}/{method}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}/{method}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}_{method}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}/{method}.{format}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)
            file_id = f"{model_name}_{method}.{format}".lower()
            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{file_id}")
            register(quant_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    for model in get_models():
        model_id = model["modelId"]
        for quant in model["quantizedFiles"]:
            method = quant["quantMethod"]
            format = quant["format"]
            file_id = f"{model_id}_{method.lower()}.{format.lower()}"

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
        from llm.utils import dicts_to_table_string

        from anaconda_models.client import Client

        api_key = get_key(key, "anaconda", "ANACONDA_CLOUD_API_KEY")

        client = Client(api_key=api_key)
        models = get_models(client=client)

        quantized = []
        for model in models:
            model_id = model["modelId"]
            quantized_files = model.pop("quantizedFiles")
            for qf in quantized_files:
                method = qf["quantMethod"]
                format = qf["format"]
                file_id = f"anaconda:{model_id}_{method}.{format.lower()}"
                qf["model_id"] = file_id
                quant = {**qf, **model}
                quantized.append(quant)
        quantized = sorted(quantized, key=lambda m: m["model_id"])

        if json_:
            import json

            import click

            click.echo(json.dumps(models, indent=4))
        else:
            to_print = []
            for model in quantized:
                to_print.append(
                    {
                        "id": model["model_id"],
                        "method": model["quantMethod"],
                        "size": f"{model['sizeBytes'] / 1024 / 1024 / 1024:.2f} GB",
                        "types": "chat, embedding",
                        "license": model["license"],
                    }
                )
            headers = cast(List[str], "id method size types license".split())
            done = dicts_to_table_string(headers, to_print)
            print("\n".join(done))
