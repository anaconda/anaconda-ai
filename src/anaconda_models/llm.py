import atexit
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional

import llm
import openai
from llm import hookimpl
from llm.default_plugins.openai_models import Chat
from llm.default_plugins.openai_models import OpenAIEmbeddingModel
from rich.console import Console
from rich.status import Status

from anaconda_models.clients import get_default_client
from anaconda_models.clients.base import ModelQuantization, Server

client = get_default_client()
console = Console(stderr=True)


class AnacondaModelMixin:
    model_id: str
    anaconda_model: ModelQuantization | None = None
    server: Server | None = None

    def _create_and_start(self, embedding: bool = False) -> None:
        if self.server is None:
            model_name = self.model_id.split(":", maxsplit=1)[1]
            text = f"{model_name} (creating)"
            with Status(text, console=console) as display:
                client = get_default_client()
                self.server = client.servers.create(model_name)
                status = client.servers.start(self.server)
                text = f"{model_name} ({status.status})"
                display.update(text)

                while status.status != "running":
                    status = client.servers.start(self.server)
                    text = f"{model_name} ({status.status})"
                    display.update(text)
            console.print(f"[bold green]✓[/] {text}", highlight=False)
            atexit.register(client.servers.stop, self.server)

        self.api_base = self.server.openai_url


class AnacondaQuantizedChat(Chat, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""

    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            key="none",
            model_name=model_id,
        )

    def execute(self, prompt, stream, response, conversation=None, key=None):  # type: ignore
        self._create_and_start(embedding=False)
        return super().execute(prompt, stream, response, conversation, key)

    def __str__(self) -> str:
        return f"Anaconda Model Chat: {self.model_id}"


class AnacondaQuantizedEmbedding(OpenAIEmbeddingModel, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""
    batch_size: int = 100

    def __init__(self, model_id: str, dimensions: Optional[Any] = None) -> None:
        super().__init__(model_id, openai_model_id=model_id, dimensions=dimensions)

    def embed_batch(self, items: Iterable[str | bytes]) -> Iterator[List[float]]:
        self._create_and_start(embedding=True)
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
    variants = []

    for id_ in model_id, model_id.lower(), model_name, model_name.lower():
        for method_ in method.upper(), method.lower():
            for format_ in format.upper(), format.lower():
                for sep in "_", "/":
                    v = f"{id_}{sep}{method_}.{format_}"
                    variants.append(v)

    return variants


@llm.hookimpl
def register_models(register: Callable) -> None:
    for model in client.models.list():
        if model.metadata.trainedFor != "text-generation":
            continue
        for quant in model.metadata.quantizations:
            if not quant.isDownloaded:
                continue

            quant_chat = AnacondaQuantizedChat(
                model_id=f"anaconda:{quant.modelFileName}"
            )
            register(quant_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    for model in client.models.list():
        if model.trainedFor != "sentence-similarity":
            continue
        for quant in sorted(model.quantizedFiles, key=lambda q: q.quantMethod):
            method = quant.quantMethod
            format = quant.format

            # file_ids = _accepted_model_name_variants(
            #     model_id, model_name, method, format
            # )
            file_ids = [f"{model.name}_{method}.{format.lower()}"]
            for file_id in file_ids:
                embed = AnacondaQuantizedEmbedding(model_id=f"anaconda:{file_id}")
                register(embed)
