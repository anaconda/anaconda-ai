from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import llm
import openai
from llm import hookimpl
from llm.default_plugins.openai_models import Chat
from llm.default_plugins.openai_models import OpenAIEmbeddingModel
from rich.console import Console

from ..clients import make_client
from ..clients.base import QuantizedFile, Server

console = Console(stderr=True)


class ServerOptions(llm.Options):
    backend: Optional[str] = None


class AnacondaModelMixin:
    model_id: str
    anaconda_model: Optional[QuantizedFile] = None
    server: Optional[Server] = None

    def _create_and_start(self, options: Optional[ServerOptions] = None) -> None:
        if self.server is None:
            model_name = self.model_id.split(":", maxsplit=1)[1]

            if options is None:
                options = ServerOptions()

            client = make_client(options.backend)

            self.server = client.servers.create(
                model=model_name,
            )
            self.server.start(console=console)

        self.api_base = self.server.openai_url


class AnacondaQuantizedChat(Chat, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""

    class Options(Chat.Options, ServerOptions):
        pass

    def __init__(self, model_id: str):
        super().__init__(
            model_id, key="none", model_name=model_id.replace("anaconda:", "")
        )

    def execute(self, prompt, stream, response, conversation=None, key=None):  # type: ignore
        self._create_and_start(options=prompt.options)
        prompt.options = Chat.Options(
            **prompt.options.model_dump(exclude=ServerOptions.model_fields.keys())
        )
        return super().execute(prompt, stream, response, conversation, key)

    def __str__(self) -> str:
        return f"Anaconda Model Chat: {self.model_id}"


class AnacondaQuantizedEmbedding(OpenAIEmbeddingModel, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""
    batch_size: int = 100

    def __init__(self, model_id: str, dimensions: Optional[Any] = None) -> None:
        super().__init__(
            model_id,
            openai_model_id=model_id.replace("anaconda:", ""),
            dimensions=dimensions,
        )

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        self._create_and_start()
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
    client = make_client()
    for model in client.models.list():
        if model.trained_for != "text-generation":
            continue
        for quant in model.quantized_files:
            if not quant.is_downloaded:
                continue

            quant_chat = AnacondaQuantizedChat(model_id=f"anaconda:{quant.identifier}")
            register(quant_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    client = make_client()
    for model in client.models.list():
        if model.trained_for != "sentence-similarity":
            continue
        for quant in model.quantized_files:
            if not quant.is_downloaded:
                continue

            quant_chat = AnacondaQuantizedEmbedding(
                model_id=f"anaconda:{quant.identifier}"
            )
            register(quant_chat)
