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
from llm.default_plugins.openai_models import Chat, AsyncChat
from llm.default_plugins.openai_models import OpenAIEmbeddingModel
from pydantic import ConfigDict
from rich.console import Console

from ..clients import AnacondaAIClient
from ..clients.base import QuantizedFile, Server

console = Console(stderr=True)


class AnacondaOptions(llm.Options):
    backend: Optional[str] = None
    site: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class AnacondaModelMixin:
    model_id: str
    anaconda_model: Optional[QuantizedFile] = None
    server: Optional[Server] = None

    def _create_and_start(
        self, options: Optional[AnacondaOptions] = None, key: Optional[str] = None
    ) -> None:
        if self.server is None:
            options = options or AnacondaOptions()

            kwargs = {}
            if key is not None:
                kwargs["api_key"] = key

            client = AnacondaAIClient(
                site=options.site, backend=options.backend, **kwargs
            )

            _, model_name = self.model_id.split(":", maxsplit=1)

            if model_name.startswith("server/"):
                _, server_name = model_name.split("server/", maxsplit=1)
                self.server = client.servers.get(server_name)
            else:
                self.server = client.servers.create(
                    model=model_name, extra_options=options.model_dump()
                )

        if not self.server.is_running:
            self.server.start(console=console)

        self.api_base = self.server.openai_url


class AnacondaQuantizedChat(Chat, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""

    class Options(Chat.Options, AnacondaOptions):
        model_config = ConfigDict(extra="allow")

    def __init__(self, model_id: str):
        super().__init__(
            model_id, model_name=model_id.replace("anaconda:", ""), supports_tools=True
        )

    def get_key(self, _: Optional[str] = None) -> Optional[str]:
        return None

    def get_client(self, _, *, async_=False):
        if async_:
            return self.server.async_openai_client()
        else:
            return self.server.openai_client()

    def execute(self, prompt, stream, response, conversation=None, key=None):  # type: ignore
        self._create_and_start(options=prompt.options, key=key)
        include = set(Chat.Options.model_fields.keys())
        prompt.options = Chat.Options(**prompt.options.model_dump(include=include))
        return super().execute(prompt, stream, response, conversation, key)

    def __str__(self) -> str:
        return f"Anaconda Model Chat: {self.model_id}"


class AsyncAnacondaQuantizedChat(AsyncChat, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""

    class Options(Chat.Options, AnacondaOptions):
        model_config = ConfigDict(extra="allow")

    def __init__(self, model_id: str):
        super().__init__(
            model_id, model_name=model_id.replace("anaconda:", ""), supports_tools=True
        )

    def get_client(self, _, *, async_=False):
        if async_:
            return self.server.async_openai_client()
        else:
            return self.server.openai_client()

    async def execute(self, prompt, stream, response, conversation=None, key=None):  # type: ignore
        self._create_and_start(options=prompt.options, key=key)
        include = set(Chat.Options.model_fields.keys())
        prompt.options = Chat.Options(**prompt.options.model_dump(include=include))
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


def create_and_validate_client() -> AnacondaAIClient:
    from requests import HTTPError

    client = AnacondaAIClient()
    try:
        client.account
        return client
    except HTTPError:
        return None


@llm.hookimpl
def register_models(register: Callable) -> None:
    client = create_and_validate_client()
    if client is None:
        return
    for model in client.models.list():
        if model.trained_for != "text-generation":
            continue
        for quant in model.quantized_files:
            alias = f"anaconda:{quant.identifier}"
            quant_chat = AnacondaQuantizedChat(model_id=alias)
            async_quant_chat = AsyncAnacondaQuantizedChat(model_id=alias)
            register(quant_chat, async_quant_chat)
    for server in client.servers.list():
        alias = f"anaconda:server/{server.id}"
        server_chat = AnacondaQuantizedChat(model_id=alias)
        async_server_chat = AsyncAnacondaQuantizedChat(model_id=alias)
        register(server_chat, async_server_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    client = create_and_validate_client()
    if client is None:
        return
    for model in client.models.list():
        if model.trained_for != "sentence-similarity":
            continue
        for quant in model.quantized_files:
            quant_chat = AnacondaQuantizedEmbedding(
                model_id=f"anaconda:{quant.identifier}"
            )
            register(quant_chat)
