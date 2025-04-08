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

from ..clients import get_default_client
from ..clients.base import ModelQuantization, Server

client = get_default_client()
console = Console(stderr=True)


class ServerOptions(llm.Options):
    api_key: Optional[str] = None
    log_disable: Optional[bool] = None
    mmproj: Optional[str] = None
    timeout: Optional[int] = None
    verbose: Optional[bool] = None
    main_gpu: Optional[int] = None
    metrics: Optional[bool] = None
    batch_size: Optional[int] = None
    cont_batching: Optional[bool] = None
    ctx_size: Optional[int] = None
    memory_f32: Optional[bool] = None
    mlock: Optional[bool] = None
    n_gpu_layers: Optional[int] = None
    rope_freq_base: Optional[int] = None
    rope_freq_scale: Optional[int] = None
    seed: Optional[int] = None
    tensor_split: Optional[List[int]] = None
    use_mmap: Optional[bool] = None
    embedding: Optional[bool] = None
    threads: Optional[int] = None
    n_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repeat_last: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temp: Optional[float] = None
    parallel: Optional[int] = None

    @property
    def api_params(self) -> dict:
        return {
            "api_key": self.api_key,
            "log_disable": self.log_disable,
            "mmproj": self.mmproj,
            "timeout": self.timeout,
            "verbose": self.verbose,
            "metrics": self.metrics,
        }

    @property
    def load_params(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "cont_batching": self.cont_batching,
            "ctx_size": self.ctx_size,
            "main_gpu": self.main_gpu,
            "memory_f32": self.memory_f32,
            "mlock": self.mlock,
            "n_gpu_layers": self.n_gpu_layers,
            "rope_freq_base": self.rope_freq_base,
            "rope_freq_scale": self.rope_freq_scale,
            "seed": self.seed,
            "tensor_split": self.tensor_split,
            "use_mmap": self.use_mmap,
            "embedding": self.embedding,
        }

    @property
    def infer_params(self) -> dict:
        return {
            "threads": self.threads,
            "n_predict": self.n_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_last": self.repeat_last,
            "repeat_penalty": self.repeat_penalty,
            "temp": self.temp,
            "parallel": self.parallel,
        }


class AnacondaModelMixin:
    model_id: str
    anaconda_model: Optional[ModelQuantization] = None
    server: Optional[Server] = None

    def _create_and_start(
        self, embedding: bool, options: Optional[ServerOptions] = None
    ) -> None:
        if self.server is None:
            model_name = self.model_id.split(":", maxsplit=1)[1]

            if options is None:
                options = ServerOptions()

            load_params = options.load_params
            load_params["embedding"] = embedding
            self.server = client.servers.create(
                model=model_name,
                api_params=options.api_params,
                load_params=options.load_params,
                infer_params=options.infer_params,
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
        self._create_and_start(embedding=False, options=prompt.options)
        prompt.options = Chat.Options(
            **prompt.options.model_dump(exclude=ServerOptions().model_fields.keys())
        )
        return super().execute(prompt, stream, response, conversation, key)

    def __str__(self) -> str:
        return f"Anaconda Model Chat: {self.model_id}"


class AnacondaQuantizedEmbedding(OpenAIEmbeddingModel, AnacondaModelMixin):
    model_id: str
    needs_key: str = ""
    batch_size: int = 100

    def __init__(self, model_id: str, dimensions: Optional[Any] = None) -> None:
        super().__init__(model_id, openai_model_id=model_id, dimensions=dimensions)

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
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


@llm.hookimpl
def register_models(register: Callable) -> None:
    for model in client.models.list():
        if model.metadata.trainedFor != "text-generation":
            continue
        for quant in model.metadata.files:
            if not quant.isDownloaded:
                continue

            quant_chat = AnacondaQuantizedChat(
                model_id=f"anaconda:{quant.modelFileName}"
            )
            register(quant_chat)


@hookimpl
def register_embedding_models(register: Callable) -> None:
    for model in client.models.list():
        if model.metadata.trainedFor != "sentence-similarity":
            continue
        for quant in model.metadata.files:
            if not quant.isDownloaded:
                continue

            quant_chat = AnacondaQuantizedEmbedding(
                model_id=f"anaconda:{quant.modelFileName}"
            )
            register(quant_chat)
