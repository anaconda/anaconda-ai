from typing import Any
from typing import Dict
from typing import Optional

from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.callbacks.base import CallbackManager
from pydantic import Field

from ..clients import AnacondaAIClient
from ..clients.base import ServerConfig


class AnacondaLLMMetadata(LLMMetadata):
    server_config: Dict[str, Any]


class AnacondaModel(OpenAI):
    """Download and run a model from Anaconda"""

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )

    _tokenizer: None = None
    tokenizer: None = None
    _server_config: ServerConfig

    def __init__(
        self,
        model: Optional[str] = None,
        server_name: Optional[str] = None,
        site: Optional[str] = None,
        backend: Optional[str] = None,
        client: Optional[AnacondaAIClient] = None,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        client = client or AnacondaAIClient(site=site, backend=backend)
        if model and server_name:
            raise ValueError("You can only have one of model= or server_name=")
        if server_name:
            server = client.servers.get(server_name)
        elif model:
            server = client.servers.create(model, extra_options=extra_options)
        else:
            raise ValueError("You must specify one of model= or server_name=")

        server.start()
        context_window = client.models.get(server.config.model_name).context_window_size

        super().__init__(
            model=server.config.model_name,
            async_openai_client=server.async_openai_client(),
            openai_client=server.openai_client(),
            reuse_client=True,
            is_chat_model=True,
            api_version="empty",
            system_prompt=system_prompt,
            context_window=context_window,
            max_tokens=max_tokens,
            is_function_calling_model=True,
            temperature=temperature,
            **kwargs,
        )

        self._server_config = server.config

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AnacondaModels"

    @property
    def metadata(self) -> AnacondaLLMMetadata:
        server_config = self._server_config.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude={"logsDir", "modelFileName"},
        )

        return AnacondaLLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            server_config=server_config,
        )


class AnacondaEmbeddingModel(OpenAIEmbedding):
    _server_config: ServerConfig

    def __init__(
        self,
        model_name: str,
        site: Optional[str] = None,
        backend: Optional[str] = None,
        client: Optional[AnacondaAIClient] = None,
        embed_batch_size: int = 10,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        num_workers: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # ensure model is not passed in kwargs, will cause error in parent class
        if "model" in kwargs:
            raise ValueError(
                "Use `model_name` instead of `model` to initialize OpenAILikeEmbedding"
            )

        client = client or AnacondaAIClient(site=site, backend=backend)
        server = client.servers.create(model_name, extra_options=extra_options)
        server.start()

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            additional_kwargs=additional_kwargs,
            api_key=server.api_key,
            api_base=server.openai_url,
            api_version="empty",
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            num_workers=num_workers,
            **kwargs,
        )

        self._server_config = server.serverConfig
