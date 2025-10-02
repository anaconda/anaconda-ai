from typing import Any
from typing import Dict
from typing import Optional

from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import LLMMetadata
from pydantic import Field

from ..clients import make_client
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
        model: str,
        site: Optional[str] = None,
        backend: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        client = make_client(site=site, backend=backend)
        server = client.servers.create(model, extra_options=extra_options)
        server.start()
        context_window = client.models.get(model).context_window_size

        super().__init__(
            model=server.serverConfig.model_name,
            api_key=server.api_key,
            api_base=server.openai_url,
            is_chat_model=True,
            api_version="empty",
            system_prompt=system_prompt,
            context_window=context_window,
            max_tokens=max_tokens,
            is_function_calling_model=False,
            temperature=temperature,
        )

        self._server_config = server.serverConfig

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
