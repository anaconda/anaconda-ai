import atexit
from typing import Optional

from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import LLMMetadata
from pydantic import Field

from anaconda_models.clients import get_default_client


class AnacondaModel(OpenAI):
    """Download and run a model from Anaconda"""

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )

    _tokenizer: None = None
    tokenizer: None = None

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> None:
        client = get_default_client()
        server = client.servers.create(model)
        server.start()
        if not server._matched:
            atexit.register(server.stop)

        context_window = client.models.get(model).metadata.contextWindowSize

        super().__init__(
            model=model,
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

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AnacondaModels"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )
