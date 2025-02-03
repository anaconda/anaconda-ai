from typing import Optional, Dict, Any

from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import LLMMetadata
from pydantic import Field

from anaconda_models.core import AnacondaQuantizedModelCache
from anaconda_models.client import Client


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
        quantization: Optional[str] = None,
        format: Optional[str] = None,
        system_prompt: Optional[str] = None,
        client: Optional[Client] = None,
        llama_cpp_kwargs: Optional[Dict[str, Any]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> None:
        cacher = AnacondaQuantizedModelCache(
            name=model, quantization=quantization, format=format, client=client
        )

        kwargs = {} if llama_cpp_kwargs is None else llama_cpp_kwargs
        service = cacher.start(**kwargs)

        super().__init__(
            model=model,
            api_key="empty",
            api_base=service.openai_url,
            is_chat_model=True,
            api_version="empty",
            system_prompt=system_prompt,
            context_window=cacher.metadata["contextWindowSize"],
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
