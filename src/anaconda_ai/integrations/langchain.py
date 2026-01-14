from typing import Any
from typing import Dict
from typing import Optional

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai.llms.base import BaseOpenAI
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator

from ..clients import AnacondaAIClient
from ..clients.base import GenericClient, Server


def _prepare_model(
    model_name: str, values: dict, client: Optional[GenericClient] = None
) -> dict:
    extra_options = values.get("extra_options", {})

    if client is None:
        client = AnacondaAIClient()

    server = client.servers.create(model=model_name, extra_options=extra_options)

    if not server.is_running:
        server.start()

    if "model_name" in values:
        values["model_name"] = server.config.model_name
    elif "model" in values:
        values["model"] = server.config.model_name
    values["server"] = server
    values["openai_api_base"] = server.openai_url
    return values


class AnacondaQuantizedLLM(BaseOpenAI):
    model: str = Field(..., alias="model_name")
    """Anaconda quantized model to use."""
    extra_options: Dict[str, Any] = Field(default_factory=dict)
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    server: Optional[Server] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anaconda-llm"

    @model_validator(mode="before")
    @classmethod
    def prepare(cls, values: Dict) -> Dict:
        raise NotImplementedError(
            "llama.cpp does not currently support v1/completions. "
            "See https://github.com/ggerganov/llama.cpp/discussions/9219"
        )
        model_name = values.get("model") or values.get("model_name")
        if not model_name:
            raise ValueError("model_name is required")
        else:
            return _prepare_model(model_name, values)

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {**self._default_params, "model": self.model}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **super()._identifying_params,
        }


class AnacondaQuantizedModelChat(BaseChatOpenAI):
    model: str = Field(..., alias="model_name")
    """Anaconda quantized model to use."""
    extra_options: Dict[str, Any] = Field(default_factory=dict)
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    server: Optional[Server] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anaconda-chat"

    @model_validator(mode="before")
    @classmethod
    def prepare(cls, values: Dict) -> Dict:
        model_name = values.get("model") or values.get("model_name")
        if not model_name:
            raise ValueError("model_name is required")
        else:
            return _prepare_model(model_name, values, client=values.get("client"))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **super()._identifying_params,
        }


class AnacondaQuantizedModelEmbeddings(OpenAIEmbeddings):
    model_name: str = Field(..., alias="model")
    """Anaconda quantized model to use."""
    extra_options: Dict[str, Any] = Field(default_factory=dict)
    check_embedding_ctx_length: bool = False
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    server: Optional[Server] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True

    @model_validator(mode="before")
    @classmethod
    def prepare(cls, values: Dict) -> Dict:
        model_name = values.get("model") or values.get("model_name")
        if not model_name:
            raise ValueError("model_name is required")
        else:
            return _prepare_model(model_name, values, client=values.get("client"))
