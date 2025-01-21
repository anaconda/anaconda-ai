from typing import Any
from typing import Dict
from urllib.parse import urljoin

from intake.readers.datatypes import LlamaCPPService
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai.llms.base import BaseOpenAI
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator

from anaconda_models.client import Client
from anaconda_models.core import AnacondaQuantizedModelCache


def _prepare_model(model_name: str, values: dict, embedding: bool = False) -> dict:
    client = Client(**values.get("anaconda_client_options", {}))
    model = AnacondaQuantizedModelCache(
        name=model_name,
        quantization=values.get("quantization"),
        format=values.get("format"),
        client=client,
    )
    llama_cpp_kwargs = values.get("llama_cpp_options", {})
    if embedding:
        llama_cpp_kwargs["embedding"] = None

    service = model.start(run_on=values.get("run_on", "local"), **llama_cpp_kwargs)

    base_url = urljoin(service.url, "/v1")

    values["anaconda_model"] = model
    values["llama_cpp_service"] = service
    values["openai_api_base"] = base_url
    return values


class AnacondaQuantizedLLM(BaseOpenAI):
    model: str = Field(..., alias="model_name")
    """model to use."""
    quantization: str | None = None
    format: str | None = None
    run_on: str = "local"
    anaconda_client_options: dict = Field(default={})
    """Options passed to Anaconda.cloud client"""
    llama_cpp_options: dict = Field(default={})
    """Options passed to llama.cpp"""
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    """Set to 'none' because llama.cpp is running locally"""
    llama_cpp_service: LlamaCPPService = Field(..., exclude=True)
    anaconda_model: AnacondaQuantizedModelCache = Field(..., exclude=True)

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
            "anaconda_metadata": self.anaconda_model.metadata,
            "llama_cpp_options": self.llama_cpp_options,
        }


class AnacondaQuantizedModelChat(BaseChatOpenAI):
    model: str = Field(..., alias="model_name")
    """model to use."""
    quantization: str | None = None
    format: str | None = None
    run_on: str = "local"
    anaconda_client_options: dict = Field(default={})
    """Options passed to Anaconda.cloud client"""
    llama_cpp_options: dict = Field(default={})
    """Options passed to llama.cpp"""
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    """Set to 'none' because llama.cpp is running locally"""
    llama_cpp_service: LlamaCPPService = Field(..., exclude=True)
    anaconda_model: AnacondaQuantizedModelCache = Field(..., exclude=True)

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
            return _prepare_model(model_name, values)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **super()._identifying_params,
            "anaconda_metadata": self.anaconda_model.metadata,
            "llama_cpp_options": self.llama_cpp_options,
        }


class AnacondaQuantizedModelEmbeddings(OpenAIEmbeddings):
    model_name: str = Field(..., alias="model")
    """model to use."""
    quantization: str | None = None
    format: str | None = None
    run_on: str = "local"
    check_embedding_ctx_length: bool = False
    anaconda_client_options: dict = Field(default={})
    """Options passed to Anaconda.cloud client"""
    llama_cpp_options: dict = Field(default={})
    """Options passed to llama.cpp"""
    openai_api_key: SecretStr = Field(default=SecretStr("none"), alias="api_key")
    """Set to 'none' because llama.cpp is running locally"""
    llama_cpp_service: LlamaCPPService = Field(..., exclude=True)
    anaconda_model: AnacondaQuantizedModelCache = Field(..., exclude=True)

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
            return _prepare_model(model_name, values, embedding=True)
