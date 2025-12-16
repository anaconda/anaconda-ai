from copy import deepcopy
from typing import Callable, Iterator, Optional, Any, Union, cast, AsyncIterator, Tuple

import openai
import litellm
from httpx import Timeout
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, GenericStreamingChunk, EmbeddingResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from ..clients import AnacondaAIClient
from ..clients.base import GenericClient


def prepare_server(model: str, options: dict):
    kwargs = deepcopy(options)

    client_kwargs = kwargs.pop("client", {})
    if isinstance(client_kwargs, dict):
        client = AnacondaAIClient(**client_kwargs)
    elif isinstance(client_kwargs, GenericClient):
        client = client_kwargs

    server_params = kwargs.pop("server", {})

    if model.startswith("server/"):
        server_name = model.split("/", maxsplit=1)[1]
        server = client.servers.get(server_name)
    else:
        server = client.servers.create(model, extra_options=server_params)
    if not server.is_running:
        server.start()

    return server


def create_and_start(
    model: str,
    timeout: Optional[Union[float, Timeout]] = None,
    kwargs: Optional[dict] = None,
) -> Tuple[openai.OpenAI, str]:
    server = prepare_server(model, kwargs or {})
    return server.openai_client(timeout=timeout), server.config.model_name


def create_and_start_async(
    model: str,
    timeout: Optional[Union[float, Timeout]] = None,
    kwargs: Optional[dict] = None,
) -> Tuple[openai.AsyncOpenAI, str]:
    server = prepare_server(model, kwargs or {})
    return server.async_openai_client(timeout=timeout), server.config.model_name


class AnacondaLLM(CustomLLM):
    def _prepare_kwargs(self, optional_params: dict) -> dict:
        inference_kwargs = optional_params.copy()
        _ = inference_kwargs.pop("stream", None)
        _ = inference_kwargs.pop("stream_options", None)
        _ = inference_kwargs.pop("max_retries", None)
        optional_kwargs = inference_kwargs.pop("optional_params", {})
        return inference_kwargs, optional_kwargs

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Optional[AsyncHTTPHandler] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )
        response = _client.chat.completions.create(
            messages=messages, model=model_name, **inference_kwargs
        )
        mresponse = ModelResponse(**response.model_dump())
        return mresponse

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Optional[AsyncHTTPHandler] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )
        response = _client.chat.completions.create(
            messages=messages, model=model_name, stream=True, **inference_kwargs
        )
        wrapped = CustomStreamWrapper(
            custom_llm_provider="openai",
            completion_stream=response,
            model=model,
            logging_obj=logging_obj,
        )

        for chunk in wrapped:
            handled = cast(
                GenericStreamingChunk,
                wrapped.handle_openai_chat_completion_chunk(chunk),
            )
            yield handled

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Optional[AsyncHTTPHandler] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start_async(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )
        response = await _client.chat.completions.create(
            messages=messages, model=model_name, **inference_kwargs
        )
        mresponse = ModelResponse(**response.model_dump())
        return mresponse

    async def astreaming(  # type: ignore
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Optional[AsyncHTTPHandler] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start_async(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )

        response = await _client.chat.completions.create(
            messages=messages, model=model_name, stream=True, **inference_kwargs
        )
        wrapped = CustomStreamWrapper(
            custom_llm_provider="openai",
            completion_stream=response,
            model=model,
            logging_obj=logging_obj,
        )

        async for chunk in wrapped:
            handled = cast(
                GenericStreamingChunk,
                wrapped.handle_openai_chat_completion_chunk(chunk),
            )
            yield handled

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )
        response = _client.embeddings.create(
            input=input, model=model_name, **inference_kwargs
        )
        eresponse = EmbeddingResponse(**response.model_dump())
        return eresponse

    async def aembedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        inference_kwargs, optional_kwargs = self._prepare_kwargs(optional_params)
        _client, model_name = create_and_start_async(
            model=model, timeout=timeout, kwargs=optional_kwargs
        )

        response = await _client.embeddings.create(
            input=input, model=model_name, **inference_kwargs
        )
        eresponse = EmbeddingResponse(**response.model_dump())
        return eresponse


# This should be moved to an entrypoint if implemented
# https://github.com/BerriAI/litellm/issues/7733
anaconda_llm = AnacondaLLM()
litellm.custom_provider_map.append(
    {"provider": "anaconda", "custom_handler": anaconda_llm}
)
