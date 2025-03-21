import atexit
from typing import Callable, Iterator, Optional, Any, Union, cast, AsyncIterator

import litellm
from httpx import Timeout
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, GenericStreamingChunk
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from anaconda_models.clients import get_default_client

client = get_default_client()


def create_and_start(model: str, timeout: Optional[Union[float, Timeout]] = None):
    server = client.servers.create(model)
    server.start()
    if not server._matched:
        atexit.register(server.stop)

    return server.openai_client(timeout=timeout)


async def async_create_and_start(
    model: str, timeout: Optional[Union[float, Timeout]] = None
):
    server = client.servers.create(model)
    server.start()
    if not server._matched:
        atexit.register(server.stop)

    return server.openai_async_client(timeout=timeout)


class AnacondaLLM(CustomLLM):
    def _prepare_inference_kwargs(self, optional_params: dict) -> dict:
        inference_kwargs = optional_params.copy()
        _ = inference_kwargs.pop("stream", None)
        _ = inference_kwargs.pop("stream_options", None)
        _ = inference_kwargs.pop("max_retries", None)
        return inference_kwargs

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
        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        _client = create_and_start(model=model, timeout=timeout)
        response = _client.chat.completions.create(
            messages=messages, model=model, **inference_kwargs
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
        _client = create_and_start(model=model, timeout=timeout)
        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        response = _client.chat.completions.create(
            messages=messages, model=model, stream=True, **inference_kwargs
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
        _client = await async_create_and_start(model=model, timeout=timeout)
        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        response = await _client.chat.completions.create(
            messages=messages, model=model, **inference_kwargs
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
        _client = await async_create_and_start(model=model, timeout=timeout)

        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        response = await _client.chat.completions.create(
            messages=messages, model=model, stream=True, **inference_kwargs
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


# This should be moved to an entrypoint if implemented
# https://github.com/BerriAI/litellm/issues/7733
anaconda_llm = AnacondaLLM()
litellm.custom_provider_map.append(
    {"provider": "anaconda", "custom_handler": anaconda_llm}
)
