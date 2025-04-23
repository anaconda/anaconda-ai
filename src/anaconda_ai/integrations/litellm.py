from typing import Callable, Iterator, Optional, Any, Union, cast, AsyncIterator, Tuple

import openai
import litellm
from httpx import Timeout
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, GenericStreamingChunk
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from ..clients import get_default_client

client = get_default_client()


def create_and_start(
    model: str, timeout: Optional[Union[float, Timeout]] = None, **kwargs: Any
) -> Tuple[openai.OpenAI, str]:
    server = client.servers.create(model, **kwargs)
    server.start()
    return server.openai_client(timeout=timeout), server.serverConfig.modelFileName


async def async_create_and_start(
    model: str, timeout: Optional[Union[float, Timeout]] = None, **kwargs: Any
) -> Tuple[openai.AsyncOpenAI, str]:
    server = client.servers.create(model, **kwargs)
    server.start()
    return server.openai_async_client(
        timeout=timeout
    ), server.serverConfig.modelFileName


class AnacondaLLM(CustomLLM):
    def _prepare_inference_kwargs(self, optional_params: dict) -> dict:
        inference_kwargs = optional_params.copy()
        _ = inference_kwargs.pop("stream", None)
        _ = inference_kwargs.pop("stream_options", None)
        _ = inference_kwargs.pop("max_retries", None)
        _ = inference_kwargs.pop("optional_params", None)
        return inference_kwargs

    def _prepare_server_kwargs(self, optional_params: dict) -> dict:
        optional = optional_params.get("optional_params", {})
        api_params = optional.get("api_params", None)
        load_params = optional.get("load_params", None)
        infer_params = optional.get("infer_params", None)
        return {
            "api_params": api_params,
            "load_params": load_params,
            "infer_params": infer_params,
        }

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
        server_kwargs = self._prepare_server_kwargs(optional_params)
        _client, model_name = create_and_start(
            model=model, timeout=timeout, **server_kwargs
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
        server_kwargs = self._prepare_server_kwargs(optional_params)
        _client, model_name = create_and_start(
            model=model, timeout=timeout, **server_kwargs
        )
        inference_kwargs = self._prepare_inference_kwargs(optional_params)
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
        server_kwargs = self._prepare_server_kwargs(optional_params)
        _client, model_name = await async_create_and_start(
            model=model, timeout=timeout, **server_kwargs
        )
        inference_kwargs = self._prepare_inference_kwargs(optional_params)
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
        server_kwargs = self._prepare_server_kwargs(optional_params)
        _client, model_name = await async_create_and_start(
            model=model, timeout=timeout, **server_kwargs
        )

        inference_kwargs = self._prepare_inference_kwargs(optional_params)
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


# This should be moved to an entrypoint if implemented
# https://github.com/BerriAI/litellm/issues/7733
anaconda_llm = AnacondaLLM()
litellm.custom_provider_map.append(
    {"provider": "anaconda", "custom_handler": anaconda_llm}
)
