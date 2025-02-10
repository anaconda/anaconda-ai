from typing import Callable, AsyncIterator, Iterator, Optional, Any, Union, cast

import litellm
from httpx import Timeout
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, GenericStreamingChunk
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from anaconda_models.core import (
    AnacondaQuantizedModelCache,
    AnacondaQuantizedModelService,
)


class AnacondaLLM(CustomLLM):
    _model: Optional[AnacondaQuantizedModelCache] = None
    _service: Optional[AnacondaQuantizedModelService] = None

    def _prepare_inference_kwargs(self, optional_params: dict) -> dict:
        inference_kwargs = optional_params.copy()
        _ = inference_kwargs.pop("stream", None)
        _ = inference_kwargs.pop("stream_options", None)
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
        acompletion: Optional[Any] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        _model = AnacondaQuantizedModelCache(name=model)
        _service = _model.start(**optional_params.pop("llama_cpp_kwargs", {}))
        _client = _service.openai_client

        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        response = _client.chat.completions.create(
            messages=messages, model=model, **inference_kwargs
        )
        mresponse = ModelResponse(**response.model_dump())
        _service.options["Process"].terminate()
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
        acompletion: Optional[Any] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        _model = AnacondaQuantizedModelCache(name=model)
        _service = _model.start(**optional_params.pop("llama_cpp_kwargs", {}))
        _client = _service.openai_client

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

        _service.options["Process"].terminate()

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
        acompletion: Optional[Any] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        _model = AnacondaQuantizedModelCache(name=model)
        _service = _model.start(**optional_params.pop("llama_cpp_kwargs", {}))
        _client = _service.openai_async_client

        inference_kwargs = self._prepare_inference_kwargs(optional_params)
        response = await _client.chat.completions.create(
            messages=messages, model=model, **inference_kwargs
        )
        mresponse = ModelResponse(**response.model_dump())
        _service.options["Process"].terminate()
        return mresponse

    async def astreaming(
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
        acompletion: Optional[Any] = None,
        litellm_params: Optional[Any] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        _model = AnacondaQuantizedModelCache(name=model)
        _service = _model.start(**optional_params.pop("llama_cpp_kwargs", {}))
        _client = _service.openai_async_client

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

        _service.options["Process"].terminate()


# This should be moved to an entrypoint if implemented
# https://github.com/BerriAI/litellm/issues/7733
anaconda_llm = AnacondaLLM()
litellm.custom_provider_map.append(
    {"provider": "anaconda", "custom_handler": anaconda_llm}
)
