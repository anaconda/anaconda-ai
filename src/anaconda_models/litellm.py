from typing import Callable, Iterator, Optional, Any, Union, cast

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
        _service = _model.start()
        _client = _service.openai_client

        response = _client.chat.completions.create(messages=messages, model=model)
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
        _service = _model.start()
        _client = _service.openai_client

        response = _client.chat.completions.create(
            messages=messages, model=model, stream=True
        )
        wrapped = CustomStreamWrapper(
            completion_stream=response, model=model, logging_obj=logging_obj
        )

        for chunk in wrapped:
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
