from typing import Any
from typing import Optional
from typing import cast

from openai import OpenAI
from openai.resources.chat import Completions
from pandasai.helpers.memory import Memory
from pandasai.llm.base import LLM
from pandasai.pipelines.pipeline_context import PipelineContext
from pandasai.prompts.base import BasePrompt

from anaconda_models.core import AnacondaQuantizedModelCache


class AnacondaModel(LLM):
    @property
    def type(self) -> str:
        return f"anaconda:{self.model.name}@{self.run_on}"

    def is_pandasai_llm(self) -> bool:
        return False

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        format: str = "gguf",
        run_on: str = "local",
        **kwargs: Any,
    ) -> None:
        self.model = AnacondaQuantizedModelCache(
            name=model_name, quantization=quantization, format=format
        )
        self.run_on = run_on
        self._inference_kwargs = kwargs

        self.client = self._prepare_client()

    def _prepare_client(self) -> Completions:
        service = self.model.start(run_on=self.run_on, **self._inference_kwargs)
        openai_base_url = f"{service.url}/v1"
        client = OpenAI(api_key="none", base_url=openai_base_url)
        return client.chat.completions

    def chat_completion(self, value: str, memory: Memory) -> str:
        messages = memory.to_openai_messages() if memory else []

        # adding current prompt as latest query message
        messages.append(
            {
                "role": "user",
                "content": value,
            }
        )

        params = {"model": self.model.name, "messages": messages}
        response = self.client.create(**params)  # type: ignore

        return cast(str, response.choices[0].message.content)

    def call(
        self, instruction: BasePrompt, context: Optional[PipelineContext] = None
    ) -> str:
        self.last_prompt = instruction.to_string()

        memory = context.memory if context else None

        return self.chat_completion(self.last_prompt, memory)  # type: ignore
