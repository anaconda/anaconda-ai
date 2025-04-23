import os
import time
from typing import Any
from typing import Dict
from typing import AsyncGenerator
from typing import Optional
from typing import Union
from urllib.parse import urljoin

import httpx
import numpy as np
import panel as pn

from ..clients import get_default_client
from ..clients.base import Server, APIParams, LoadParams, InferParams

HERE = os.path.dirname(__file__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class AnacondaModelHandler:
    server: Optional[Server] = None

    def __init__(
        self,
        model_id: str,
        display_throughput: bool = False,
        system_message: Optional[str] = None,
        client_options: Optional[dict] = None,
        api_params: Optional[Union[Dict[str, Any], APIParams]] = None,
        load_params: Optional[Union[Dict[str, Any], LoadParams]] = None,
        infer_params: Optional[Union[Dict[str, Any], InferParams]] = None,
    ) -> None:
        self.display_throughput = display_throughput

        if system_message is None:
            self.system_message = DEFAULT_SYSTEM_MESSAGE
        else:
            self.system_message = system_message

        self.client_options = {} if client_options is None else client_options
        self.api_params = api_params
        self.load_params = load_params
        self.infer_params = infer_params

        self.messages = [{"content": self.system_message, "role": "system"}]

        self.model_id = model_id

        self.server = None
        self._get_or_create_service()

        self.avatar = os.path.join(HERE, "Anaconda_Logo.png")

    def _get_or_create_service(self) -> None:
        client = get_default_client()
        if self.server is None:
            self.server = client.servers.create(
                model=self.model_id,
                api_params=self.api_params,
                load_params=self.load_params,
                infer_params=self.infer_params,
            )

        self.server.start()
        self.model_name = self.server.serverConfig.modelFileName
        self.client = self.server.openai_async_client()

    async def throughput(self, message: str, timedelta: float) -> float:
        url = urljoin(str(self.client.base_url), "/tokenize")
        res = httpx.post(url, json={"content": message})
        if not res.is_success:
            return len(message.split()) / timedelta

        tokens = len(res.json()["tokens"])
        return tokens / timedelta

    def extract(self, chunk: Any) -> str:
        text = chunk.choices[0].delta.content
        return "" if text is None else text

    async def grab_n(self, aiterable: Any, n: int = 5) -> str:
        cached = ""
        items = 0
        async for chunk in aiterable:
            cached += self.extract(chunk)
            items += 1
            if items == n:
                break
        return cached

    async def callback(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ) -> AsyncGenerator[Union[pn.chat.ChatMessage, dict], None]:
        if self.display_throughput:
            message = pn.chat.ChatMessage(
                user=self.model_id,
                avatar=self.avatar,
            )

            indicator = pn.indicators.Gauge(
                name="LLM Throughput",
                value=0,
                bounds=(0, 100),
                format="{value} T/s",
                colors=[(0.25, "green"), (0.75, "gold"), (1, "red")],
            )
            message._composite.append(indicator)

            yield message

        self.messages.append({"content": contents, "role": "user"})

        t0 = time.time()
        chunks = await self.client.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.model_name,
            stream=True,
            **self.client_options,  # type: ignore
        )

        # Grab the first few chunks to stabilize the throughput
        if self.display_throughput:
            full_text = await self.grab_n(chunks, n=5)
        else:
            full_text = ""

        async for chunk in chunks:  # type: ignore
            full_text += self.extract(chunk)

            if self.display_throughput:
                message = pn.chat.ChatMessage(
                    full_text,
                    user=self.model_id,
                    avatar=self.avatar,
                )

                timedelta = time.time() - t0
                throughput = await self.throughput(full_text, timedelta)
                indicator.value = np.round(throughput, decimals=2)
                message._composite.append(indicator)
                yield message
            else:
                yield {
                    "object": full_text,
                    "user": self.model_id,
                    "avatar": self.avatar,
                }

        if not self.display_throughput:
            timedelta = time.time() - t0
            throughput = await self.throughput(full_text, timedelta)
            yield {
                "object": full_text,
                "user": self.model_id,
                "avatar": self.avatar,
                "footer_objects": [f"{throughput:.2f} tokens/second"],
            }

        self.messages.append({"content": full_text, "role": "system"})
