from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel, OpenAIEmbeddingSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile, OpenAIJsonSchemaTransformer
from pydantic_ai.providers import Provider

from ..clients import AnacondaAIClient
from ..clients.base import Server


class AnacondaChatModelSettings(OpenAIChatModelSettings, total=False):
    extra_options: Dict[str, Any]


class AnacondaModelProfile(OpenAIModelProfile):
    pass


class AnacondaProvider(Provider[AsyncOpenAI]):
    @property
    def name(self) -> str:
        return "anaconda"

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url="", api_key="")

    def model_profile(self, model_name: str) -> OpenAIModelProfile | None:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_chat_thinking_field="reasoning",
            openai_chat_send_back_thinking_parts="tags",
        )


class AnacondaMixin:
    _server: Optional[Server] = None

    def _get_openai_client(
        self, model_name: str, extra_options: dict, client: AnacondaAIClient
    ) -> AsyncOpenAI:
        if self._server is None:
            if model_name.startswith("server/"):
                _, server_name = model_name.split("server/", maxsplit=1)
                self._server = client.servers.get(server_name)
            else:
                self._server = client.servers.create(
                    model=model_name, extra_options=extra_options
                )

        if not self._server.is_running:
            self._server.start()

        return self._server.async_openai_client()


class AnacondaChatModel(OpenAIChatModel, AnacondaMixin):
    _server: Optional[Server] = None

    def __init__(
        self,
        model_name: str,
        backend: Optional[str] = None,
        site: Optional[str] = None,
        client: Optional[AnacondaAIClient] = None,
        profile: Optional[AnacondaModelProfile] = None,
        settings: Optional[AnacondaChatModelSettings] = None,
    ) -> None:
        super().__init__(
            model_name, settings=settings, profile=profile, provider=AnacondaProvider()
        )

        anaconda_client = client or AnacondaAIClient(backend=backend, site=site)

        extra_options: Dict[str, Any] = (
            settings.get("extra_options", {}) if settings else {}
        )
        openai_client = self._get_openai_client(
            model_name, extra_options, anaconda_client
        )
        self.client = openai_client

        self.profile = AnacondaModelProfile().update(self.profile)


class AnacondaEmbeddingSettings(OpenAIEmbeddingSettings, total=False):
    extra_options: Dict[str, Any]


class AnacondaEmbeddingModel(OpenAIEmbeddingModel, AnacondaMixin):
    _server: Optional[Server] = None

    def __init__(
        self,
        model_name: str,
        *,
        backend: Optional[str] = None,
        site: Optional[str] = None,
        client: Optional[AnacondaAIClient] = None,
        settings: AnacondaEmbeddingSettings | None = None,
    ):
        super().__init__(model_name, provider=AnacondaProvider(), settings=settings)

        anaconda_client = client or AnacondaAIClient(backend=backend, site=site)

        extra_options: Dict[str, Any] = (
            settings.get("extra_options", {}) if settings else {}
        )
        openai_client = self._get_openai_client(
            model_name, extra_options, anaconda_client
        )
        self._client = openai_client

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return self._model_name
