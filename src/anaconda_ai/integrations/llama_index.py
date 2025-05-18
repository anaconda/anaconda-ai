from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import Set
from typing import Tuple
from typing_extensions import Self

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.postgres.base import PGType
from pydantic import Field

from ..clients import get_default_client
from ..clients.base import (
    GenericClient,
    APIParams,
    LoadParams,
    InferParams,
    ServerConfig,
)
from ..spec import AISpec, DEFAULT_SPEC_PATH


class AnacondaLLMMetadata(LLMMetadata):
    server_config: Dict[str, Any]


class AnacondaModel(OpenAI):
    """Download and run a model from Anaconda"""

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )

    _tokenizer: None = None
    tokenizer: None = None
    _server_config: ServerConfig

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        api_params: Optional[Union[Dict[str, Any], APIParams]] = None,
        load_params: Optional[Union[Dict[str, Any], LoadParams]] = None,
        infer_params: Optional[Union[Dict[str, Any], InferParams]] = None,
        client: Optional[GenericClient] = None,
    ) -> None:
        if client is None:
            client = get_default_client()

        server = client.servers.create(
            model,
            api_params=api_params,
            load_params=load_params,
            infer_params=infer_params,
        )
        server.start()
        context_window = client.models.get(model).metadata.contextWindowSize

        super().__init__(
            model=server.serverConfig.modelFileName,
            api_key=server.api_key,
            api_base=server.openai_url,
            is_chat_model=True,
            api_version="empty",
            system_prompt=system_prompt,
            context_window=context_window,
            max_tokens=max_tokens,
            is_function_calling_model=False,
            temperature=temperature,
        )

        self._server_config = server.serverConfig

    @classmethod
    def load_from_spec(cls, key: str, path: Path = DEFAULT_SPEC_PATH) -> Self:
        spec = AISpec.load(path)
        if key not in spec.inference:
            raise ValueError(f"The key {key} is not defined as an inference in {path}")

        server_config = spec.inference[key]

        system_prompt = getattr(server_config, "system_prompt", None)
        temperature = getattr(server_config, "temperature", DEFAULT_TEMPERATURE)
        max_tokens = getattr(server_config, "max_tokens", None)

        llm = cls(
            model=server_config.model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_params=server_config.api_params,
            load_params=server_config.load_params,
            infer_params=server_config.infer_params,
        )
        return llm

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AnacondaModels"

    @property
    def metadata(self) -> AnacondaLLMMetadata:
        server_config = self._server_config.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude={"logsDir", "modelFileName"},
        )

        return AnacondaLLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            server_config=server_config,
        )


class AnacondaEmbedding(OpenAILikeEmbedding):
    _server_config: ServerConfig

    def __init__(
        self,
        model_name: str,
        embed_batch_size: int = 10,
        dimensions: Optional[int] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        api_params: Optional[Union[Dict[str, Any], APIParams]] = None,
        load_params: Optional[Union[Dict[str, Any], LoadParams]] = None,
        reuse_client: bool = True,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        client: Optional[GenericClient] = None,
    ) -> None:
        if client is None:
            client = get_default_client()

        if load_params is None:
            load_params = {}

        if isinstance(load_params, LoadParams):
            load_params.embedding = True
        else:
            load_params["embedding"] = True

        server = client.servers.create(
            model_name,
            api_params=api_params,
            load_params=load_params,
        )
        server.start()

        super().__init__(
            model_name=server.serverConfig.modelFileName,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            api_key=server.api_key or "empty",
            api_base=server.openai_url,
            is_chat_model=True,
            api_version="empty",
            additional_kwargs=additional_kwargs,
        )

        self._server_config = server.serverConfig

    @classmethod
    def class_name(cls) -> str:
        return "AnacondaEmbedding"

    @classmethod
    def load_from_spec(cls, key: str, path: Path = DEFAULT_SPEC_PATH) -> Self:
        spec = AISpec.load(path)
        if key not in spec.inference:
            raise ValueError(f"The key {key} is not defined as an inference in {path}")

        server_config = spec.inference[key]

        embed_batch_size = getattr(server_config, "embed_batch_size", 10)
        dimensions = getattr(server_config, "dimensions", 1024)
        max_retries = getattr(server_config, "max_retries", 10)
        timeout = getattr(server_config, "timeout", 60.0)

        embed = cls(
            model_name=server_config.model,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            max_retries=max_retries,
            timeout=timeout,
        )

        return embed


class AnacondaVectorStore(PGVectorStore):
    def __init__(
        self,
        table_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1024,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        initialization_fail_on_error: bool = False,
        use_halfvec: bool = False,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
        client: Optional[GenericClient] = None,
    ) -> None:
        if client is None:
            client = get_default_client()

        vector_db = client.vector_db.create()

        host = vector_db.host
        port = vector_db.port
        database = vector_db.database
        user = vector_db.user
        password = vector_db.password

        connection_string = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        async_connection_string = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )

        if table_name and "-" in table_name:
            raise ValueError(f"Table name {table_name}, dashes are not allowed")

        super().__init__(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            initialization_fail_on_error=initialization_fail_on_error,
            use_halfvec=use_halfvec,
            indexed_metadata_keys=indexed_metadata_keys,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AnacondaVectorStore"

    @classmethod
    def load_from_spec(cls, path: Path = DEFAULT_SPEC_PATH) -> Self:
        spec = AISpec.load(path)

        vector_db_config = spec.vector_db
        if vector_db_config is None:
            raise ValueError(f"No vector_db defined in {path}")

        table_name = getattr(vector_db_config, "table_name", None)
        schema_name = getattr(vector_db_config, "schema_name", None)
        hybrid_search = getattr(vector_db_config, "hybrid_search", False)
        text_search_config = getattr(vector_db_config, "text_search_config", "english")
        embed_dim = getattr(vector_db_config, "embed_dim", 1024)
        cache_ok = getattr(vector_db_config, "cache_ok", False)
        perform_setup = getattr(vector_db_config, "perform_setup", True)
        debug = getattr(vector_db_config, "debug", False)
        use_jsonb = getattr(vector_db_config, "use_jsonb", False)
        hnsw_kwargs = getattr(vector_db_config, "hnsw_kwargs", None)
        use_halfvec = getattr(vector_db_config, "use_halfvec", False)
        indexed_metadata_keys = getattr(vector_db_config, "indexed_metadata_keys", None)

        vector_store = cls(
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            use_halfvec=use_halfvec,
            indexed_metadata_keys=indexed_metadata_keys,
        )

        return vector_store
