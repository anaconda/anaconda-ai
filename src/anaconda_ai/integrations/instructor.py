from typing import Any, Optional, overload, Literal

import instructor
from instructor.auto_client import from_provider as original_from_provider
from instructor.cache import BaseCache
from instructor.models import KnownModelName

from ..clients import AnacondaAIClient
from ..clients.base import Server


def _get_server(
    model_name: str, extra_options: dict, client: AnacondaAIClient
) -> Server:
    if model_name.startswith("server/"):
        _, server_name = model_name.split("server/", maxsplit=1)
        server = client.servers.get(server_name)
    else:
        server = client.servers.create(model=model_name, extra_options=extra_options)

    if not server.is_running:
        server.start()

    return server


def from_anaconda(
    model: str,
    backend: Optional[str] = None,
    site: Optional[str] = None,
    client: AnacondaAIClient | None = None,
    mode: instructor.Mode | None = None,
    extra_options: Optional[dict[str, Any]] = None,
    async_client: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    mode = instructor.mode.Mode.JSON

    server = _get_server(
        model_name=model,
        extra_options=extra_options or {},
        client=client or AnacondaAIClient(backend=backend, site=site),
    )

    if async_client:
        client = server.openai_client()
        create = client.chat.completions.create
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.UNKNOWN,
            mode=mode,
            model=server.config.model_name,
            **kwargs,
        )
    else:
        client = server.async_openai_client()
        create = client.chat.completions.create
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.UNKNOWN,
            mode=mode,
            model=server.config.model_name,
            **kwargs,
        )


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_provider(
    model: str | KnownModelName,  # noqa: UP007
    async_client: bool = False,
    cache: BaseCache | None = None,
    mode: instructor.Mode | None = None,  # noqa: ARG001, UP007
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:  # noqa: UP007
    provider, model_name = model.split("/", maxsplit=1)
    if provider == "anaconda":
        return from_anaconda(model_name, async_client=async_client, mode=mode, **kwargs)
    else:
        return original_from_provider(
            model=model, async_client=async_client, cache=cache, mode=mode, **kwargs
        )


instructor.from_provider = from_provider
