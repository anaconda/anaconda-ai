from typing import Any, Optional, Union, MutableMapping, Sequence
from warnings import warn

from anaconda_auth.config import AnacondaAuthSite

from ..config import AnacondaAIConfig
from ..exceptions import UnknownBackendError
from .base import GenericClient
from .ai_navigator import AINavigatorClient
from .ai_catalyst import AICatalystClient

clients = {"ai-navigator": AINavigatorClient, "ai-catalyst": AICatalystClient}


class AnacondaAIClient(GenericClient):
    """Client for Anaconda AI models and servers"""

    name: str

    def __init__(
        self,
        site: Optional[Union[str, AnacondaAuthSite]] = None,
        base_uri: Optional[str] = None,
        domain: Optional[str] = None,
        auth_domain_override: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_version: Optional[str] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
        extra_headers: Optional[Union[str, dict]] = None,
        hash_hostname: Optional[bool] = None,
        proxy_servers: Optional[MutableMapping[str, str]] = None,
        client_cert: Optional[str] = None,
        client_cert_key: Optional[str] = None,
        backend: Optional[str] = None,
        stop_server_on_exit: Optional[bool] = None,
        server_operations_timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        try:
            self.__class__ = clients[backend or AnacondaAIConfig().backend]
        except KeyError:
            raise UnknownBackendError(f"There is no known backend called {backend}")

        self.__init__(  # type: ignore
            site=site,
            base_uri=base_uri,
            domain=domain,
            auth_domain_override=auth_domain_override,
            api_key=api_key,
            user_agent=user_agent,
            api_version=api_version,
            ssl_verify=ssl_verify,
            extra_headers=extra_headers,
            hash_hostname=hash_hostname,
            proxy_servers=proxy_servers,
            client_cert=client_cert,
            client_cert_key=client_cert_key,
            **kwargs,
        )


def get_backends() -> Sequence[str]:
    """Return a list of all available backends"""
    return list(clients.keys())


def get_default_client(*args: Any, **kwargs: Any) -> GenericClient:
    warn("get_default_client is deprecated, use AnacondaAIClient(...) instead")
    return AnacondaAIClient(*args, **kwargs)
