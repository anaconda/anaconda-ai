from typing import Any, Optional, Union, MutableMapping
from warnings import warn

from anaconda_auth.config import AnacondaAuthSite

from ..config import AnacondaAIConfig
from .base import GenericClient
from .ai_navigator import AINavigatorClient

clients = {"ai-navigator": AINavigatorClient}


class AnacondaAIClient(GenericClient):
    """Client for Anaconda AI models and servers"""

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
    ):
        if backend is None:
            config = AnacondaAIConfig()
            self.__class__ = clients[config.backend]  # type: ignore
        else:
            self.__class__ = clients[backend]  # type: ignore

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
        )


def get_default_client(*args: Any, **kwargs: Any) -> GenericClient:
    warn("get_default_client is deprecated, use AnacondaAIClient(...) instead")
    return AnacondaAIClient(*args, **kwargs)


def make_client(
    backend: Optional[str] = None, site: Optional[str] = None, **kwargs: Any
) -> GenericClient:
    if backend is None:
        config = AnacondaAIConfig()
        return clients[config.backend](site=site, **kwargs)
    else:
        return clients[backend](site=site, **kwargs)


__all__ = ["make_client"]
