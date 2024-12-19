from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from requests import Response
from requests.exceptions import ConnectionError

from anaconda_cloud_auth.client import BaseClient
from anaconda_cloud_auth.client import BearerAuth
from anaconda_cloud_auth.config import AnacondaCloudConfig
from anaconda_models import __version__ as version
from anaconda_models.config import ModelsConfig


class Client(BaseClient):
    _user_agent = f"anaconda-models/{version}"

    def __init__(
        self,
        domain: Optional[str] = None,
        auth_domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        extra_headers: Optional[Union[str, dict]] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if domain is not None:
            kwargs["domain"] = domain
        if ssl_verify is not None:
            kwargs["ssl_verify"] = ssl_verify
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers

        self._config = ModelsConfig(**kwargs)

        super().__init__(
            domain=self._config.domain,
            user_agent=user_agent,
            extra_headers=self._config.extra_headers,
            ssl_verify=self._config.ssl_verify,
        )

        auth_kwargs: Dict[str, Any] = {}
        if auth_domain is not None:
            auth_kwargs["domain"] = auth_domain
        if api_key is not None:
            auth_kwargs["api_key"] = api_key
        auth_config = AnacondaCloudConfig(**auth_kwargs)
        self.auth = BearerAuth(domain=auth_config.domain, api_key=auth_config.api_key)


class AINavigatorClient(BaseClient):
    _user_agent = f"anaconda-models/{version}"

    def __init__(
        self,
        port: Optional[int] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if port is not None:
            kwargs["ai_navigator"] = {"port": port}

        self._config = ModelsConfig(**kwargs)

        domain = f"localhost:{self._config.ai_navigator.port}"

        super().__init__(
            domain=domain,
            ssl_verify=False,
        )

        self._base_uri = f"http://{domain}"

    def request(
        self,
        method: Union[str, bytes],
        url: Union[str, bytes],
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        try:
            return super().request(method, url, *args, **kwargs)
        except ConnectionError:
            raise RuntimeError(
                "Could not connect to AI Navigator. It may not be running."
            )
