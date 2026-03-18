from typing import Any, Callable, Dict, Optional

import requests
from anaconda_auth.client import BaseClient as AuthBaseClient

from ..config import AnacondaAIConfig
from ..exceptions import ProjectsAPIError, SystemPromptNotFoundError
from .base import (
    BaseSystemPrompts,
    GenericClient,
    PromptListResponse,
    PromptSummary,
    SystemPrompt,
)
from .ai_navigator import (
    AINavigatorModels,
    AINavigatorServers,
    AINavigatorVectorDbServer,
    AiNavigatorVersion,
)


def _derive_prompt_name(project_name: str) -> str:
    """Derive the prompt file name from a project name.

    The advisor creates project names as ``f"{prompt_name}-{hex_suffix}"``,
    so stripping everything after the last hyphen recovers the original
    prompt name used in the file path ``prompts/{prompt_name}.json``.
    """
    return project_name.rsplit("-", 1)[0]


class AnacondaDesktopSystemPrompts(BaseSystemPrompts):
    """System prompt operations via the Anaconda Platform Collections API.

    Unlike models/servers/vector_db (which reuse AINavigator classes),
    this class talks directly to the cloud Projects API because
    AINavigator does not support system prompts.

    The *client* passed to this class must be an
    ``anaconda_auth.client.BaseClient`` authenticated against the
    user's Anaconda Cloud domain so that the correct cloud auth token
    is sent — the desktop client's local API key is not valid for
    cloud endpoints.

    The Projects API base URL is derived from the client's domain
    (e.g., ``https://anaconda.com``), so it automatically follows
    whichever Anaconda instance the user is logged into.
    """

    @property
    def _projects_api_base_url(self) -> str:
        """Derive the Projects API base URL from the client's auth domain."""
        return f"https://{self.client.config.domain}"

    def _api_request(
        self, request_fn: Callable[..., requests.Response], *args: Any, **kwargs: Any
    ) -> requests.Response:
        """Execute an API request with standardised error handling.

        Wraps *request_fn* (e.g. ``self.client.get``,
        ``self.client.post``) so that connectivity errors and
        server-side failures are surfaced as :class:`ProjectsAPIError`
        consistently.
        """
        try:
            response = request_fn(*args, **kwargs)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            raise ProjectsAPIError(f"Failed to connect to Projects API: {exc}") from exc

        if response.status_code >= 500:
            raise ProjectsAPIError(f"Projects API returned {response.status_code}")

        return response

    def list(self, *, next_page_url: Optional[str] = None) -> PromptListResponse:
        """List advisor-generated system prompts owned by the authenticated user."""
        if next_page_url is not None:
            url = next_page_url
        else:
            base_url = self._projects_api_base_url
            url = f"{base_url}/api/projects/?owner=me&tag=advisor"

        response = self._api_request(self.client.get, url)

        data = response.json()
        items = [
            PromptSummary(
                name=item["name"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
            )
            for item in data.get("items", [])
        ]
        return PromptListResponse(
            items=items,
            next_page_url=data.get("next_page_url"),
        )

    def get(self, name: str) -> SystemPrompt:
        """Retrieve a system prompt by its full project name."""
        base_url = self._projects_api_base_url

        # Step 1: Look up project by name
        lookup_url = f"{base_url}/api/projects/?owner=me&tag=advisor&name={name}"
        response = self._api_request(self.client.get, lookup_url)

        data = response.json()
        items = data.get("items", [])
        if not items:
            raise SystemPromptNotFoundError(f"System prompt '{name}' not found")

        project = items[0]
        project_id = project["id"]
        prompt_name = _derive_prompt_name(project["name"])

        # Step 2: Download the prompt file
        file_url = (
            f"{base_url}/api/projects/{project_id}/files/prompts/{prompt_name}.json"
        )
        file_response = self._api_request(self.client.get, file_url)

        if file_response.status_code == 404:
            raise SystemPromptNotFoundError(f"System prompt '{name}' not found")

        file_data = file_response.json()
        if "system_prompt" not in file_data:
            raise ProjectsAPIError(
                f"Prompt file for '{name}' has unexpected format: "
                f"missing 'system_prompt' key"
            )

        return SystemPrompt(
            name=project["name"],
            system_prompt=file_data["system_prompt"],
            created_at=project["created_at"],
            updated_at=project["updated_at"],
        )


class AnacondaDesktopClient(GenericClient):
    name: str = "anaconda-desktop"

    def __init__(self, app_name: Optional[str] = None, **kwargs: Any) -> None:
        ai_kwargs: Dict[str, Any] = {"backends": {"anaconda_desktop": {}}}
        if app_name is not None:
            ai_kwargs["backends"]["anaconda_desktop"]["app_name"] = app_name

        config = AnacondaAIConfig.model_validate(ai_kwargs)
        domain = f"localhost:{config.backends.anaconda_desktop.port}"
        api_key = config.backends.anaconda_desktop.api_key

        super().__init__(domain=domain, api_key=api_key)
        self._base_uri = f"http://{domain}"

        if app_name is not None:
            self.ai_config.backends.anaconda_desktop.app_name = app_name

        self.models = AINavigatorModels(self)
        self.servers = AINavigatorServers(self)
        self.vector_db = AINavigatorVectorDbServer(self)
        # System prompts use the cloud Projects API (not AINavigator's local API),
        # so they need a cloud-authenticated client rather than the desktop one.
        self.system_prompts = AnacondaDesktopSystemPrompts(AuthBaseClient())

    @property
    def online(self) -> bool:
        res = self.get("/api")
        return res.status_code < 400

    def get_version(self) -> Dict[str, str]:
        res = self.get("api")
        return AiNavigatorVersion(**res.json()["data"]).model_dump()
