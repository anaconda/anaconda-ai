"""Unit tests for system_prompts resource (list and get)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

from anaconda_ai.clients.anaconda_desktop import AnacondaDesktopSystemPrompts
from anaconda_ai.clients.base import (
    BaseSystemPrompts,
    GenericClient,
    PromptListResponse,
    PromptSummary,
    SystemPrompt,
)
from anaconda_ai.exceptions import ProjectsAPIError, SystemPromptNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Provide a mock GenericClient with config.domain set for auth-derived base URL."""
    client = MagicMock(spec=GenericClient)
    # config.domain is how anaconda_auth resolves the authenticated domain;
    # since config comes from BaseClient (parent), we need to explicitly
    # set it on the mock as spec=GenericClient doesn't auto-inherit it.
    client.config = MagicMock()
    client.config.domain = "test.anaconda.com"
    return client


@pytest.fixture()
def desktop_prompts(mock_client: MagicMock) -> AnacondaDesktopSystemPrompts:
    """Provide an AnacondaDesktopSystemPrompts instance backed by a mock client."""
    return AnacondaDesktopSystemPrompts(mock_client)


# ---------------------------------------------------------------------------
# Sample API response data
# ---------------------------------------------------------------------------

SAMPLE_PROJECTS = [
    {
        "id": "proj-uuid-1",
        "name": "finance-coding-assistant-a3f2",
        "created_at": "2026-03-12T10:00:00Z",
        "updated_at": "2026-03-12T11:00:00Z",
        "metadata": {"tags": ["advisor"]},
        "owner": {"id": "user-uuid", "type": "user"},
    },
    {
        "id": "proj-uuid-2",
        "name": "python-tutor-b1c3",
        "created_at": "2026-03-11T09:00:00Z",
        "updated_at": "2026-03-11T10:00:00Z",
        "metadata": {"tags": ["advisor"]},
        "owner": {"id": "user-uuid", "type": "user"},
    },
    {
        "id": "proj-uuid-3",
        "name": "data-science-helper-ff02",
        "created_at": "2026-03-10T08:00:00Z",
        "updated_at": "2026-03-10T09:00:00Z",
        "metadata": {"tags": ["advisor"]},
        "owner": {"id": "user-uuid", "type": "user"},
    },
]


def _make_response(status_code: int = 200, json_data: object = None) -> MagicMock:
    """Create a mock requests.Response with the given status code and JSON body."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ===========================================================================
# US1: list() tests
# ===========================================================================


class TestSystemPromptsList:
    """Tests for client.system_prompts.list()."""

    def test_list_returns_three_prompt_summaries(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() with 3 projects returns 3 PromptSummary items with correct fields."""
        mock_client.get.return_value = _make_response(
            200,
            {
                "items": SAMPLE_PROJECTS,
                "num_items": 3,
                "next_page_url": None,
            },
        )

        result = desktop_prompts.list()

        assert isinstance(result, PromptListResponse)
        assert len(result.items) == 3
        assert all(isinstance(item, PromptSummary) for item in result.items)
        assert result.items[0].name == "finance-coding-assistant-a3f2"
        assert result.items[0].created_at == datetime(
            2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc
        )
        assert result.items[1].name == "python-tutor-b1c3"
        assert result.items[2].name == "data-science-helper-ff02"
        assert result.next_page_url is None

    def test_list_empty_returns_empty_response(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() with 0 projects returns empty PromptListResponse (not an error)."""
        mock_client.get.return_value = _make_response(
            200,
            {"items": [], "num_items": 0, "next_page_url": None},
        )

        result = desktop_prompts.list()

        assert isinstance(result, PromptListResponse)
        assert len(result.items) == 0
        assert result.next_page_url is None

    def test_list_passes_through_next_page_url(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() with next_page_url present passes it through in response."""
        next_url = "https://test.anaconda.com/api/projects/?owner=me&tag=advisor&page=2"
        mock_client.get.return_value = _make_response(
            200,
            {
                "items": SAMPLE_PROJECTS[:1],
                "num_items": 1,
                "next_page_url": next_url,
            },
        )

        result = desktop_prompts.list()

        assert result.next_page_url == next_url

    def test_list_5xx_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() on 5xx from Projects API raises ProjectsAPIError."""
        mock_client.get.return_value = _make_response(500)

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.list()

    def test_list_connection_error_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() on network/connection error raises ProjectsAPIError."""
        mock_client.get.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.list()

    def test_list_timeout_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() on request timeout raises ProjectsAPIError."""
        mock_client.get.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.list()

    def test_list_calls_correct_url(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list() calls the correct Projects API URL."""
        mock_client.get.return_value = _make_response(
            200,
            {"items": [], "num_items": 0, "next_page_url": None},
        )

        desktop_prompts.list()

        mock_client.get.assert_called_once_with(
            "https://test.anaconda.com/api/projects/?owner=me&tag=advisor"
        )

    def test_list_with_next_page_url_uses_provided_url(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list(next_page_url=...) uses the provided URL instead of building one."""
        cursor_url = (
            "https://test.anaconda.com/api/projects/"
            "?owner=me&tag=advisor&cursor=abc123&limit=1000"
        )
        mock_client.get.return_value = _make_response(
            200,
            {
                "items": SAMPLE_PROJECTS[1:2],
                "num_items": 1,
                "next_page_url": None,
            },
        )

        result = desktop_prompts.list(next_page_url=cursor_url)

        mock_client.get.assert_called_once_with(cursor_url)
        assert isinstance(result, PromptListResponse)
        assert len(result.items) == 1
        assert result.items[0].name == "python-tutor-b1c3"
        assert result.next_page_url is None

    def test_list_with_next_page_url_error_handling(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """list(next_page_url=...) raises ProjectsAPIError on 5xx."""
        cursor_url = (
            "https://test.anaconda.com/api/projects/"
            "?owner=me&tag=advisor&cursor=abc123&limit=1000"
        )
        mock_client.get.return_value = _make_response(500)

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.list(next_page_url=cursor_url)

    def test_list_pagination_round_trip(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """First page returns next_page_url, second call with it returns last page."""
        next_url = (
            "https://test.anaconda.com/api/projects/"
            "?owner=me&tag=advisor&cursor=xyz789&limit=1000"
        )
        # First page
        page1_resp = _make_response(
            200,
            {
                "items": SAMPLE_PROJECTS[:1],
                "num_items": 1,
                "next_page_url": next_url,
            },
        )
        # Second page
        page2_resp = _make_response(
            200,
            {
                "items": SAMPLE_PROJECTS[1:2],
                "num_items": 1,
                "next_page_url": None,
            },
        )
        mock_client.get.side_effect = [page1_resp, page2_resp]

        page1 = desktop_prompts.list()
        assert page1.next_page_url == next_url
        assert page1.items[0].name == "finance-coding-assistant-a3f2"

        page2 = desktop_prompts.list(next_page_url=page1.next_page_url)
        assert page2.next_page_url is None
        assert page2.items[0].name == "python-tutor-b1c3"


# ===========================================================================
# US2: get() tests
# ===========================================================================


class TestSystemPromptsGet:
    """Tests for client.system_prompts.get()."""

    def test_get_returns_system_prompt(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """get() returns SystemPrompt with correct fields."""
        # Step 1: name lookup returns 1 project
        lookup_resp = _make_response(
            200,
            {
                "items": [SAMPLE_PROJECTS[0]],
                "num_items": 1,
                "next_page_url": None,
            },
        )
        # Step 2: file download returns prompt content
        file_resp = _make_response(
            200,
            {"system_prompt": "You are a finance coding assistant."},
        )
        mock_client.get.side_effect = [lookup_resp, file_resp]

        result = desktop_prompts.get("finance-coding-assistant-a3f2")

        assert isinstance(result, SystemPrompt)
        assert result.name == "finance-coding-assistant-a3f2"
        assert result.system_prompt == "You are a finance coding assistant."
        assert result.created_at == datetime(2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc)
        assert result.updated_at == datetime(2026, 3, 12, 11, 0, 0, tzinfo=timezone.utc)

    def test_get_nonexistent_raises_not_found(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """get() with nonexistent name raises SystemPromptNotFoundError."""
        mock_client.get.return_value = _make_response(
            200,
            {"items": [], "num_items": 0, "next_page_url": None},
        )

        with pytest.raises(SystemPromptNotFoundError):
            desktop_prompts.get("nonexistent-name")

    def test_get_404_on_file_download_raises_not_found(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """get() on 404 from file download raises SystemPromptNotFoundError."""
        lookup_resp = _make_response(
            200,
            {
                "items": [SAMPLE_PROJECTS[0]],
                "num_items": 1,
                "next_page_url": None,
            },
        )
        file_resp = _make_response(404)
        mock_client.get.side_effect = [lookup_resp, file_resp]

        with pytest.raises(SystemPromptNotFoundError):
            desktop_prompts.get("finance-coding-assistant-a3f2")

    def test_get_5xx_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """get() on 5xx from Projects API raises ProjectsAPIError."""
        mock_client.get.return_value = _make_response(500)

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.get("finance-coding-assistant-a3f2")

    def test_get_prompt_name_derivation_standard(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """Prompt name derivation: finance-coding-assistant-a3f2 -> prompts/finance-coding-assistant.json."""
        lookup_resp = _make_response(
            200,
            {
                "items": [SAMPLE_PROJECTS[0]],
                "num_items": 1,
                "next_page_url": None,
            },
        )
        file_resp = _make_response(200, {"system_prompt": "test prompt"})
        mock_client.get.side_effect = [lookup_resp, file_resp]

        desktop_prompts.get("finance-coding-assistant-a3f2")

        # Verify the file download URL uses the correct derived prompt name
        file_url = mock_client.get.call_args_list[1][0][0]
        assert "/files/prompts/finance-coding-assistant.json" in file_url

    def test_get_prompt_name_derivation_multi_hyphen(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """Multi-hyphen name: my-cool-agent-abc123 -> prompts/my-cool-agent.json."""
        project = {
            "id": "proj-uuid-multi",
            "name": "my-cool-agent-abc123",
            "created_at": "2026-03-12T10:00:00Z",
            "updated_at": "2026-03-12T11:00:00Z",
            "metadata": {"tags": ["advisor"]},
            "owner": {"id": "user-uuid", "type": "user"},
        }
        lookup_resp = _make_response(
            200,
            {"items": [project], "num_items": 1, "next_page_url": None},
        )
        file_resp = _make_response(200, {"system_prompt": "test prompt"})
        mock_client.get.side_effect = [lookup_resp, file_resp]

        desktop_prompts.get("my-cool-agent-abc123")

        file_url = mock_client.get.call_args_list[1][0][0]
        assert "/files/prompts/my-cool-agent.json" in file_url

    def test_get_prompt_name_derivation_single_segment(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """Single-segment name: simple-ff02 -> prompts/simple.json."""
        project = {
            "id": "proj-uuid-simple",
            "name": "simple-ff02",
            "created_at": "2026-03-12T10:00:00Z",
            "updated_at": "2026-03-12T11:00:00Z",
            "metadata": {"tags": ["advisor"]},
            "owner": {"id": "user-uuid", "type": "user"},
        }
        lookup_resp = _make_response(
            200,
            {"items": [project], "num_items": 1, "next_page_url": None},
        )
        file_resp = _make_response(200, {"system_prompt": "test prompt"})
        mock_client.get.side_effect = [lookup_resp, file_resp]

        desktop_prompts.get("simple-ff02")

        file_url = mock_client.get.call_args_list[1][0][0]
        assert "/files/prompts/simple.json" in file_url

    def test_get_malformed_prompt_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """Malformed prompt file (missing system_prompt key) raises ProjectsAPIError."""
        lookup_resp = _make_response(
            200,
            {
                "items": [SAMPLE_PROJECTS[0]],
                "num_items": 1,
                "next_page_url": None,
            },
        )
        # Prompt file is missing the system_prompt key
        file_resp = _make_response(200, {"some_other_key": "value"})
        mock_client.get.side_effect = [lookup_resp, file_resp]

        with pytest.raises(ProjectsAPIError, match="missing 'system_prompt' key"):
            desktop_prompts.get("finance-coding-assistant-a3f2")

    def test_get_connection_error_raises_projects_api_error(
        self, desktop_prompts: AnacondaDesktopSystemPrompts, mock_client: MagicMock
    ) -> None:
        """get() on network error raises ProjectsAPIError."""
        mock_client.get.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        with pytest.raises(ProjectsAPIError):
            desktop_prompts.get("finance-coding-assistant-a3f2")
