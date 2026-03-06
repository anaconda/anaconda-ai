from pathlib import Path
from typing import Any
from typing import Protocol
from unittest.mock import MagicMock, patch

import pytest
from click.testing import Result
from pytest import MonkeyPatch
from typer.testing import CliRunner

from anaconda_cli_base.cli import app

SUBCOMMANDS = [
    "version",
    "models",
    "download",
    "launch",
    "servers",
    "stop",
    "launch-vectordb",
    "delete-vectordb",
    "stop-vectordb",
    "create-table",
    "drop-table",
    "list-tables",
    "claude",
    "opencode",
]


class CLIInvoker(Protocol):
    def __call__(self, *args: str) -> Any: ...


@pytest.fixture()
def invoke_cli(tmp_path: Path, monkeypatch: MonkeyPatch) -> CLIInvoker:
    """Returns a function, which can be used to call the CLI from within a temporary directory."""
    runner = CliRunner()

    monkeypatch.chdir(tmp_path)

    def f(*args: str) -> Result:
        return runner.invoke(app, args)

    return f


@pytest.mark.parametrize("action", SUBCOMMANDS)
def test_feature_action(invoke_cli: CLIInvoker, action: str) -> None:
    result = invoke_cli("ai", action, "--help")
    assert result.exit_code == 0


class TestServerIdParsing:
    """Tests for server/<id> positional argument parsing (US2)."""

    def test_parse_server_id(self, invoke_cli: CLIInvoker) -> None:
        """Test that 'server/abc123' is parsed as server ID, not model name."""
        mock_server = MagicMock()
        mock_server.id = "abc123"
        mock_server.url = "http://localhost:8080"
        mock_server.openai_url = "http://localhost:8080/v1"
        mock_server.api_key = "test-key"
        mock_server.is_running = True
        mock_server.config.model_name = "test-model"
        mock_server.config.params = {}
        mock_server.status = "running"

        mock_client = MagicMock()
        mock_client.servers.get.return_value = mock_server

        with (
            patch("anaconda_ai.cli.AnacondaAIClient", return_value=mock_client),
            patch("anaconda_ai.cli.find_agent_binary", return_value="/usr/bin/claude"),
            patch("anaconda_ai.cli.run_agent_foreground", return_value=0),
        ):
            _ = invoke_cli("ai", "claude", "server/abc123")

        # Verify servers.get was called with the extracted ID
        mock_client.servers.get.assert_called_once_with("abc123")
        # Verify servers.create was NOT called (existing server path)
        mock_client.servers.create.assert_not_called()

    def test_server_id_not_found(self, invoke_cli: CLIInvoker) -> None:
        """Test that unknown server ID shows error message."""
        mock_client = MagicMock()
        mock_client.servers.get.side_effect = Exception("not found")

        with (
            patch("anaconda_ai.cli.AnacondaAIClient", return_value=mock_client),
            patch("anaconda_ai.cli.find_agent_binary", return_value="/usr/bin/claude"),
        ):
            result = invoke_cli("ai", "claude", "server/xyz999")

        assert result.exit_code == 1
        assert "not found or not running" in result.output


class TestExtraServerOptions:
    """Tests for extra server options parsing (US5)."""

    def test_extra_server_options_parsed(self, invoke_cli: CLIInvoker) -> None:
        """Test that --ctx-size=4096 --jinja are passed to servers.create as extra_options."""
        mock_server = MagicMock()
        mock_server.id = "srv1"
        mock_server.url = "http://localhost:8080"
        mock_server.openai_url = "http://localhost:8080/v1"
        mock_server.api_key = "test-key"
        mock_server.is_running = True
        mock_server._matched = False
        mock_server.config.model_name = "test-model"
        mock_server.config.params = {}
        mock_server.status = "running"

        mock_client = MagicMock()
        mock_client.servers.create.return_value = mock_server

        with (
            patch("anaconda_ai.cli.AnacondaAIClient", return_value=mock_client),
            patch("anaconda_ai.cli.find_agent_binary", return_value="/usr/bin/claude"),
            patch("anaconda_ai.cli.run_agent_foreground", return_value=0),
        ):
            _ = invoke_cli(
                "ai",
                "claude",
                "TestModel/Q4_K_M",
                "--ctx-size=4096",
                "--jinja",
            )

        # Verify servers.create was called with the extra options
        create_call = mock_client.servers.create.call_args
        assert create_call is not None
        extra_options = create_call.kwargs.get("extra_options", {})
        assert extra_options.get("ctx-size") == "4096"
        assert extra_options.get("jinja") is True


class TestSeparatorParsing:
    """Tests for '--' separator behavior with AgentCommand.parse_args."""

    def test_separator_preserved_and_typed_options_parsed(
        self, invoke_cli: CLIInvoker
    ) -> None:
        """Test that '--' splits server options from agent args, and typed
        options work regardless of position (before or after positional)."""
        mock_server = MagicMock()
        mock_server.id = "srv1"
        mock_server.url = "http://localhost:8080"
        mock_server.openai_url = "http://localhost:8080/v1"
        mock_server.api_key = "test-key"
        mock_server.is_running = True
        mock_server._matched = False
        mock_server.config.model_name = "test-model"
        mock_server.config.params = {}
        mock_server.status = "running"

        mock_client = MagicMock()
        mock_client.servers.create.return_value = mock_server

        captured_agent_args = []

        def mock_run_agent(binary_path, agent_args, env, cleanup_fn=None):
            captured_agent_args.extend(agent_args)
            return 0

        with (
            patch("anaconda_ai.cli.AnacondaAIClient", return_value=mock_client),
            patch("anaconda_ai.cli.find_agent_binary", return_value="/usr/bin/claude"),
            patch("anaconda_ai.cli.run_agent_foreground", side_effect=mock_run_agent),
        ):
            # Typed option AFTER positional (natural UX), server opts, --, agent args
            _ = invoke_cli(
                "ai",
                "claude",
                "TestModel/Q4_K_M",
                "--detach",
                "--ctx-size=4096",
                "--",
                "--verbose",
                "--no-confirm",
            )

        # --detach was parsed by typer (server should not be cleaned up)
        # Server create was called with server options only
        create_call = mock_client.servers.create.call_args
        assert create_call is not None
        extra_options = create_call.kwargs.get("extra_options", {})
        assert extra_options.get("ctx-size") == "4096"
        # Agent received only args after '--'
        assert captured_agent_args == ["--verbose", "--no-confirm"]

    def test_separator_splits_server_and_agent_args(
        self, invoke_cli: CLIInvoker
    ) -> None:
        """Test '--' correctly splits ctx.args into server options and agent args."""
        mock_server = MagicMock()
        mock_server.id = "srv1"
        mock_server.url = "http://localhost:8080"
        mock_server.openai_url = "http://localhost:8080/v1"
        mock_server.api_key = "test-key"
        mock_server.is_running = True
        mock_server._matched = False
        mock_server.config.model_name = "test-model"
        mock_server.config.params = {}
        mock_server.status = "running"

        mock_client = MagicMock()
        mock_client.servers.create.return_value = mock_server

        captured_agent_args = []

        def mock_run_agent(binary_path, agent_args, env, cleanup_fn=None):
            captured_agent_args.extend(agent_args)
            return 0

        with (
            patch("anaconda_ai.cli.AnacondaAIClient", return_value=mock_client),
            patch("anaconda_ai.cli.find_agent_binary", return_value="/usr/bin/claude"),
            patch("anaconda_ai.cli.run_agent_foreground", side_effect=mock_run_agent),
        ):
            _ = invoke_cli(
                "ai",
                "claude",
                "TestModel/Q4_K_M",
                "--ctx-size=4096",
                "--jinja",
                "--",
                "--verbose",
                "--no-confirm",
            )

        # Server options parsed correctly
        create_call = mock_client.servers.create.call_args
        extra_options = create_call.kwargs.get("extra_options", {})
        assert extra_options == {"ctx-size": "4096", "jinja": True}
        # Agent args are only what comes after '--'
        assert captured_agent_args == ["--verbose", "--no-confirm"]
