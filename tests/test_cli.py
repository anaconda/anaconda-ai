from pathlib import Path
from typing import Any
from typing import Protocol

import pytest
from click.testing import Result
from pytest import MonkeyPatch
from typer.testing import CliRunner

from anaconda_cli_base.cli import app

SUBCOMMANDS = {
    "version",
    "models",
    "download",
    "launch",
    "servers",
    "stop",
    "config",
}

DEPRECATED_SUBCOMMANDS = {
    "launch-vectordb",
    "delete-vectordb",
    "stop-vectordb",
    "create-table",
    "drop-table",
    "list-tables",
}


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


@pytest.mark.parametrize("action", DEPRECATED_SUBCOMMANDS)
def test_deprecated_vectordb_commands(invoke_cli: CLIInvoker, action: str) -> None:
    result = invoke_cli("ai", action, "--help")
    assert result.exit_code == 0
    assert "deprecated" in result.stdout.lower()


@pytest.mark.parametrize("action", {"config"})
def test_no_args_is_help(invoke_cli: CLIInvoker, action: str) -> None:
    result = invoke_cli("ai", action)
    assert result.exit_code == 2
    assert "Usage" in result.stdout
