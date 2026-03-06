import json
import sys
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import List
from typing import MutableMapping
from typing import Tuple
from typing import Union

import typer
from requests.exceptions import HTTPError
from rich.console import RenderableType
from rich.prompt import Confirm
from rich.table import Column
from rich.table import Table
from typer.core import TyperCommand

from anaconda_ai.config import AnacondaAIConfig
from anaconda_cli_base import console
from .clients import AnacondaAIClient, clients
from .clients.base import GenericClient, Server, VectorDbTableSchema
from .agents import AGENTS, find_agent_binary
from .process import run_agent_foreground
from ._version import __version__

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")

CHECK_MARK = "[bold green]✔︎[/bold green]"


class AgentCommand(TyperCommand):
    """Custom command class that intercepts '--' before click consumes it.

    Click silently strips '--' from args during parsing, making it impossible
    to distinguish server options from agent arguments in ctx.args. This
    subclass splits args on '--' in parse_args() before click sees them,
    stashing agent arguments in ctx.meta['agent_args'].
    """

    def parse_args(self, ctx: typer.Context, args: list) -> list:  # type: ignore[override]
        if "--" in args:
            idx = args.index("--")
            ctx.meta["agent_args"] = list(args[idx + 1 :])
            args = list(args[:idx])
        else:
            ctx.meta["agent_args"] = []
        return super().parse_args(ctx, args)


AS_JSON = Annotated[
    bool, typer.Option("--json", is_flag=True, help="Print output as JSON")
]


def get_running_servers(client: GenericClient) -> Sequence[Server]:
    try:
        servers = client.servers.list()
        return servers
    except (HTTPError, AttributeError):
        return []


def _list_models(
    client: GenericClient, show_blocked: Optional[bool] = None
) -> Tuple[RenderableType, Sequence[MutableMapping[str, Any]]]:
    models = client.models.list()
    servers = get_running_servers(client)
    table = Table(
        Column("Model", no_wrap=True),
        "Params (B)",
        "Quantizations\ndownloaded in bold\ngreen for active servers",
        "Trained for",
        header_style="bold green",
    )

    data: List[MutableMapping[str, Any]] = []

    if show_blocked is None:
        show_blocked = AnacondaAIConfig().show_blocked_models

    for model in sorted(models, key=lambda m: m.name):
        quantizations = []
        quant_data = []
        for quant in model.quantized_files:
            if not quant.is_allowed and not show_blocked:
                continue

            matched_servers = [
                s for s in servers if s.config.model_name.endswith(quant.identifier)
            ]

            if quant.is_allowed:
                color = "green" if matched_servers else ""
                emphasis = "bold" if (quant.is_downloaded or matched_servers) else "dim"
            else:
                color = "bright_red"
                emphasis = "dim"

            method = (
                f"[{emphasis} {color}]{quant.quant_method.upper()}[/{emphasis} {color}]"
            )

            quant_dict = {
                "method": quant.quant_method,
                "running": bool(matched_servers),
                "downloaded": quant.is_downloaded,
                "blocked": not quant.is_allowed,
            }
            quant_data.append(quant_dict)

            quantizations.append(method)

        if quantizations:
            quants = ", ".join(quantizations)
            parameters = f"{model.num_parameters / 1e9:8.2f}"
            table.add_row(model.name, parameters, quants, model.trained_for)
            data.append(
                {
                    "model": model.name,
                    "parameters": model.num_parameters,
                    "quantizations": quant_data,
                    "trained_for": model.trained_for,
                }
            )
    return table, data


def _model_info(
    client: GenericClient, model_id: str
) -> Tuple[RenderableType, MutableMapping[str, Any]]:
    info = client.models.get(model_id)
    servers = get_running_servers(client)

    table = Table.grid(padding=1, pad_edge=True)
    table.title = model_id
    table.add_column("Metadata", no_wrap=True, justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Description", info.description)
    parameters = f"{info.num_parameters / 1e9:8.2f}B"
    table.add_row("Parameters", parameters)
    table.add_row("Trained For", info.trained_for)

    data: MutableMapping[str, Any] = {
        "name": model_id,
        "description": info.description,
        "parameters": info.num_parameters,
        "trained_for": info.trained_for,
        "quantizations": [],
    }

    quantized = Table(
        Column("Filename", no_wrap=True),
        "Method",
        "Downloaded",
        "Max Ram (GB)",
        "Size (GB)",
        "Server(s)",
        header_style="bold green",
    )
    for quant in info.quantized_files:
        method = quant.quant_method.upper()
        if not quant.is_allowed:
            method = f"[bright_red]{method}[/bright_red]"
        downloaded = CHECK_MARK if quant.is_downloaded else ""
        matched_servers = [
            s for s in servers if s.config.model_name.endswith(quant.identifier)
        ]
        running = CHECK_MARK if matched_servers else ""

        ram = f"{quant.max_ram_usage / 1024 / 1024 / 1024:.2f}"
        size = f"{quant.size_bytes / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(quant.identifier, method, downloaded, ram, size, running)

        data["quantizations"].append(
            {
                "filename": quant.identifier,
                "downloaded": quant.is_downloaded,
                "running": bool(matched_servers),
                "ram": quant.max_ram_usage,
                "size": quant.size_bytes,
            }
        )

    table.add_row("Quantized Files", quantized)
    return table, data


@app.command(name="version")
def version(
    backend: Annotated[Optional[str], typer.Option(help="Select backend")] = None,
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """Version information of SDK and Backend"""
    versions: Dict[str, str] = {}

    versions["anaconda-ai"] = __version__

    try:
        client = AnacondaAIClient(backend=backend, site=site)
        backend_version = client.get_version()
        versions.update(backend_version)
    except Exception:
        console.print(f"Backend {client.name} not reachable.")

    if as_json:
        console.print_json(data=versions)
    else:
        table = Table("Component", "Version", header_style="bold green")
        for component, version in versions.items():
            table.add_row(component, version)
        console.print(table)


@app.command(name="models")
def models(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="Optional Model name for detailed information"),
    ] = None,
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    show_blocked: Optional[bool] = typer.Option(
        None,
        "--show-blocked/--no-show-blocked",
        help="Show or hide unavailable models.",
    ),
    as_json: AS_JSON = False,
) -> None:
    """Model information"""
    client = AnacondaAIClient(backend=backend, site=site)
    data: Union[Sequence[MutableMapping[str, Any]], MutableMapping[str, Any]]
    if model_id is None:
        renderable, data = _list_models(client, show_blocked=show_blocked)
    else:
        renderable, data = _model_info(client, model_id)

    if as_json:
        console.print_json(data=data)
    else:
        console.print(renderable)


@app.command(name="download")
def download(
    model: str = typer.Argument(help="Model name with quantization"),
    force: bool = typer.Option(
        False, help="Force re-download of model if already downloaded."
    ),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Hard-link model file to this path after download"
        ),
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """Download a model"""
    client = AnacondaAIClient(backend=backend, site=site)
    client.models.download(
        model, show_progress=not as_json, force=force, console=console, path=output
    )

    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command(name="remove")
def remove(
    model: str = typer.Argument(help="Model name with quantization"),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """Remove a downloaded a model"""
    client = AnacondaAIClient(backend=backend, site=site)
    client.models.delete(model)
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command(
    name="launch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch(
    ctx: typer.Context,
    model: str = typer.Argument(
        help="Name of the quantized model, it will download first if needed.",
    ),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    remove: bool = typer.Option(
        True,
        "--rm/--detach",
        help="Stop and remove server on ctrl-C (default) or leave running with --detach.",
    ),
    show: Optional[bool] = typer.Option(
        False, help="Open your webbrowser when the server starts."
    ),
    as_json: AS_JSON = False,
) -> None:
    """Launch an inference server for a model"""
    extra_options = {}
    for arg in ctx.args:
        if not arg.startswith("--"):
            raise ValueError("Extra server args must be passed as --key=value or --key")
        key, *value = arg[2:].split("=", maxsplit=1)

        if len(value) > 1:
            raise ValueError(arg)

        extra_options[key] = True if not value else value[0]

    client = AnacondaAIClient(backend=backend, site=site)

    server = client.servers.create(
        model=model,
        extra_options=extra_options,
        show_progress=not as_json,
        console=console,
    )
    server.start(show_progress=not as_json, leave_running=True, console=console)
    table, data = _server_info(server)
    if as_json:
        console.print_json(data=data)
    else:
        console.print(table, soft_wrap=True)
    if show:
        import webbrowser

        webbrowser.open(server.url)

    if not remove:
        return

    try:
        console.print("[it]This server will stop and delete on [/it] [bold]^C[/bold]")
        while True:
            pass
    except KeyboardInterrupt:
        if server._matched:
            return

        if remove:
            server.stop(show_progress=not as_json, console=console)
            server.delete(show_progress=not as_json, console=console)
        return


def _servers_list(
    servers: Sequence[Server],
) -> Tuple[RenderableType, Sequence[MutableMapping[str, Any]]]:
    table = Table(
        Column("Server ID", no_wrap=True),
        "Model Name",
        "Status",
        header_style="bold green",
    )

    data = []

    for server in servers:
        table.add_row(
            str(server.id),
            str(server.config.model_name),
            server.status,
        )

        data.append(
            {
                "server_id": server.id,
                "model": server.config.model_name,
                "status": server.status,
            }
        )

    return table, data


def _server_info(server: Server) -> Tuple[RenderableType, MutableMapping[str, Any]]:
    table = Table.grid(padding=1, pad_edge=True)
    table.add_column("Metadata", justify="center", style="bold green")
    table.add_column("Value", justify="left", no_wrap=True)
    table.add_row("Server", server.id)
    table.add_row("Model", server.config.model_name)
    table.add_row("OpenAI Compatible URL", server.openai_url)
    table.add_row("Status", server.status)
    table.add_row("Parameters", json.dumps(server.config.params, indent=2))

    data = {
        "model": server.config.model_name,
        "openai_url": server.openai_url,
        "status": server.status,
        "parameters": server.config.params,
    }

    return table, data


@app.command("servers")
def servers(
    server: Annotated[Optional[str], typer.Argument(help="Server ID")] = None,
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """List running servers"""
    client = AnacondaAIClient(backend=backend, site=site)

    data: Union[Sequence[MutableMapping[str, Any]], MutableMapping[str, Any]]
    if server:
        s = client.servers.get(server)
        renderable, data = _server_info(s)
    else:
        servers = client.servers.list()
        renderable, data = _servers_list(servers)

    if as_json:
        console.print_json(data=data)
    else:
        console.print(renderable)


@app.command("stop")
def stop(
    server: str = typer.Argument(help="ID of the server to stop"),
    remove: bool = typer.Option(
        False,
        "--remove",
        "--rm",
        help="Delete server on stop. Not supported by all backends.",
    ),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    client = AnacondaAIClient(backend=backend, site=site)
    s = client.servers.get(server)
    if s.is_running:
        s.stop(show_progress=not as_json)

    if remove:
        s.delete(show_progress=not as_json)

    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("launch-vectordb")
def launch_vector_db(as_json: AS_JSON = False) -> None:
    """
    Starts a vector db
    """
    client = AnacondaAIClient()
    result = client.vector_db.create(show_progress=not as_json)

    table = Table.grid(padding=1, pad_edge=True)
    table.title = "Vector DB"
    table.add_column("Field", justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Running", CHECK_MARK)
    table.add_row("Host", result.host)
    table.add_row("Port", str(result.port))
    table.add_row("User", result.user)
    table.add_row("Password", result.password)
    table.add_row("Database", result.database)
    table.add_row("URI", result.uri)

    if as_json:
        console.print_json(data=result.model_dump())
    else:
        console.print(table)


@app.command("delete-vectordb")
def delete_vector_db(as_json: AS_JSON = False) -> None:
    """
    Deletes the vector db
    """
    client = AnacondaAIClient()
    client.vector_db.delete()
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("stop-vectordb")
def stop_vector_db(as_json: AS_JSON = False) -> None:
    """
    Stops the vector db
    """
    client = AnacondaAIClient()
    _ = client.vector_db.stop()
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("list-tables")
def list_tables(as_json: AS_JSON = False) -> None:
    """
    Lists all tables in the vector db
    """
    client = AnacondaAIClient()
    tables = client.vector_db.get_tables()

    if as_json:
        console.print_json(data=[t.model_dump() for t in tables])
    else:
        db_table = Table.grid(padding=1, pad_edge=True)
        for table in tables:
            columns = Table("Name", "Type", "Constraints", header_style="bold green")
            columns.title = table.name
            for column in table.table_schema.columns:
                columns.add_row(column.name, column.type, ",".join(column.constraints))
            db_table.add_row(columns)
        console.print(db_table)


@app.command("drop-table")
def drop_table(
    table: str = typer.Argument(help="Name of the table to drop"),
    as_json: AS_JSON = False,
) -> None:
    """
    Drops a table from the vector db
    """
    client = AnacondaAIClient()
    client.vector_db.drop_table(table)
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("create-table")
def create_table(
    table: str = typer.Argument(help="Name of the table to create"),
    schema: str = typer.Argument(help="Schema of the table to create"),
    as_json: AS_JSON = False,
) -> None:
    """
    Creates a table in the vector db
    """
    client = AnacondaAIClient()
    validated_schema = VectorDbTableSchema.model_validate_json(schema)
    client.vector_db.create_table(table, validated_schema)
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


def _confirm_write(
    sites: AnacondaAIConfig,
    yes: Optional[bool],
    preserve_existing_keys: bool = True,
) -> None:
    if yes is True:
        sites.write_config(preserve_existing_keys=preserve_existing_keys)
    elif yes is False:
        sites.write_config(dry_run=True, preserve_existing_keys=preserve_existing_keys)
    else:
        sites.write_config(dry_run=True, preserve_existing_keys=preserve_existing_keys)
        if Confirm.ask("Confirm:"):
            sites.write_config(preserve_existing_keys=preserve_existing_keys)


@app.command(
    "config",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def configure(
    backend: Annotated[str | None, typer.Option(help="Set the default backend")] = None,
    stop_server_on_exit: Annotated[bool | None, typer.Option()] = None,
    server_operations_timeout: Annotated[
        int | None,
        typer.Option(help="Timeout (seconds) waiting for server start, default is 60"),
    ] = None,
    show_blocked_models: Annotated[
        bool | None, typer.Option(help="Show blocked models in CLI.")
    ] = None,
    yes: Annotated[
        Optional[bool],
        typer.Option(
            "--yes/--dry-run",
            "-y",
            help="Confirm changes and write, use --dry-run to print diff but do not write",
        ),
    ] = None,
) -> None:
    config = AnacondaAIConfig()
    if backend is not None:
        if backend not in clients:
            console.print(f"{backend} is not a supported backend.")
        config.backend = backend  # type: ignore

    if server_operations_timeout is not None:
        config.server_operations_timeout = server_operations_timeout

    if stop_server_on_exit is not None:
        config.stop_server_on_exit = stop_server_on_exit

    if show_blocked_models is not None:
        config.show_blocked_models = show_blocked_models

    _confirm_write(config, yes=yes)


@app.command("mcp")
def mcp_server(
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="Transport: stdio or streamable-http"),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option("--host", help="Host for streamable-http (default 127.0.0.1)"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port for streamable-http (default 8000)"),
    ] = 8000,
) -> None:
    """Run the Anaconda AI MCP server (list models/servers, start/stop/remove servers).

    Requires: pip install 'anaconda-ai[mcp]'
    Use stdio for IDE/client integration; use streamable-http for HTTP at host:port/mcp.
    """
    try:
        from anaconda_ai.mcp_server import run
    except ImportError as e:
        console.print(
            "[red]MCP server requires the mcp package.[/] "
            "Install with: [bold]pip install 'anaconda-ai[mcp]'[/]"
        )
        raise typer.Exit(1) from e
    run(transport=transport, host=host, port=port)


def _run_wrapper(
    agent_name: str,
    ctx: typer.Context,
    model_or_server: str,
    site: Optional[str],
    backend: Optional[str],
    remove: bool,
    as_json: bool,
) -> None:
    """Shared logic for coding agent wrapper commands (claude, opencode).

    Orchestrates: parse positional arg → resolve server → check agent binary →
    build env → fork+exec agent → cleanup on exit.
    """
    # Look up agent definition
    agent = AGENTS.get(agent_name)
    if agent is None:
        console.print(f"[red]Unknown agent: {agent_name}[/]")
        raise typer.Exit(1)

    # Check agent binary is installed
    binary_path = find_agent_binary(agent.binary)
    if binary_path is None:
        console.print(f"[red]{agent.name} is not installed.[/] {agent.install_hint}")
        raise typer.Exit(1)

    # Parse extra server options from ctx.args (agent args already extracted by AgentCommand)
    extra_options: Dict[str, Any] = {}
    agent_args: List[str] = ctx.meta.get("agent_args", [])
    for arg in ctx.args:
        if not arg.startswith("--"):
            raise ValueError("Extra server args must be passed as --key=value or --key")
        key, *value = arg[2:].split("=", maxsplit=1)
        if len(value) > 1:
            raise ValueError(arg)
        extra_options[key] = True if not value else value[0]

    client = AnacondaAIClient(backend=backend, site=site)

    # Resolve server: model name or server/<id>
    server_owned = False
    if model_or_server.startswith("server/"):
        # Connect to existing server by ID
        server_id = model_or_server[len("server/") :]
        try:
            server = client.servers.get(server_id)
        except Exception:
            console.print(f"[red]Server '{server_id}' not found or not running[/]")
            raise typer.Exit(1)
        if not server.is_running:
            console.print(f"[red]Server '{server_id}' not found or not running[/]")
            raise typer.Exit(1)
        model_name = server.config.model_name
    else:
        # Launch or reuse server for model
        model_name = model_or_server
        server = None
        try:
            server = client.servers.create(
                model=model_name,
                extra_options=extra_options,
                show_progress=not as_json,
                console=console,
            )
            server.start(show_progress=not as_json, leave_running=True, console=console)
            server_owned = not server._matched
        except KeyboardInterrupt:
            # Interrupt cleanup: stop server if we started creating it
            console.print("\n[yellow]Interrupted. Cleaning up...[/]")
            try:
                if server is not None and server_owned:
                    server.stop(show_progress=not as_json, console=console)
                    server.delete(show_progress=not as_json, console=console)
            except Exception:
                pass
            raise typer.Exit(130)

    # Show server info
    table, data = _server_info(server)
    if as_json:
        console.print_json(data=data)
    else:
        console.print(table, soft_wrap=True)

    # Build agent-specific environment variables
    env = agent.build_env(server, model_name)

    # Build agent-specific CLI args (e.g., --model for OpenCode) and prepend to user args
    injected_args = agent.build_agent_args(model_name)
    agent_args = injected_args + agent_args

    # Build cleanup function
    cleanup_fn = None
    if server_owned and remove:

        def _do_cleanup() -> None:
            server.stop(show_progress=not as_json, console=console)
            server.delete(show_progress=not as_json, console=console)

        cleanup_fn = _do_cleanup

    # Fork+exec the agent
    try:
        exit_code = run_agent_foreground(
            binary_path=binary_path,
            agent_args=agent_args,
            env=env,
            cleanup_fn=cleanup_fn,
        )
    except KeyboardInterrupt:
        # Ctrl+C during agent startup — clean up if we own the server
        if cleanup_fn is not None:
            try:
                cleanup_fn()
            except Exception:
                pass
        exit_code = 130

    sys.exit(exit_code)


@app.command(
    name="claude",
    cls=AgentCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def claude(
    ctx: typer.Context,
    model_or_server: str = typer.Argument(
        help="Model name (e.g., OpenHermes-2.5-Mistral-7B/Q4_K_M) or server/<id> to connect to existing server.",
    ),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    remove: bool = typer.Option(
        True,
        "--rm/--detach",
        help="Stop server on exit (default) or leave running with --detach.",
    ),
    as_json: AS_JSON = False,
) -> None:
    """Launch Claude Code connected to an Anaconda AI inference server.

    Use -- to separate server options from agent arguments.

    \b
    Examples:
        anaconda ai claude MyModel/Q4_K_M
        anaconda ai claude MyModel/Q4_K_M --detach --ctx-size=4096 -- --verbose
        anaconda ai claude server/abc123 -- --no-confirm
    """
    _run_wrapper(
        agent_name="claude",
        ctx=ctx,
        model_or_server=model_or_server,
        site=site,
        backend=backend,
        remove=remove,
        as_json=as_json,
    )


@app.command(
    name="opencode",
    cls=AgentCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def opencode(
    ctx: typer.Context,
    model_or_server: str = typer.Argument(
        help="Model name (e.g., OpenHermes-2.5-Mistral-7B/Q4_K_M) or server/<id> to connect to existing server.",
    ),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    remove: bool = typer.Option(
        True,
        "--rm/--detach",
        help="Stop server on exit (default) or leave running with --detach.",
    ),
    as_json: AS_JSON = False,
) -> None:
    """Launch OpenCode connected to an Anaconda AI inference server.

    Use -- to separate server options from agent arguments.

    \b
    Examples:
        anaconda ai opencode MyModel/Q4_K_M
        anaconda ai opencode MyModel/Q4_K_M --detach --ctx-size=4096 -- --theme dark
        anaconda ai opencode server/abc123
    """
    _run_wrapper(
        agent_name="opencode",
        ctx=ctx,
        model_or_server=model_or_server,
        site=site,
        backend=backend,
        remove=remove,
        as_json=as_json,
    )
