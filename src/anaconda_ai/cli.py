import json
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Optional
from typing import Sequence
from typing import List
from typing import MutableMapping
from typing import Tuple

import typer
from requests.exceptions import HTTPError
from rich.console import RenderableType
from rich.table import Column
from rich.table import Table

from anaconda_ai.config import AnacondaAIConfig
from anaconda_cli_base import console
from .clients import AnacondaAIClient
from .clients.base import GenericClient, Server, VectorDbTableSchema
from ._version import __version__

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")

CHECK_MARK = "[bold green]✔︎[/bold green]"


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
def version() -> None:
    """Version information of SDK and AI Navigator"""
    console.print(f"SDK: {__version__}")

    try:
        client = AnacondaAIClient()
        version = client.get_version()
        console.print(version)
    except Exception:
        console.print("AI Navigator not found. Is it running?")


@app.command(name="models")
def models(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="Optional Model name for detailed information"),
    ] = None,
    site: Annotated[
        Optional[str], "--at", typer.Option(help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    show_blocked: Optional[bool] = typer.Option(
        None,
        "--show-blocked/--no-show-blocked",
        help="Show or hide unavailable models.",
    ),
    as_json: bool = typer.Option(False, "--json", is_flag=True),
) -> None:
    """Model information"""
    client = AnacondaAIClient(backend=backend, site=site)
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
        Optional[str], "--at", typer.Option(help="Site defined in config")
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
) -> None:
    """Download a model"""
    client = AnacondaAIClient(backend=backend, site=site)
    client.models.download(
        model, show_progress=True, force=force, console=console, path=output
    )
    console.print("[green]Success[/green]")


@app.command(name="remove")
def remove(
    model: str = typer.Argument(help="Model name with quantization"),
    site: Annotated[
        Optional[str], "--at", typer.Option(help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    """Remove a downloaded a model"""
    client = AnacondaAIClient(backend=backend, site=site)
    client.models.delete(model)
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
        Optional[str], "--at", typer.Option(help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    detach: bool = typer.Option(
        default=False, help="Start model server and leave it running."
    ),
    show: Optional[bool] = typer.Option(
        False, help="Open your webbrowser when the server starts."
    ),
) -> None:
    """Launch an inference server for a model"""
    extra_options = {}
    for arg in ctx.args:
        if not arg.startswith("--"):
            continue
        key, *value = arg[2:].split("=", maxsplit=1)

        if len(value) > 1:
            raise ValueError(arg)

        extra_options[key] = True if value[0] is None else value[0]

    client = AnacondaAIClient(backend=backend, site=site)

    server = client.servers.create(model=model, extra_options=extra_options)
    server.start(show_progress=True, leave_running=True)
    _server_info(server)
    if show:
        import webbrowser

        webbrowser.open(server.url)

    if client.servers.always_detach:
        return
    elif detach:
        return

    try:
        while True:
            pass
    except KeyboardInterrupt:
        if server._matched:
            return

        server.stop(show_progress=True)
        return


def _servers_list(servers: Sequence[Server]) -> None:
    table = Table(
        Column("Server ID", no_wrap=True),
        "Model Name",
        "Status",
        header_style="bold green",
    )

    for server in servers:
        table.add_row(
            str(server.id),
            str(server.config.model_name),
            server.status,
        )

    console.print(table)


def _server_info(server: Server) -> None:
    table = Table.grid(padding=1, pad_edge=True)
    table.title = server.id
    table.add_column("Metadata", justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Model", server.config.model_name)
    table.add_row("OpenAI Compatible URL", server.openai_url)
    table.add_row("Status", server.status)
    table.add_row("Parameters", json.dumps(server.config.params, indent=2))
    console.print(table)


@app.command("servers")
def servers(
    server: Annotated[Optional[str], typer.Argument(help="Server ID")] = None,
    site: Annotated[
        Optional[str], "--at", typer.Option(help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    """List running servers"""
    client = AnacondaAIClient(backend=backend, site=site)

    if server:
        s = client.servers.get(server)
        _server_info(s)
    else:
        servers = client.servers.list()
        _servers_list(servers)


@app.command("stop")
def stop(
    server: str = typer.Argument(help="ID of the server to stop"),
    remove: bool = typer.Argument(
        default=False, help="Delete server on stop. Not supported by all backends."
    ),
    site: Annotated[
        Optional[str], "--at", typer.Option(help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    client = AnacondaAIClient(backend=backend, site=site)
    s = client.servers.get(server)
    s.stop(show_progress=True)

    if remove:
        client.servers.delete(server)


@app.command("launch-vectordb")
def launch_vector_db() -> None:
    """
    Starts a vector db
    """
    client = AnacondaAIClient()
    result = client.vector_db.create()
    console.print(result)


@app.command("delete-vectordb")
def delete_vector_db() -> None:
    """
    Deletes the vector db
    """
    client = AnacondaAIClient()
    client.vector_db.delete()
    console.print("Vector db deleted")


@app.command("stop-vectordb")
def stop_vector_db() -> None:
    """
    Stops the vector db
    """
    client = AnacondaAIClient()
    result = client.vector_db.stop()
    console.print(result)


@app.command("list-tables")
def list_tables() -> None:
    """
    Lists all tables in the vector db
    """
    client = AnacondaAIClient()
    tables = client.vector_db.get_tables()
    console.print(tables)


@app.command("drop-table")
def drop_table(
    table: str = typer.Argument(help="Name of the table to drop"),
) -> None:
    """
    Drops a table from the vector db
    """
    client = AnacondaAIClient()
    client.vector_db.drop_table(table)
    console.print(f"Table {table} dropped")


@app.command("create-table")
def create_table(
    table: str = typer.Argument(help="Name of the table to create"),
    schema: str = typer.Argument(help="Schema of the table to create"),
) -> None:
    """
    Creates a table in the vector db
    """
    client = AnacondaAIClient()
    validated_schema = VectorDbTableSchema.model_validate_json(schema)
    client.vector_db.create_table(table, validated_schema)
    console.print(f"Table {table} created")
