from pathlib import Path
from typing import Annotated
from typing import Optional

import typer
from rich.console import RenderableType
from rich.status import Status
from rich.table import Column
from rich.table import Table

from anaconda_cli_base import console
from .clients import make_client
from .clients.base import GenericClient, QuantizedFile, Server, VectorDbTableSchema
from ._version import __version__

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")

CHECK_MARK = "[bold green]✔︎[/bold green]"


def get_running_servers(
    client: GenericClient, quantization: QuantizedFile
) -> list[Server]:
    try:
        servers = [
            s
            for s in client.servers.list()
            if s.serverConfig.model_name.endswith(quantization.identifier)
            and s.status == "running"
        ]
        return servers
    except AttributeError:
        return []


def _list_models(client: GenericClient) -> RenderableType:
    models = client.models.list()
    table = Table(
        Column("Model", no_wrap=True),
        "Params (B)",
        "Quantizations\ndownloaded in bold\ngreen when running",
        "Trained for",
        header_style="bold green",
    )
    for model in sorted(models, key=lambda m: m.name):
        quantizations = []
        for quant in model.quantized_files:
            if quant.is_downloaded:
                servers = get_running_servers(client, quant)
                color = "green" if servers else ""
                method = f"[bold {color}]{quant.quant_method}[/bold {color}]"
            else:
                method = f"[dim]{quant.quant_method}[/dim]"

            quantizations.append(method)

        quants = ", ".join(quantizations)

        parameters = f"{model.num_parameters/1e9:8.2f}"
        table.add_row(model.name, parameters, quants, model.trained_for)
    return table


def _model_info(client: GenericClient, model_id: str) -> RenderableType:
    info = client.models.get(model_id)

    table = Table.grid(padding=1, pad_edge=True)
    table.title = model_id
    table.add_column("Metadata", no_wrap=True, justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Description", info.description)
    parameters = f"{info.num_parameters/1e9:8.2f}B"
    table.add_row("Parameters", parameters)
    table.add_row("Trained For", info.trained_for)

    quantized = Table(
        Column("Filename", no_wrap=True),
        "Method",
        "Downloaded",
        "Max Ram (GB)",
        "Size (GB)",
        "Running",
        header_style="bold green",
    )
    for quant in info.quantized_files:
        method = quant.quant_method
        downloaded = CHECK_MARK if quant.is_downloaded else ""
        servers = get_running_servers(client, quant)
        running = CHECK_MARK if servers else ""

        ram = f"{quant.max_ram_usage / 1024 / 1024 / 1024:.2f}"
        size = f"{quant.size_bytes / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(quant.identifier, method, downloaded, ram, size, running)

    table.add_row("Quantized Files", quantized)
    return table


@app.command(name="version")
def version() -> None:
    """Version information of SDK and AI Navigator"""
    console.print(f"SDK: {__version__}")

    try:
        client = make_client()
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
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    """Model information"""
    client = make_client(backend)
    if model_id is None:
        renderable = _list_models(client)
    else:
        renderable = _model_info(client, model_id)
    console.print(renderable)


@app.command(name="download")
def download(
    model: str = typer.Argument(help="Model name with quantization"),
    force: bool = typer.Option(
        False, help="Force re-download of model if already downloaded."
    ),
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(help="Hard-link model file to this path after download"),
    ] = None,
) -> None:
    """Download a model"""
    client = make_client(backend)
    client.models.download(
        model, show_progress=True, force=force, console=console, path=output
    )
    console.print("[green]Success[/green]")


@app.command(name="remove")
def remove(
    model: str = typer.Argument(help="Model name with quantization"),
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    """Remove a downloaded a model"""
    client = make_client(backend)
    client.models.delete(model)
    console.print("[green]Success[/green]")


@app.command(
    name="launch",
    # context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch(
    model: str = typer.Argument(
        help="Name of the quantized model, it will download first if needed.",
    ),
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    detach: bool = typer.Option(
        default=False, help="Start model server and leave it running."
    ),
    show: Optional[bool] = typer.Option(
        False, help="Open your webbrowser when the server starts."
    ),
    force_download: bool = typer.Option(
        False, help="Download the model file even if it is already cached"
    ),
) -> None:
    """Launch an inference server for a model"""

    client = make_client(backend)
    client.models.download(model, force=force_download)

    text = f"{model} (creating)"
    with Status(text, console=console) as display:
        server = client.servers.create(
            model=model,
            # api_params=api_params,
            # load_params=load_params,
            # infer_params=infer_params,
        )
        client.servers.start(server)
        status = client.servers.status(server)
        text = f"{model} ({status})"
        display.update(text)

        while status != "running":
            status = client.servers.status(server)
            text = f"{model} ({status})"
            display.update(text)
    console.print(f"[bold green]✓[/] {text}", highlight=False)
    console.print(f"URL: [link='{server.url}']{server.url}[/link]")
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

        with Status(f"{model} (stopping)", console=console) as display:
            client.servers.stop(server)
            display.update(f"{model} (stopped)")
        return


@app.command("servers")
def servers(
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    """List running servers"""
    client = make_client(backend)
    servers = client.servers.list()

    table = Table(
        Column("Server ID", no_wrap=True),
        "Model Name",
        "Status",
        "OpenAI BaseURL",
        "Params",
        header_style="bold green",
    )

    for server in servers:
        params = server.serverConfig.model_dump_json(
            indent=2,
            exclude={
                "model_name",
            },
            exclude_none=True,
            exclude_defaults=True,
        )
        table.add_row(
            str(server.id),
            str(server.serverConfig.model_name),
            server.status,
            server.openai_url,
            params,
        )

    console.print(table)


@app.command("stop")
def stop(
    server: str = typer.Argument(help="ID of the server to stop"),
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
) -> None:
    client = make_client(backend)
    client.servers.stop(server)
    client.servers.delete(server)


@app.command("launch-vectordb")
def launch_vector_db() -> None:
    """
    Starts a vector db
    """
    client = make_client()
    result = client.vector_db.create()
    console.print(result)


@app.command("delete-vectordb")
def delete_vector_db() -> None:
    """
    Deletes the vector db
    """
    client = make_client()
    client.vector_db.delete()
    console.print("Vector db deleted")


@app.command("stop-vectordb")
def stop_vector_db() -> None:
    """
    Stops the vector db
    """
    client = make_client()
    result = client.vector_db.stop()
    console.print(result)


@app.command("list-tables")
def list_tables() -> None:
    """
    Lists all tables in the vector db
    """
    client = make_client()
    tables = client.vector_db.get_tables()
    console.print(tables)


@app.command("drop-table")
def drop_table(
    table: str = typer.Argument(help="Name of the table to drop"),
) -> None:
    """
    Drops a table from the vector db
    """
    client = make_client()
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
    client = make_client()
    validated_schema = VectorDbTableSchema.model_validate_json(schema)
    client.vector_db.create_table(table, validated_schema)
    console.print(f"Table {table} created")
