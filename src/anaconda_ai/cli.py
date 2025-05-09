from typing import Annotated
from typing import Optional

import typer
from rich.console import RenderableType
from rich.status import Status
from rich.table import Column
from rich.table import Table

from anaconda_cli_base import console
from .clients import get_default_client
from .clients.base import GenericClient, ModelQuantization, Server, VectorDbTableSchema
from ._version import __version__
app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")

CHECK_MARK = "[bold green]✔︎[/bold green]"


def get_running_servers(
    client: GenericClient, quantization: ModelQuantization
) -> list[Server]:
    servers = [
        s
        for s in client.servers.list()
        if s.serverConfig.modelFileName.endswith(quantization.modelFileName)
        and s.status == "running"
    ]
    return servers


def _list_models(client: GenericClient) -> RenderableType:
    models = client.models.list()
    table = Table(
        Column("Model", no_wrap=True),
        "Params (B)",
        "Quantizations\ndownloaded in bold\ngreen when running",
        "Trained for",
        header_style="bold green",
    )
    for model in sorted(models, key=lambda m: m.id):
        quantizations = []
        for quant in model.metadata.files:
            if quant.isDownloaded:
                servers = get_running_servers(client, quant)
                color = "green" if servers else ""
                method = f"[bold {color}]{quant.method}[/bold {color}]"
            else:
                method = f"[dim]{quant.method}[/dim]"

            quantizations.append(method)

        quants = ", ".join(quantizations)

        parameters = f"{model.metadata.numParameters/1e9:8.2f}"
        table.add_row(model.id, parameters, quants, model.metadata.trainedFor)
    return table


def _model_info(client: GenericClient, model_id: str) -> RenderableType:
    info = client.models.get(model_id)

    table = Table.grid(padding=1, pad_edge=True)
    table.title = model_id
    table.add_column("Metadata", no_wrap=True, justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Description", info.metadata.description)
    parameters = f"{info.metadata.numParameters/1e9:8.2f}B"
    table.add_row("Parameters", parameters)
    table.add_row("Trained For", info.metadata.trainedFor)

    quantized = Table(
        Column("Filename", no_wrap=True),
        "Method",
        "Downloaded",
        "Max Ram (GB)",
        "Size (GB)",
        "Running",
        header_style="bold green",
    )
    for quant in info.metadata.files:
        method = quant.method
        downloaded = CHECK_MARK if quant.isDownloaded else ""
        servers = get_running_servers(client, quant)
        running = CHECK_MARK if servers else ""

        ram = f"{quant.maxRamUsage / 1024 / 1024 / 1024:.2f}"
        size = f"{quant.sizeBytes / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(quant.modelFileName, method, downloaded, ram, size, running)

    table.add_row("Quantized Files", quantized)
    return table

@app.command(name="version")
def version() -> None:
    """Version information of SDK and AI Navigator"""
    console.print(f"SDK: {__version__}")

    try:
        client = get_default_client()
        version = client.get_version()
        console.print(version)
    except Exception as e:
        console.print("AI Navigator not found. Is it running?")

@app.command(name="models")
def models(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="Optional Model name for detailed information"),
    ] = None,
) -> None:
    """Model information"""
    client = get_default_client()
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
) -> None:
    """Download a model"""
    client = get_default_client()
    client.models.download(model, show_progress=True, force=force, console=console)
    console.print("[green]Success[/green]")


@app.command(name="remove")
def remove(
    model: str = typer.Argument(help="Model name with quantization"),
) -> None:
    """Remove a downloaded a model"""
    client = get_default_client()
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
    detach: bool = typer.Option(
        default=False, help="Start model server and leave it running."
    ),
    show: Optional[bool] = typer.Option(
        False, help="Open your webbrowser when the server starts."
    ),
    port: Optional[int] = typer.Option(
        0,
        help="Port number for the server. Default is to find a free open port",
    ),
    force_download: bool = typer.Option(
        False, help="Download the model file even if it is already cached"
    ),
    api_key: Optional[str] = None,
    log_disable: Optional[bool] = None,
    mmproj: Optional[str] = None,
    timeout: Optional[int] = None,
    verbose: Optional[bool] = None,
    main_gpu: Optional[int] = None,
    metrics: Optional[bool] = None,
    batch_size: Optional[int] = None,
    cont_batching: Optional[bool] = None,
    ctx_size: Optional[int] = None,
    memory_f32: Optional[bool] = None,
    mlock: Optional[bool] = None,
    n_gpu_layers: Optional[int] = None,
    rope_freq_base: Optional[int] = None,
    rope_freq_scale: Optional[int] = None,
    seed: Optional[int] = None,
    tensor_split: Optional[str] = None,
    use_mmap: Optional[bool] = None,
    embedding: Optional[bool] = None,
    threads: Optional[int] = None,
    n_predict: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repeat_last: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    temp: Optional[float] = None,
    parallel: Optional[int] = None,
) -> None:
    """Launch an inference server for a model"""

    client = get_default_client()
    client.models.download(model, force=force_download)

    api_params = {
        "port": port,
        "api_key": api_key,
        "log_disable": log_disable,
        "mmproj": mmproj,
        "timeout": timeout,
        "verbose": verbose,
        "metrics": metrics,
    }

    if tensor_split:
        try:
            split_tensors = [float(i) for i in tensor_split.split(",")]
        except ValueError:
            raise ValueError("--split-tensors must be a comma separated lit of floats")
    else:
        split_tensors = None

    load_params = {
        "batch_size": batch_size,
        "cont_batching": cont_batching,
        "ctx_size": ctx_size,
        "main_gpu": main_gpu,
        "memory_f32": memory_f32,
        "mlock": mlock,
        "n_gpu_layers": n_gpu_layers,
        "rope_freq_base": rope_freq_base,
        "rope_freq_scale": rope_freq_scale,
        "seed": seed,
        "tensor_split": split_tensors,
        "use_mmap": use_mmap,
        "embedding": embedding,
    }

    infer_params = {
        "threads": threads,
        "n_predict": n_predict,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_last": repeat_last,
        "repeat_penalty": repeat_penalty,
        "temp": temp,
        "parallel": parallel,
    }

    text = f"{model} (creating)"
    with Status(text, console=console) as display:
        server = client.servers.create(
            model=model,
            api_params=api_params,
            load_params=load_params,
            infer_params=infer_params,
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

    if detach:
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
def servers() -> None:
    """List running servers"""
    client = get_default_client()
    servers = client.servers.list()

    table = Table(
        Column("ID", no_wrap=True),
        "Model",
        "URL",
        "Params",
        header_style="bold green",
    )

    for server in servers:
        if not server.is_running:
            continue

        params = server.serverConfig.model_dump_json(
            indent=2,
            exclude={"modelFileName", "logsDir", "apiParams"},
            exclude_none=True,
            exclude_defaults=True,
        )
        table.add_row(
            str(server.id),
            str(server.serverConfig.modelFileName),
            server.openai_url,
            params,
        )

    console.print(table)


@app.command("stop")
def stop(
    server: str = typer.Argument(help="ID of the server to stop"),
) -> None:
    client = get_default_client()
    client.servers.stop(server)


@app.command("launch-vectordb")
def launch_vector_db(
) -> None:
    """
    Starts a vector db
    """
    client = get_default_client()
    result = client.vector_db.create()
    console.print(result)

@app.command("delete-vectordb")
def delete_vector_db(
) -> None:
    """
    Deletes the vector db
    """
    client = get_default_client()
    client.vector_db.delete()
    console.print("Vector db deleted")

@app.command("stop-vectordb")
def stop_vector_db(
) -> None:
    """
    Stops the vector db
    """
    client = get_default_client()
    result = client.vector_db.stop()
    console.print(result)

@app.command("list-tables")
def list_tables(
) -> None:
    """
    Lists all tables in the vector db
    """
    client = get_default_client()
    tables = client.vector_db.get_tables()
    console.print(tables)

@app.command("drop-table")
def drop_table(
    table: str = typer.Argument(help="Name of the table to drop"),
) -> None:
    """
    Drops a table from the vector db
    """
    client = get_default_client()
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
    client = get_default_client()
    validated_schema = VectorDbTableSchema.model_validate_json(schema)
    client.vector_db.create_table(table, validated_schema)
    console.print(f"Table {table} created")
