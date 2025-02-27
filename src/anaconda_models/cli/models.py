from typing import Optional

import typer
from rich.status import Status
from rich.table import Column
from rich.table import Table

from anaconda_cli_base import console
from anaconda_models.clients import get_default_client

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")


@app.command(name="list")
def models_list() -> None:
    """List models"""
    client = get_default_client()
    models = client.models.list()
    table = Table(
        Column("Model", no_wrap=True),
        "Params (B)",
        "Quantizations\n(downloaded in bold)",
        "Trained for",
        header_style="bold green",
    )
    for model in sorted(models, key=lambda m: m.id):
        quantizations = []
        for quant in model.metadata.quantizations:
            if quant.isDownloaded:
                method = f"[bold green]{quant.method}[/bold green]"
            else:
                method = quant.method

            quantizations.append(method)

        quants = ", ".join(quantizations)

        parameters = model.metadata.numParameters
        table.add_row(
            model.id,
            parameters,
            quants,
            model.metadata.trainedFor,
        )
    console.print(table)


@app.command(name="info")
def models_info(model_id: str = typer.Argument(help="Model id")) -> None:
    """Information about a single model"""
    client = get_default_client()
    info = client.models.get(model_id)
    if info is None:
        console.print(f"{model_id} not found")
        return

    table = Table.grid(padding=1, pad_edge=True)
    table.title = model_id
    table.add_column("Metadata", no_wrap=True, justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Description", info.metadata.description)
    table.add_row("Parameters", info.metadata.numParameters)
    table.add_row("Trained For", info.metadata.trainedFor)

    quantized = Table(
        Column("Filename", no_wrap=True),
        "Method",
        "Downloaded",
        "Max Ram (GB)",
        "Size (GB)",
        header_style="bold green",
    )
    for quant in info.metadata.quantizations:
        method = quant.method
        downloaded = "[bold green]✔︎[/bold green]" if quant.isDownloaded else ""

        ram = f"{quant.maxRamUsage / 1024 / 1024 / 1024:.2f}"
        size = f"{quant.sizeBytes / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(quant.modelFileName, method, downloaded, ram, size)

    table.add_row("Quantized Files", quantized)

    console.print(table)


@app.command(name="download")
def models_download(
    model: str = typer.Argument(help="Model name with quantization"),
    force: bool = typer.Option(
        False, help="Force re-download of model if already downloaded."
    ),
) -> None:
    """Download a model"""
    client = get_default_client()
    path = client.models.download(model, force=force)
    console.print(path)


@app.command(
    name="launch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def models_launch(
    ctx: typer.Context,
    model: str = typer.Argument(
        help="Name of the quantized model or catalog entry, it will download first if needed.",
    ),
    detach: bool = typer.Option(
        default=False, help="Start model server and leave it running."
    ),
    show: Optional[bool] = typer.Option(
        False, help="Open your webbrowser when the inference service starts."
    ),
    port: Optional[int] = typer.Option(
        0,
        help="Port number for the inference service. Default is to find a free open port",
    ),
    force_download: bool = typer.Option(
        False, help="Download the model file even if it is already cached"
    ),
    show_logs: Optional[bool] = typer.Option(True, help="Stream server logs"),
) -> None:
    """Launch an inference server for a model"""
    text = f"{model} (creating)"
    with Status(text, console=console) as display:
        client = get_default_client()
        server = client.servers.create(model, api_params={"port": port})
        status = client.servers.start(server)
        text = f"{model} ({status.status})"
        display.update(text)

        while status.status != "running":
            status = client.servers.start(server)
            text = f"{model} ({status.status})"
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
        with Status(f"{model} (stopping)", console=console) as display:
            client.servers.stop(server)
        return
