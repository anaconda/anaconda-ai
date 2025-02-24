from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer
from rich.status import Status
from rich.table import Column
from rich.table import Table

from anaconda_cli_base import console
from anaconda_models.client import get_default_client

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")


@app.command(name="list")
def models_list(
    downloaded_only: Annotated[
        bool,
        typer.Option(
            help="List only models where one or more quantizations have been downloaded"
        ),
    ] = False,
) -> None:
    """List models"""
    client = get_default_client()
    models = client.models.list()
    table = Table(
        Column("Model", no_wrap=True),
        "Type",
        "Params (B)",
        "Quantizations\n(downloaded in bold)",
        "Trained for",
        "License",
        header_style="bold green",
    )
    for model in sorted(models, key=lambda m: m.id):
        quantizations = []
        for quant in model.quantizedFiles:
            method = quant.quantMethod
            if quant.is_downloaded:
                method = f"[bold green]{method}[/bold green]"
                if downloaded_only:
                    quantizations.append(method)

            if not downloaded_only:
                quantizations.append(method)

        if downloaded_only and (not quantizations):
            continue
        quants = ", ".join(sorted(quantizations))

        parameters = f"{model.numParameters/1e9:8.2f}"
        table.add_row(
            model.modelId,
            model.modelType,
            parameters,
            quants,
            model.trainedFor,
            model.license,
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
    table.add_row("Description", info.description)
    table.add_row("Parameters", f"{info.numParameters/1e9:.2f} B")
    table.add_row("Context Window", str(info.contextWindowSize))
    table.add_row("Trained For", info.trainedFor)
    table.add_row("Model Type", info.modelType)
    table.add_row("Source URL", str(info.sourceUrl))
    table.add_row("License", info.license)
    table.add_row("First Published", info.firstPublished.isoformat())
    table.add_row("Languages", ", ".join(info.languages))

    quantized = Table(
        Column("Id", no_wrap=True),
        "Method",
        "Format",
        "Downloaded",
        "Evals",
        "Max Ram (GB)",
        "Size (GB)",
        header_style="bold green",
    )
    for quant in info.quantizedFiles:
        method = quant.quantMethod
        format = quant.format
        file_id = f"{model_id}_{method}.{format.lower()}"
        downloaded = "[bold green]✔︎[/bold green]" if quant.is_downloaded else ""

        evals = Table(show_header=False)
        for eval in sorted(quant.evaluations, key=lambda e: e.name):
            evals.add_row(eval.name, f"{eval.value:8.4f}")

        ram = f"{quant.maxRamUsage / 1024 / 1024 / 1024:.2f}"
        size = f"{quant.sizeBytes / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(file_id, method, format, downloaded, evals, ram, size)

    table.add_row("Quantized Files", quantized)

    console.print(table)


@app.command(name="download")
def models_download(
    model: str = typer.Argument(help="Model name with quantization"),
    directory: Optional[Path] = typer.Option(
        default=None,
        help="Directory into which the model will be downloaded. If different from the models_path",
    ),
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
        server = client.servers.create(model)
        status = client.servers.start(server)
        text = f"{model} ({status.status})"
        display.update(text)

        while status.status != "running":
            status = client.servers.start(server)
            text = f"{model} ({status.status})"
            display.update(text)
    console.print(f"[bold green]✓[/] {text}", highlight=False)
    console.print(f"[link='{server.url}']{server.url}[/link]")
    if show:
        import webbrowser

        webbrowser.open(server.url)

    while True:
        pass
