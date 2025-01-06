import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import intake
import typer
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Column
from rich.table import Table
from ruamel.yaml import YAML

from anaconda_cli_base import console
from anaconda_models.client import Client
from anaconda_models.catalog import AnacondaQuantizedModel
from anaconda_models.core import AnacondaQuantizedModelCache
from anaconda_models.core import get_models
from anaconda_models.core import model_info

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")


def _args_to_kwargs(args: list[str]) -> dict:
    kwargs: dict = {}
    for arg in args:
        if arg.startswith("--"):
            if "=" in arg:
                k, v = arg[2:].split("=", maxsplit=1)
                if k == "system-prompt-file":
                    import fsspec

                    path = fsspec.open_local(f"simplecache::{v}")
                    v = path
            else:
                k = arg[2:]
                v = ""
            kwargs[k] = v

    return kwargs


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
    client = Client()
    models = get_models(client=client)
    table = Table(
        Column("Model", no_wrap=True),
        "Type",
        "Params (B)",
        "Quantizations\n(downloaded in bold)",
        "Trained for",
        "License",
        header_style="bold green",
    )
    for model in sorted(models, key=lambda m: m["id"]):
        quantizations = []
        for quant in model["quantizedFiles"]:
            method = quant["quantMethod"]
            cacher = AnacondaQuantizedModelCache(
                name=model["id"], quantization=method, client=client
            )
            if cacher.is_cached:
                method = f"[bold green]{method}[/bold green]"
                if downloaded_only:
                    quantizations.append(method)

            if not downloaded_only:
                quantizations.append(method)

        if downloaded_only and (not quantizations):
            continue
        quants = ", ".join(sorted(quantizations))

        parameters = f"{model['numParameters']/1e9:8.2f}"
        table.add_row(
            model["modelId"],
            model["modelType"],
            parameters,
            quants,
            model["trainedFor"],
            model["license"],
        )
    console.print(table)


@app.command(name="info")
def models_info(model_id: str = typer.Argument(help="Model id")) -> None:
    """Information about a single model"""
    client = Client()
    info = model_info(model_id, client=client)
    if info is None:
        console.print(f"{model_id} not found")
        return

    table = Table.grid(padding=1, pad_edge=True)
    table.title = model_id
    table.add_column("Metadata", no_wrap=True, justify="center", style="bold green")
    table.add_column("Value", justify="left")
    table.add_row("Description", info["description"])
    table.add_row("Parameters", f"{info['numParameters']/1e9:.2f} B")
    table.add_row("Context Window", str(info["contextWindowSize"]))
    table.add_row("Trained For", info["trainedFor"])
    table.add_row("Model Type", info["modelType"])
    table.add_row("Source URL", info["sourceUrl"])
    table.add_row("License", info["license"])
    table.add_row("First Published", info["firstPublished"])
    table.add_row("Languages", ", ".join(info["languages"]))

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
    for quant in sorted(info["quantizedFiles"], key=lambda q: q["quantMethod"]):
        method = quant["quantMethod"]
        format = quant["format"]
        file_id = f"{model_id}_{method}.{format.lower()}"
        cacher = AnacondaQuantizedModelCache(
            name=info["id"], quantization=method, client=client
        )
        downloaded = "[bold green]✔︎[/bold green]" if cacher.is_cached else ""

        evals = Table(show_header=False)
        for eval in sorted(quant["evaluations"], key=lambda e: e["name"]):
            evals.add_row(eval["name"], f"{eval['value']:8.4f}")

        ram = f"{quant['maxRamUsage'] / 1024 / 1024 / 1024:.2f}"
        size = f"{quant['sizeBytes'] / 1024 / 1024 / 1024:.2f}"
        quantized.add_row(file_id, method, format, downloaded, evals, ram, size)

    table.add_row("Quantized Files", quantized)

    console.print(table)


DEFAULT_CATALOG_PATH = Path("./catalog.yaml")


@app.command(name="download")
def models_download(
    model: Optional[str] = typer.Argument(
        default=None, help="Name of the quantized model or catalog entry"
    ),
    directory: Optional[Path] = typer.Option(
        default=None,
        help="Directory into which the model will be downloaded. Default is the AI Navigator download path.",
    ),
    catalog: Optional[Path] = typer.Option(
        default=None, help="Path to catalog file with model alias"
    ),
    force: bool = typer.Option(
        False, help="Force re-download of model if already downloaded."
    ),
) -> None:
    """Download a model"""
    if (model is None) and (catalog is None):
        catalog = DEFAULT_CATALOG_PATH
        model = "model"
        if not catalog.exists():
            print(f"{catalog} not found")
            raise typer.Abort(f"{catalog} not found")

    if catalog is None and (model is not None):
        model_cache = AnacondaQuantizedModelCache(model, directory=directory)
    else:
        model = "model" if model is None else model
        cat = intake.from_yaml_file(str(catalog))
        if model not in cat:
            print("no model entry")
            raise typer.Abort()
        entry = cat[model]
        if not isinstance(entry, AnacondaQuantizedModel):
            print(
                "model entry is not of the type anaconda_models.catalog.AnacondaQuantizedModel"
            )
            raise typer.Abort()

        reader = entry.to_reader("AnacondaQuantizedModelReader")
        model_cache = reader.read()

    path = model_cache.download(force=force)
    print(path)


@app.command(
    name="launch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def models_launch(
    ctx: typer.Context,
    model: Optional[str] = typer.Argument(
        default=None,
        help="Name of the quantized model or catalog entry, it will download first if needed.",
    ),
    run_on: Optional[str] = typer.Option(default=None, help="Where to run the service"),
    catalog: Optional[Path] = typer.Option(default=None, help="Path to catalog file"),
    show: Optional[bool] = typer.Option(
        True, help="Open your webbrowser when the inference service starts."
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
    if (model is None) and (catalog is None):
        catalog = DEFAULT_CATALOG_PATH
        model = "model"
        if not catalog.exists():
            print(f"{catalog} not found")
            raise typer.Abort(f"Default catalog path {catalog} not found")

    if catalog is None and (model is not None):
        cacher = AnacondaQuantizedModelCache(model)
        llama_cpp_kwargs = {}
        model_id = model
    else:
        model = "model" if model is None else model
        cat = intake.from_yaml_file(str(catalog))
        if model not in cat:
            print("no model entry")
            raise typer.Abort()
        entry = cat[model]
        if isinstance(entry, AnacondaQuantizedModel):
            cacher = entry.to_reader("AnacondaQuantizedModelCache").read()
            llama_cpp_kwargs = entry.llama_cpp_options
            model_id = entry.name
        else:
            cacher = cat[model].read()
            llama_cpp_kwargs = {}
            model_id = model

    parsed_kwargs = _args_to_kwargs(ctx.args)
    kwargs = {**parsed_kwargs, **llama_cpp_kwargs, **{"port": port}}

    if force_download:
        cacher.download(force=True)

    server = cacher.start(run_on=run_on, **kwargs)
    if show:
        server.open()

    table = Table.grid(padding=1, pad_edge=True)
    table.title = f"{model_id} llama.cpp server"
    table.add_column("    ", no_wrap=True, justify="center", style="bold green")
    table.add_column("    ")
    table.add_row("URL", f"[link='{server.url}']{server.url}[/link]")
    console.print(table)

    logs = Table.grid(padding=1)
    logs.title = "Server Logs"
    if show_logs and server.options.get("log_file", False):
        with open(server.options["log_file"]) as f:
            console.log(escape(f.read()))
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline().strip()
                if not line:
                    time.sleep(0.2)
                    continue
                console.log(escape(line))
    else:
        while True:
            pass


class Exports(Enum):
    intake = "intake"
    ollama = "ollama"


@app.command(
    name="export",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def models_export(
    ctx: typer.Context,
    model: str = typer.Argument(
        help="Name of the quantized model.",
    ),
    to: Exports = typer.Option(help="Export model to various formats and frameworks."),
) -> None:
    """Export model file and configuration"""
    if to.value == "intake":
        entry = AnacondaQuantizedModel(model, metadata={})
        reader = entry.to_reader()
        server = entry.auto_pipeline("LlamaCPPService")
        parsed_kwargs = _args_to_kwargs(ctx.args)
        server.steps[-1][2].update(parsed_kwargs)

        catalog = intake.entry.Catalog()
        catalog["model"] = reader
        catalog["server"] = server
        catalog.aliases = {"model": "model", "server": "server"}

        yaml = YAML(typ="safe")
        contents = yaml.dump(catalog.to_dict())
        lexer = Syntax(contents, "yaml")
        console.print(lexer)
    elif to.value == "ollama":
        anaconda_model = AnacondaQuantizedModelCache(name=model)
        path = anaconda_model.download()

        kwargs: dict[str, str] = _args_to_kwargs(args=ctx.args)

        system_prompt = kwargs.pop("system", None)
        if system_prompt:
            system_prompt = f'SYSTEM """{system_prompt}"""'
        else:
            system_prompt = ""

        parameters = []
        for k, v in kwargs.items():
            param = f"PARAMETER {k} {v}"
            parameters.append(param)
        params = "\n".join(parameters)

        modelfile = f"""\
# Path to cached model
from {path}

# Inference parameters
{params}

# System Prompt
{system_prompt}

# License Information
LICENSE \"\"\"{anaconda_model.metadata['license']}\"\"\"
"""

        console.print(modelfile)
