import json
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

from anaconda_ai.config import AnacondaAIConfig
from anaconda_cli_base import console
from .clients import AnacondaAIClient, clients
from .clients.base import GenericClient, Server, VectorDbTableSchema
from ._version import __version__

app = typer.Typer(add_completion=False, help="Actions for Anaconda curated models")

CHECK_MARK = "[bold green]✔︎[/bold green]"
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
def launch_vector_db(
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Starts a vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
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
def delete_vector_db(
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Deletes the vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
    client.vector_db.delete()
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("stop-vectordb")
def stop_vector_db(
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Stops the vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
    _ = client.vector_db.stop()
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("list-tables")
def list_tables(
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Lists all tables in the vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
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
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Drops a table from the vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
    client.vector_db.drop_table(table)
    if as_json:
        console.print_json(data={"status": "success"})
    else:
        console.print("[green]Success[/green]")


@app.command("create-table")
def create_table(
    table: str = typer.Argument(help="Name of the table to create"),
    schema: str = typer.Argument(help="Schema of the table to create"),
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option(help="Select inference backend")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """
    Creates a table in the vector db
    """
    client = AnacondaAIClient(backend=backend, site=site)
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
    no_args_is_help=True,
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


@app.command("stage", no_args_is_help=True)
def stage(
    model: Annotated[
        Optional[str],
        typer.Argument(
            help="Model name with quantization (e.g. Qwen2.5-7B-Instruct/Q4_K_M)"
        ),
    ] = None,
    list_staged: Annotated[
        bool, typer.Option("--list", is_flag=True, help="List staged models in S3")
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", is_flag=True, help="Re-upload even if already staged"),
    ] = False,
    bucket: Annotated[Optional[str], typer.Option(help="S3 bucket override")] = None,
    aws_profile: Annotated[Optional[str], typer.Option(help="AWS profile name")] = None,
    site: Annotated[
        Optional[str], typer.Option("--at", help="Site defined in config")
    ] = None,
    as_json: AS_JSON = False,
) -> None:
    """Stage a model to S3 for SageMaker deployment"""
    try:
        from anaconda_ai.integrations.sagemaker import (
            AnacondaModel,
            _resolve_stage_config,
        )
    except ImportError as e:
        console.print(
            "[red]SageMaker integration requires the sagemaker-core package.[/] "
            "Install with: [bold]pip install 'anaconda-ai[sagemaker]'[/]"
        )
        raise typer.Exit(1) from e

    import boto3

    boto_session = boto3.Session(profile_name=aws_profile)

    if list_staged:
        cfg = _resolve_stage_config(boto_session, bucket)
        s3 = boto_session.client("s3")
        prefix = cfg.prefix.rstrip("/") + "/" if cfg.prefix else ""

        resp = s3.list_objects_v2(Bucket=cfg.bucket, Prefix=prefix, Delimiter="/")
        models = []
        for common_prefix in resp.get("CommonPrefixes", []):
            model_prefix = common_prefix["Prefix"]
            sub = s3.list_objects_v2(Bucket=cfg.bucket, Prefix=model_prefix)
            for obj in sub.get("Contents", []):
                if obj["Key"].endswith(".gguf"):
                    model_path = model_prefix[len(prefix) :].rstrip("/")
                    size_gb = obj["Size"] / (1024**3)
                    models.append(
                        {
                            "model": model_path,
                            "size_gb": round(size_gb, 2),
                            "s3_uri": f"s3://{cfg.bucket}/{obj['Key']}",
                        }
                    )

        if as_json:
            console.print_json(data=models)
        else:
            if not models:
                console.print(f"No staged models in s3://{cfg.bucket}/{prefix}")
            else:
                table = Table("Model", "Size (GB)", "S3 URI", header_style="bold green")
                for m in models:
                    table.add_row(m["model"], f"{m['size_gb']:.2f}", m["s3_uri"])
                console.print(table)
        return

    if model is None:
        console.print("[red]Provide a model name or use --list[/]")
        raise typer.Exit(1)

    sm = AnacondaModel(model_id=model, site=site, aws_profile=aws_profile)
    s3_uri = sm.stage(bucket=bucket, force=force)

    if as_json:
        console.print_json(data={"status": "success", "s3_uri": s3_uri})
    else:
        console.print(f"[green]Success[/green] {s3_uri}")


@app.command("deploy", no_args_is_help=True)
def deploy(
    model: str = typer.Argument(help="Model name with quantization"),
    instance_type: Annotated[
        str,
        typer.Option(
            help="SageMaker instance type (e.g. ml.g5.2xlarge)",
            rich_help_panel="Endpoint",
        ),
    ] = "ml.g5.2xlarge",
    image_uri: Annotated[
        str, typer.Option(help="Container image ECR URI", rich_help_panel="Endpoint")
    ] = ...,  # type: ignore[assignment]
    endpoint_name: Annotated[
        Optional[str],
        typer.Option(
            help="Endpoint name (auto-generated if omitted)", rich_help_panel="Endpoint"
        ),
    ] = None,
    role: Annotated[
        Optional[str],
        typer.Option(help="SageMaker execution role ARN", rich_help_panel="AWS"),
    ] = None,
    stage: Annotated[
        bool,
        typer.Option(
            "--stage",
            is_flag=True,
            help="Stage model to S3 before deploying (faster cold start)",
            rich_help_panel="Staging",
        ),
    ] = False,
    build_only: Annotated[
        bool,
        typer.Option(
            "--build-only",
            is_flag=True,
            help="Register model only, do not create endpoint",
            rich_help_panel="Endpoint",
        ),
    ] = False,
    bucket: Annotated[
        Optional[str],
        typer.Option(help="S3 bucket for staging", rich_help_panel="Staging"),
    ] = None,
    aws_profile: Annotated[
        Optional[str], typer.Option(help="AWS profile name", rich_help_panel="AWS")
    ] = None,
    ctx_size: Annotated[
        Optional[int],
        typer.Option(help="Context window size", rich_help_panel="llama-server"),
    ] = None,
    n_gpu_layers: Annotated[
        Optional[int],
        typer.Option(help="GPU layers to offload", rich_help_panel="llama-server"),
    ] = None,
    parallel: Annotated[
        Optional[int],
        typer.Option(help="Parallel inference slots", rich_help_panel="llama-server"),
    ] = None,
    flash_attn: Annotated[
        Optional[bool],
        typer.Option(
            "--flash-attn/--no-flash-attn",
            help="Flash attention",
            rich_help_panel="llama-server",
        ),
    ] = None,
    cont_batching: Annotated[
        Optional[bool],
        typer.Option(
            "--cont-batching/--no-cont-batching",
            help="Continuous batching",
            rich_help_panel="llama-server",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(help="Logical batch size", rich_help_panel="llama-server"),
    ] = None,
    ubatch_size: Annotated[
        Optional[int],
        typer.Option(help="Physical batch size", rich_help_panel="llama-server"),
    ] = None,
    cache_type_k: Annotated[
        Optional[str],
        typer.Option(
            help="KV cache type for K (f16, q8_0, q4_0)", rich_help_panel="llama-server"
        ),
    ] = None,
    cache_type_v: Annotated[
        Optional[str],
        typer.Option(
            help="KV cache type for V (f16, q8_0, q4_0)", rich_help_panel="llama-server"
        ),
    ] = None,
    mlock: Annotated[
        Optional[bool],
        typer.Option(
            "--mlock/--no-mlock",
            help="Lock model in RAM",
            rich_help_panel="llama-server",
        ),
    ] = None,
    jinja: Annotated[
        Optional[bool],
        typer.Option(
            "--jinja/--no-jinja",
            help="Jinja template engine",
            rich_help_panel="llama-server",
        ),
    ] = None,
    reasoning: Annotated[
        Optional[str],
        typer.Option(
            help="Reasoning mode (on, off, auto)", rich_help_panel="llama-server"
        ),
    ] = None,
    reasoning_budget: Annotated[
        Optional[int],
        typer.Option(help="Thinking token budget", rich_help_panel="llama-server"),
    ] = None,
    volume_size: Annotated[
        Optional[int],
        typer.Option(help="EBS volume size in GB", rich_help_panel="Endpoint"),
    ] = None,
    routing_strategy: Annotated[
        str,
        typer.Option(
            help="Traffic routing (LEAST_OUTSTANDING_REQUESTS or RANDOM)",
            rich_help_panel="Endpoint",
        ),
    ] = "LEAST_OUTSTANDING_REQUESTS",
    kms_key_id: Annotated[
        Optional[str],
        typer.Option(help="KMS key ARN for encryption", rich_help_panel="Endpoint"),
    ] = None,
    security_group_ids: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated security group IDs for VPC",
            rich_help_panel="Endpoint",
        ),
    ] = None,
    subnets: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated subnet IDs for VPC", rich_help_panel="Endpoint"
        ),
    ] = None,
    site: Annotated[
        Optional[str],
        typer.Option("--at", help="Site defined in config", rich_help_panel="Anaconda"),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            is_flag=True,
            help="Print output as JSON",
            rich_help_panel="Anaconda",
        ),
    ] = False,
) -> None:
    """Deploy a model to SageMaker"""
    try:
        from anaconda_ai.integrations.sagemaker import AnacondaModel
    except ImportError as e:
        console.print(
            "[red]SageMaker integration requires the sagemaker-core package.[/] "
            "Install with: [bold]pip install 'anaconda-ai[sagemaker]'[/]"
        )
        raise typer.Exit(1) from e

    sm = AnacondaModel(
        model_id=model,
        site=site,
        role=role,
        image_uri=image_uri,
        aws_profile=aws_profile,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        parallel=parallel,
        flash_attn=flash_attn,
        cont_batching=cont_batching,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        mlock=mlock,
        jinja=jinja,
        reasoning=reasoning,
        reasoning_budget=reasoning_budget,
    )

    if build_only:
        sm_model = sm.build(stage=stage, stage_bucket=bucket)
        model_name = sm_model.model_name
        region = sm._boto_session.region_name

        if as_json:
            console.print_json(
                data={
                    "status": "success",
                    "model_name": model_name,
                    "region": region,
                    "console_url": f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/models/{model_name}",
                }
            )
        else:
            console.print(f"""\n[bold green]✓[/] Model [bold]{model_name}[/] registered

[bold]Console:[/]
  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/models/{model_name}""")
        return

    vpc = None
    if security_group_ids and subnets:
        from sagemaker.core.shapes.shapes import VpcConfig

        vpc = VpcConfig(
            security_group_ids=security_group_ids.split(","),
            subnets=subnets.split(","),
        )

    endpoint = sm.deploy(
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        stage=stage,
        stage_bucket=bucket,
        volume_size_in_gb=volume_size,
        routing_strategy=routing_strategy,
        vpc_config=vpc,
        kms_key_id=kms_key_id,
    )

    ep = endpoint.endpoint_name
    region = sm._boto_session.region_name
    profile = aws_profile or "default"

    if as_json:
        console.print_json(
            data={
                "status": "success",
                "endpoint_name": ep,
                "region": region,
                "console_url": f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{ep}",
            }
        )
    else:
        console.print(f"""\n[bold green]✓[/] Endpoint [bold]{ep}[/] InService

[bold]Python:[/]
  from sagemaker.core.resources import Endpoint
  import boto3, json

  endpoint = Endpoint.get("{ep}", session=boto3.Session(profile_name="{profile}"), region="{region}")
  response = endpoint.invoke(
      body=json.dumps({{"messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 256}}),
      content_type="application/json",
  )
  print(json.loads(response.body))

[bold]AWS CLI:[/]
  aws sagemaker-runtime invoke-endpoint --profile {profile} --region {region} \\
    --endpoint-name {ep} \\
    --content-type application/json \\
    --cli-binary-format raw-in-base64-out \\
    --body '{{"messages":[{{"role":"user","content":"Hello"}}],"max_tokens":256}}' \\
    /dev/stderr 1>/dev/null 2>&1

[bold]Console:[/]
  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{ep}""")
