import os
from typing import Generator, List, Optional
from typing_extensions import Annotated

import typer
from conda import plugins

from anaconda_models.core import AnacondaQuantizedModelCache

CONDA_ROOT = os.getenv("CONDA_ROOT")

app = typer.Typer(
    add_completion=False,
    help="Actions for Anaconda curated models",
    no_args_is_help=True,
)


@app.command("install", help="Install curated models into an environment")
def models_install(
    models: Annotated[List[str], typer.Argument(help="Quantized model name")],
    n: Annotated[Optional[str], typer.Option("-n", help="Environment name")] = None,
    p: Annotated[Optional[str], typer.Option("-p", help="Environment prefix")] = None,
) -> None:
    for name in models:
        model = AnacondaQuantizedModelCache(name=name)
        path = model.download()
    if n:
        os.link(path, "...")
    elif p:
        os.link(path, "...")
    else:
        os.link(path, "CONDA_PREFIX/...")


@app.command("list")
def models_list(): ...


@app.command("launch")
def models_launch(): ...


@plugins.hookimpl
def conda_subcommands() -> Generator[plugins.CondaSubcommand, None, None]:
    yield plugins.CondaSubcommand(
        name="models",
        summary="Anaconda Models integration",
        action=lambda args: app(args=args),
    )


# @plugins.hookimpl
# def conda_post_commands() -> Generator[plugins.CondaPostCommand, None, None]:
#     yield plugins.CondaPostCommand(
#         name="assist-search-recommendation",
#         action=recommend_assist_search,
#         run_for={"search"},
#     )


# @plugins.hookimpl
# def conda_pre_commands() -> Generator[plugins.CondaPreCommand, None, None]:
#     yield plugins.CondaPreCommand(
#         name="error-handler", action=error_handler, run_for=ALL_COMMANDS
#     )
