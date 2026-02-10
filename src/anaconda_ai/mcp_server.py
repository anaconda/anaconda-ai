"""MCP server for Anaconda AI: list models/servers, start/stop/remove servers."""

from __future__ import annotations

from typing import Any, Optional

from anaconda_ai.clients import AnacondaAIClient
from anaconda_ai.exceptions import AnacondaAIException, UnknownBackendError

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP server requires the mcp package. Install with: pip install 'anaconda-ai[mcp]'"
    )


NAME = "anaconda-ai"
DESCRIPTION = (
    "List Anaconda AI models and servers; start, stop, and remove inference servers."
)

mcp = FastMCP(
    NAME,
    # description=DESCRIPTION,
    json_response=True,
)


@mcp.tool()
def list_models(
    backend: Optional[str] = None,
    site: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List available Anaconda AI models and their quantizations.

    Returns model name, parameters, quantizations (method, downloaded, running), and trained_for.
    Use backend to choose 'ai-catalyst', 'ai-navigator', or 'anaconda-desktop' (default from config).
    """
    try:
        client = AnacondaAIClient(backend=backend, site=site)
        return [m.model_dump() for m in client.models.list()]
        # return _models_to_data(client)
    except (AnacondaAIException, UnknownBackendError) as e:
        return [{"error": str(e)}]


@mcp.tool()
def list_servers(
    backend: Optional[str] = None,
    site: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List running Anaconda AI inference servers.

    Returns server_id, model name, status, url, and openai_url for each server.
    """
    try:
        client = AnacondaAIClient(backend=backend, site=site)
        servers = client.servers.list()
        return [s.model_dump(exclude={"api_key"}) for s in servers]
        # return _servers_to_data(servers)
    except (AnacondaAIException, UnknownBackendError) as e:
        return [{"error": str(e)}]


@mcp.tool()
def start_server(
    model: str,
    backend: Optional[str] = None,
    site: Optional[str] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Create and start an inference server for a quantized model.

    model: Quantized model name, e.g. 'OpenHermes-2.5-Mistral-7B/Q4_K_M' or 'model_name/Q5_K_M'.
    Optionally pass backend/site. If name is provided it may be used as server name (backend-dependent).
    Downloads the model if needed, then starts the server. Returns server_id, model, status, openai_url.
    """
    try:
        client = AnacondaAIClient(backend=backend, site=site)
        extra_options: Optional[dict[str, Any]] = None
        if name is not None:
            extra_options = {"name": name}
        server = client.servers.create(
            model=model,
            extra_options=extra_options,
            show_progress=False,
        )
        server.start(show_progress=False, leave_running=True, wait=False)
        return {
            "server_id": str(server.id),
            "model": server.config.model_name,
            "status": server.status,
            "url": server.url,
            "openai_url": server.openai_url,
        }
    except (AnacondaAIException, UnknownBackendError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def stop_server(
    server_id: str,
    backend: Optional[str] = None,
    site: Optional[str] = None,
) -> dict[str, Any]:
    """Stop a running inference server by server ID.

    Does not remove the server configuration; use remove_server to delete it.
    """
    try:
        client = AnacondaAIClient(backend=backend, site=site)
        s = client.servers.get(server_id)
        if s.is_running:
            s.stop(show_progress=False)
        return {"status": "success", "server_id": server_id}
    except (AnacondaAIException, UnknownBackendError) as e:
        return {"error": str(e)}


@mcp.tool()
def remove_server(
    server_id: str,
    backend: Optional[str] = None,
    site: Optional[str] = None,
    stop_first: bool = True,
) -> dict[str, Any]:
    """Remove an inference server (delete its configuration).

    If stop_first is True (default), stops the server before removing.
    """
    try:
        client = AnacondaAIClient(backend=backend, site=site)
        s = client.servers.get(server_id)
        if stop_first and s.is_running:
            s.stop(show_progress=False)
        s.delete(show_progress=False)
        return {"status": "success", "server_id": server_id}
    except (AnacondaAIException, UnknownBackendError) as e:
        return {"error": str(e)}


def run(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the MCP server. Default: stdio. Use transport='streamable-http' for HTTP."""
    mcp.settings.port = port
    mcp.settings.host = host
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Anaconda AI MCP server")
    p.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport: stdio (default) or streamable-http",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="Host for HTTP (default 127.0.0.1)"
    )
    p.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP (default 8000)"
    )
    args = p.parse_args()
    run(transport=args.transport, host=args.host, port=args.port)
