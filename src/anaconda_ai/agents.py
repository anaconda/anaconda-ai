"""Agent definitions for coding agent wrapper commands.

Each supported coding agent (Claude Code, OpenCode, etc.) is represented
by an AgentDefinition that specifies how to find the binary, build the
environment variables, and provide installation guidance.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .clients.base import Server


@dataclass
class AgentDefinition:
    """Defines a supported coding agent and how to configure it."""

    name: str
    binary: str
    build_env: Callable[[Server, str], Dict[str, str]]
    install_hint: str


def find_agent_binary(binary_name: str) -> Optional[str]:
    """Find the agent binary on the system PATH.

    Returns the full path to the binary, or None if not found.
    """
    return shutil.which(binary_name)


def build_env_claude(server: Server, model_name: str) -> Dict[str, str]:
    """Build environment variables for Claude Code.

    Claude Code uses the Anthropic API format. All Anaconda AI backends
    serve the /v1/messages route at their base URL.

    Uses server.url (raw base URL) — Claude Code appends /v1/messages itself.
    """
    return {
        "ANTHROPIC_BASE_URL": server.url,
        "ANTHROPIC_API_KEY": server.api_key or "not-needed",
    }


def build_env_opencode(server: Server, model_name: str) -> Dict[str, str]:
    """Build environment variables for OpenCode.

    OpenCode requires inline JSON configuration via OPENCODE_CONFIG_CONTENT
    with a custom provider pointing at the server's OpenAI-compatible endpoint.

    Uses server.openai_url (base URL + /v1) for the OpenAI-compatible endpoint.
    """
    config = {
        "provider": {
            "anaconda": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": server.openai_url,
                    "apiKey": server.api_key or "not-needed",
                },
                "models": {
                    model_name: {"name": model_name},
                },
            }
        },
        "model": f"anaconda/{model_name}",
    }
    return {
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
    }


# Agent registry — keyed by CLI command name
AGENTS: Dict[str, AgentDefinition] = {
    "claude": AgentDefinition(
        name="Claude Code",
        binary="claude",
        build_env=build_env_claude,
        install_hint="Install with: npm install -g @anthropic-ai/claude-code",
    ),
    "opencode": AgentDefinition(
        name="OpenCode",
        binary="opencode",
        build_env=build_env_opencode,
        install_hint="Install from: https://opencode.ai",
    ),
}
