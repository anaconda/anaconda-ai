"""Agent definitions for coding agent wrapper commands.

Each supported coding agent (Claude Code, OpenCode, etc.) is represented
by an AgentDefinition that specifies how to find the binary, build the
environment variables, and provide installation guidance.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .clients.base import Server


@dataclass
class AgentDefinition:
    """Defines a supported coding agent and how to configure it."""

    name: str
    binary: str
    build_env: Callable[[Server, str], Dict[str, str]]
    build_agent_args: Callable[[str], List[str]]
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


def build_agent_args_claude(model_name: str) -> List[str]:
    """Build extra CLI arguments for Claude Code.

    Claude Code accepts --model to set the model for the session. While the
    local server ignores the model field in API requests, passing it ensures
    Claude Code's UI displays the correct model name and future-proofs against
    servers that may validate it.
    """
    return ["--model", model_name]


def build_agent_args_opencode(model_name: str) -> List[str]:
    """Build extra CLI arguments for OpenCode.

    OpenCode's TUI accepts --model as the highest-priority model selection
    mechanism (thread.ts#L71-L75), bypassing isModelValid() checks that can
    silently drop the config "model" field. This matches the OpenCode SDK's
    createOpencodeTui() pattern in packages/sdk/js/src/v2/server.ts.
    """
    return [f"--model=anaconda/{model_name}"]


# Agent registry — keyed by CLI command name
AGENTS: Dict[str, AgentDefinition] = {
    "claude": AgentDefinition(
        name="Claude Code",
        binary="claude",
        build_env=build_env_claude,
        build_agent_args=build_agent_args_claude,
        install_hint="Install with: npm install -g @anthropic-ai/claude-code",
    ),
    "opencode": AgentDefinition(
        name="OpenCode",
        binary="opencode",
        build_env=build_env_opencode,
        build_agent_args=build_agent_args_opencode,
        install_hint="Install from: https://opencode.ai",
    ),
}
