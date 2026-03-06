"""Tests for coding agent wrapper agent definitions and utilities."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from anaconda_ai.agents import (
    AGENTS,
    AgentDefinition,
    build_agent_args_claude,
    build_agent_args_opencode,
    build_env_claude,
    build_env_opencode,
    find_agent_binary,
)


class TestBuildEnvClaude:
    """Tests for build_env_claude function."""

    def test_build_env_claude(self) -> None:
        """Test that build_env_claude returns correct Anthropic environment variables."""
        # Create a mock Server with url and api_key
        server = MagicMock()
        server.url = "http://localhost:8080"
        server.api_key = "test-key"

        # Call the function
        result = build_env_claude(server, "test-model")

        # Assert the result contains the expected environment variables
        assert result == {
            "ANTHROPIC_BASE_URL": "http://localhost:8080",
            "ANTHROPIC_API_KEY": "test-key",
        }


class TestBuildEnvOpencode:
    """Tests for build_env_opencode function."""

    def test_build_env_opencode(self) -> None:
        """Test that build_env_opencode returns correct OpenCode configuration."""
        server = MagicMock()
        server.openai_url = "http://localhost:8080/v1"
        server.api_key = "test-key"

        result = build_env_opencode(server, "test-model")

        assert "OPENCODE_CONFIG_CONTENT" in result

        config = json.loads(result["OPENCODE_CONFIG_CONTENT"])

        assert "provider" in config
        assert "anaconda" in config["provider"]
        provider = config["provider"]["anaconda"]
        assert provider["options"]["baseURL"] == "http://localhost:8080/v1"
        assert provider["options"]["apiKey"] == "test-key"
        assert "test-model" in provider["models"]
        assert config["model"] == "anaconda/test-model"


class TestFindAgentBinary:
    """Tests for find_agent_binary function."""

    def test_find_agent_binary_found(self) -> None:
        """Test that find_agent_binary returns the path when binary is found."""
        with patch("anaconda_ai.agents.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/claude"

            result = find_agent_binary("claude")

            assert result == "/usr/bin/claude"
            mock_which.assert_called_once_with("claude")

    def test_find_agent_binary_not_found(self) -> None:
        """Test that find_agent_binary returns None when binary is not found."""
        with patch("anaconda_ai.agents.shutil.which") as mock_which:
            mock_which.return_value = None

            result = find_agent_binary("claude")

            assert result is None
            mock_which.assert_called_once_with("claude")


class TestAgentsRegistry:
    """Tests for AGENTS registry (US4)."""

    def test_agents_registry_has_claude_and_opencode(self) -> None:
        """Test that AGENTS dict contains claude and opencode with correct fields."""
        assert "claude" in AGENTS
        assert "opencode" in AGENTS

        claude = AGENTS["claude"]
        assert isinstance(claude, AgentDefinition)
        assert claude.name == "Claude Code"
        assert claude.binary == "claude"
        assert claude.build_env is build_env_claude

        opencode = AGENTS["opencode"]
        assert isinstance(opencode, AgentDefinition)
        assert opencode.name == "OpenCode"
        assert opencode.binary == "opencode"
        assert opencode.build_env is build_env_opencode

    def test_agent_install_hints(self) -> None:
        """Test that each agent's install_hint is non-empty and actionable."""
        for name, agent in AGENTS.items():
            assert agent.install_hint, f"Agent '{name}' has empty install_hint"
            assert len(agent.install_hint) > 10, (
                f"Agent '{name}' install_hint too short: {agent.install_hint}"
            )


class TestBuildAgentArgs:
    """Tests for per-agent CLI argument builders (Phase 11)."""

    def test_build_agent_args_opencode(self) -> None:
        result = build_agent_args_opencode("MyModel")
        assert result == ["--model=anaconda/MyModel"]

    def test_build_agent_args_opencode_with_quant(self) -> None:
        result = build_agent_args_opencode("OpenHermes-2.5-Mistral-7B/Q4_K_M")
        assert result == ["--model=anaconda/OpenHermes-2.5-Mistral-7B/Q4_K_M"]

    def test_build_agent_args_claude(self) -> None:
        result = build_agent_args_claude("MyModel")
        assert result == ["--model", "MyModel"]
