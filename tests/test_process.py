"""Tests for coding agent wrapper process launcher."""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch


from anaconda_ai.process import run_agent_foreground


class TestRunAgentForegroundExitCode:
    """Tests for run_agent_foreground exit code handling."""

    def test_run_agent_foreground_exit_code(self, monkeypatch) -> None:
        """Test that run_agent_foreground returns the child's exit code."""
        # Mock os.fork to return positive pid (parent path)
        mock_fork = MagicMock(return_value=12345)
        monkeypatch.setattr(os, "fork", mock_fork)

        # Mock os.waitpid to return (pid, exit_status_for_42)
        # Exit code 42 is encoded as 42 << 8 in the status
        exit_status = 42 << 8
        mock_waitpid = MagicMock(return_value=(12345, exit_status))
        monkeypatch.setattr(os, "waitpid", mock_waitpid)

        # Mock os.setpgid
        mock_setpgid = MagicMock()
        monkeypatch.setattr(os, "setpgid", mock_setpgid)

        # Mock signal.signal
        mock_signal = MagicMock(return_value=signal.SIG_DFL)
        monkeypatch.setattr(signal, "signal", mock_signal)

        # Mock _safe_tcsetpgrp
        with patch("anaconda_ai.process._safe_tcsetpgrp"):
            result = run_agent_foreground("/usr/bin/claude", ["--help"], {})

        # Assert the exit code is 42
        assert result == 42


class TestRunAgentForegroundEnvPassed:
    """Tests for run_agent_foreground environment variable passing."""

    def test_run_agent_foreground_env_passed(self, monkeypatch) -> None:
        """Test that run_agent_foreground passes environment variables to child."""
        # Mock os.fork to return 0 (child path)
        mock_fork = MagicMock(return_value=0)
        monkeypatch.setattr(os, "fork", mock_fork)

        # Mock os.setpgid
        mock_setpgid = MagicMock()
        monkeypatch.setattr(os, "setpgid", mock_setpgid)

        # Mock signal.signal
        mock_signal = MagicMock(return_value=signal.SIG_DFL)
        monkeypatch.setattr(signal, "signal", mock_signal)

        # Mock os.execvpe to capture args and raise SystemExit
        mock_execvpe = MagicMock(side_effect=SystemExit(0))
        monkeypatch.setattr(os, "execvpe", mock_execvpe)

        # Mock os.open and os.close for /dev/tty
        mock_open = MagicMock(return_value=3)
        mock_close = MagicMock()
        monkeypatch.setattr(os, "open", mock_open)
        monkeypatch.setattr(os, "close", mock_close)

        # Mock os.tcsetpgrp
        mock_tcsetpgrp = MagicMock()
        monkeypatch.setattr(os, "tcsetpgrp", mock_tcsetpgrp)

        # Mock os.getpgrp
        mock_getpgrp = MagicMock(return_value=1000)
        monkeypatch.setattr(os, "getpgrp", mock_getpgrp)

        # Call run_agent_foreground with test env vars
        test_env = {"TEST_VAR": "test_value"}
        try:
            run_agent_foreground("/usr/bin/claude", ["--help"], test_env)
        except SystemExit:
            pass

        # Verify os.execvpe was called with the correct environment
        assert mock_execvpe.called
        call_args = mock_execvpe.call_args
        # call_args[0] is (binary_path, argv, env)
        env_passed = call_args[0][2]
        assert "TEST_VAR" in env_passed
        assert env_passed["TEST_VAR"] == "test_value"


class TestRunAgentForegroundArgsForwarded:
    """Tests for run_agent_foreground argument forwarding."""

    def test_run_agent_foreground_args_forwarded(self, monkeypatch) -> None:
        """Test that run_agent_foreground forwards arguments to the agent."""
        # Mock os.fork to return 0 (child path)
        mock_fork = MagicMock(return_value=0)
        monkeypatch.setattr(os, "fork", mock_fork)

        # Mock os.setpgid
        mock_setpgid = MagicMock()
        monkeypatch.setattr(os, "setpgid", mock_setpgid)

        # Mock signal.signal
        mock_signal = MagicMock(return_value=signal.SIG_DFL)
        monkeypatch.setattr(signal, "signal", mock_signal)

        # Mock os.execvpe to capture args and raise SystemExit
        mock_execvpe = MagicMock(side_effect=SystemExit(0))
        monkeypatch.setattr(os, "execvpe", mock_execvpe)

        # Mock os.open and os.close for /dev/tty
        mock_open = MagicMock(return_value=3)
        mock_close = MagicMock()
        monkeypatch.setattr(os, "open", mock_open)
        monkeypatch.setattr(os, "close", mock_close)

        # Mock os.tcsetpgrp
        mock_tcsetpgrp = MagicMock()
        monkeypatch.setattr(os, "tcsetpgrp", mock_tcsetpgrp)

        # Mock os.getpgrp
        mock_getpgrp = MagicMock(return_value=1000)
        monkeypatch.setattr(os, "getpgrp", mock_getpgrp)

        # Call run_agent_foreground with test args
        test_args = ["--help", "--verbose"]
        try:
            run_agent_foreground("/usr/bin/claude", test_args, {})
        except SystemExit:
            pass

        # Verify os.execvpe was called with the correct argv
        assert mock_execvpe.called
        call_args = mock_execvpe.call_args
        # call_args[0] is (binary_path, argv, env)
        argv_passed = call_args[0][1]
        assert argv_passed == ["/usr/bin/claude", "--help", "--verbose"]


class TestWindowsFallback:
    """Tests for Windows fallback using subprocess.Popen."""

    def test_windows_fallback(self, monkeypatch) -> None:
        """Test that run_agent_foreground falls back to subprocess.Popen on Windows."""
        # Delete the fork attribute from os module to simulate Windows
        monkeypatch.delattr(os, "fork", raising=False)

        # Mock subprocess.Popen
        mock_popen = MagicMock()
        mock_popen.wait.return_value = 0
        mock_popen_class = MagicMock(return_value=mock_popen)

        with patch("subprocess.Popen", mock_popen_class):
            result = run_agent_foreground("/usr/bin/claude", ["--help"], {})

        # Verify subprocess.Popen was called
        assert mock_popen_class.called
        # Verify the exit code is returned
        assert result == 0


class TestCleanupCalledForOwnedServer:
    """Tests for cleanup behavior after agent exits (US3)."""

    def test_cleanup_called_for_owned_server(self, monkeypatch) -> None:
        """Test that cleanup_fn is called after child exits for owned servers."""
        # Mock os.fork to return positive pid (parent path)
        mock_fork = MagicMock(return_value=12345)
        monkeypatch.setattr(os, "fork", mock_fork)

        # Mock os.waitpid — child exited normally with code 0
        exit_status = 0 << 8
        mock_waitpid = MagicMock(return_value=(12345, exit_status))
        monkeypatch.setattr(os, "waitpid", mock_waitpid)

        # Mock os.setpgid
        monkeypatch.setattr(os, "setpgid", MagicMock())

        # Mock signal.signal
        monkeypatch.setattr(signal, "signal", MagicMock(return_value=signal.SIG_DFL))

        # Create a cleanup function that tracks calls
        cleanup_fn = MagicMock()

        with patch("anaconda_ai.process._safe_tcsetpgrp"):
            result = run_agent_foreground(
                "/usr/bin/claude", ["--help"], {}, cleanup_fn=cleanup_fn
            )

        assert result == 0
        cleanup_fn.assert_called_once()

    def test_cleanup_not_called_for_detach(self, monkeypatch) -> None:
        """Test that no cleanup happens when cleanup_fn is None (detach mode)."""
        mock_fork = MagicMock(return_value=12345)
        monkeypatch.setattr(os, "fork", mock_fork)

        exit_status = 0 << 8
        mock_waitpid = MagicMock(return_value=(12345, exit_status))
        monkeypatch.setattr(os, "waitpid", mock_waitpid)

        monkeypatch.setattr(os, "setpgid", MagicMock())
        monkeypatch.setattr(signal, "signal", MagicMock(return_value=signal.SIG_DFL))

        with patch("anaconda_ai.process._safe_tcsetpgrp"):
            # Pass cleanup_fn=None to simulate detach mode
            result = run_agent_foreground(
                "/usr/bin/claude", ["--help"], {}, cleanup_fn=None
            )

        assert result == 0
        # No way to assert "not called" on None, but the point is no exception

    def test_cleanup_not_called_for_preexisting_server(self, monkeypatch) -> None:
        """Test that cleanup_fn=None is passed for pre-existing servers (no cleanup)."""
        mock_fork = MagicMock(return_value=12345)
        monkeypatch.setattr(os, "fork", mock_fork)

        exit_status = 0 << 8
        mock_waitpid = MagicMock(return_value=(12345, exit_status))
        monkeypatch.setattr(os, "waitpid", mock_waitpid)

        monkeypatch.setattr(os, "setpgid", MagicMock())
        monkeypatch.setattr(signal, "signal", MagicMock(return_value=signal.SIG_DFL))

        # Pre-existing server scenario: cleanup_fn is None
        with patch("anaconda_ai.process._safe_tcsetpgrp"):
            result = run_agent_foreground("/usr/bin/claude", [], {}, cleanup_fn=None)

        assert result == 0
