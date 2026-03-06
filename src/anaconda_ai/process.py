"""Process launcher for coding agent wrapper commands.

Implements the fork+exec+tcsetpgrp pattern to give the coding agent
full terminal control while retaining the parent's ability to do
post-exit cleanup (e.g., stopping inference servers).

On platforms without os.fork (Windows), falls back to subprocess.Popen.
"""

from __future__ import annotations

import os
import signal
import sys
from typing import Callable, Dict, List, Optional


def _safe_tcsetpgrp(fd: int, pgrp: int) -> None:
    """Set pgrp as the controller of the tty fd.

    Safe wrapper that handles:
    - fd not connected to a terminal (ENOTTY)
    - Current process not owning the terminal (avoids SIGTTOU)
    """
    import errno

    try:
        curr_pgrp = os.tcgetpgrp(fd)
    except OSError as e:
        if e.errno == errno.ENOTTY:
            return
        raise

    if curr_pgrp == os.getpgrp():
        os.tcsetpgrp(fd, pgrp)


def _child_exec(binary_path: str, agent_args: List[str], env: Dict[str, str]) -> None:
    """Child process: set up process group, terminal control, and exec the agent.

    This function never returns — it replaces the process with the agent.
    """
    # Step 1: Create new process group (child is leader)
    os.setpgid(0, 0)

    # Step 2: Become foreground process group (may get SIGTTOU, ignore it)
    old_handler = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        try:
            tty_fd = os.open("/dev/tty", os.O_RDWR)
            os.tcsetpgrp(tty_fd, os.getpgrp())
            os.close(tty_fd)
        except OSError:
            pass  # no terminal
    finally:
        signal.signal(signal.SIGTTOU, old_handler)

    # Step 3: Reset signal handlers to default (undo any SIG_IGN from parent)
    for sig in (
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTSTP,
        signal.SIGTTIN,
        signal.SIGTTOU,
    ):
        signal.signal(sig, signal.SIG_DFL)

    # Step 4: Build full environment and exec
    full_env = {**os.environ, **env}
    argv = [binary_path] + agent_args
    os.execvpe(binary_path, argv, full_env)


def _parent_wait_and_cleanup(
    child_pid: int,
    cleanup_fn: Optional[Callable[[], None]],
) -> int:
    """Parent process: wait for child, reclaim terminal, run cleanup.

    Returns the child's exit code.
    """
    # Close the race window: set child's pgid from parent side too
    try:
        os.setpgid(child_pid, child_pid)
    except (ProcessLookupError, PermissionError):
        pass  # child already exec'd (EACCES) or exited (ESRCH)

    # Give terminal to child's process group
    try:
        _safe_tcsetpgrp(sys.stdin.fileno(), child_pid)
    except (OSError, ValueError):
        pass  # no terminal

    # Ignore SIGINT, SIGTSTP, SIGTTIN, SIGTTOU while child is running.
    # The parent is a background process — it must not respond to terminal
    # job-control signals (SIGTSTP/SIGTTIN/SIGTTOU) or keyboard interrupt
    # (SIGINT). The child's process group is the exclusive foreground; these
    # signals are meant for it, not the parent. This matches POSIX shell
    # behavior (glibc manual §27.6.4).
    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    old_sigtstp = signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    old_sigttin = signal.signal(signal.SIGTTIN, signal.SIG_IGN)
    old_sigttou = signal.signal(signal.SIGTTOU, signal.SIG_IGN)

    _, status = os.waitpid(child_pid, 0)

    try:
        _safe_tcsetpgrp(sys.stdin.fileno(), os.getpgrp())
    except (OSError, ValueError):
        pass

    try:
        import termios

        saved = termios.tcgetattr(sys.stdin.fileno())
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
    except Exception:
        pass

    if cleanup_fn is not None:
        try:
            cleanup_fn()
        except Exception:
            pass  # best-effort cleanup

    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTSTP, old_sigtstp)
    signal.signal(signal.SIGTTIN, old_sigttin)
    signal.signal(signal.SIGTTOU, old_sigttou)

    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _run_with_subprocess(
    binary_path: str,
    agent_args: List[str],
    env: Dict[str, str],
    cleanup_fn: Optional[Callable[[], None]],
) -> int:
    """Windows/fallback: run agent via subprocess.Popen without process group isolation."""
    import subprocess

    full_env = {**os.environ, **env}
    argv = [binary_path] + agent_args
    try:
        proc = subprocess.Popen(argv, env=full_env)
        exit_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        exit_code = proc.wait()
    finally:
        if cleanup_fn is not None:
            try:
                cleanup_fn()
            except Exception:
                pass
    return exit_code


def run_agent_foreground(
    binary_path: str,
    agent_args: List[str],
    env: Dict[str, str],
    cleanup_fn: Optional[Callable[[], None]] = None,
) -> int:
    """Launch a coding agent with full terminal control and optional post-exit cleanup.

    Uses os.fork() + os.execvpe() with POSIX job control on Unix.
    Falls back to subprocess.Popen on Windows (no os.fork).

    The parent's waitpid() does NOT catch or suppress errors from the agent
    process — whatever exit code the agent returns is propagated.

    Args:
        binary_path: Full path to the agent binary.
        agent_args: Arguments to pass to the agent (from after '--').
        env: Environment variables to inject (merged with os.environ).
        cleanup_fn: Optional function to call after the agent exits
                     (e.g., stop inference server). Not called in detach mode.

    Returns:
        The agent's exit code.
    """
    if not hasattr(os, "fork"):
        return _run_with_subprocess(binary_path, agent_args, env, cleanup_fn)

    pid = os.fork()
    if pid == 0:
        # Child process — exec the agent (never returns)
        try:
            _child_exec(binary_path, agent_args, env)
        except Exception:
            os._exit(127)
    else:
        # Parent process — wait and cleanup
        return _parent_wait_and_cleanup(pid, cleanup_fn)
