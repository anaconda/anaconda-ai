# Research: Coding Agent Wrapper Commands

**Feature Branch**: `001-coding-agent-wrapper`
**Date**: 2026-03-06

## Decision 1: Process Execution Strategy

### Decision

Use `os.fork()` + `os.execvpe()` in the child with `os.setpgid()` + `os.tcsetpgrp()` for foreground process group transfer. The parent waits via `os.waitpid()` and performs server cleanup after the child exits. For `--detach` mode, use `os.execvpe()` directly (no fork needed since no cleanup required).

### Rationale

The wrapper must satisfy two competing requirements:
1. **Full terminal interactivity** — coding agents are TUI apps that need raw mode, alternate screen buffer, Ctrl+C/Z handling, and terminal resize events.
2. **Post-exit server cleanup** — the wrapper must stop inference servers it launched when the agent exits.

`os.execvpe()` alone gives perfect TTY but no cleanup (process is replaced). `subprocess.Popen()` alone allows cleanup but the child shares the parent's process group — both receive SIGINT, and the child isn't the exclusive foreground process. The fork+exec+tcsetpgrp pattern solves both:

- **Child** calls `os.setpgid(0, 0)` (new process group), then `os.tcsetpgrp()` (become foreground), then `os.execvpe()` (become the agent). The agent IS the process — perfect TTY control.
- **Parent** is in the background process group. `os.waitpid()` is a pure kernel call that works from background. After it returns, the parent reclaims the terminal with `tcsetpgrp()` and runs cleanup.

This is the exact pattern used by:
- Every POSIX shell (bash, zsh, fish) for foreground job execution
- Microsoft's debugpy launcher (`debuggee.py`) for interactive debugging
- Chromium OS's snakeoil library for namespace-isolated process execution
- The glibc manual's canonical shell implementation (§27.6.4 "Launching Jobs")

### Alternatives Considered

| Approach | TTY Control | Cleanup | Why Rejected |
|----------|------------|---------|--------------|
| `os.execvpe()` only | ✅ Perfect | ❌ None | Cannot stop inference server after agent exits |
| `subprocess.Popen()` only | ⚠️ Shared pgrp | ✅ | Parent and child both receive SIGINT; child not exclusive foreground |
| `subprocess.Popen(process_group=0)` + `tcsetpgrp()` | ✅ Full | ✅ | Requires Python 3.11+; project targets 3.10+ |
| `subprocess.Popen(preexec_fn=...)` | ✅ Full | ✅ | `preexec_fn` has deadlock warnings in CPython docs; fork+exec is cleaner |

### Implementation Protocol

The three non-negotiable steps (from the glibc shell implementation manual):

1. **`setpgid(0, 0)`** in child AND `setpgid(child_pid, child_pid)` in parent — creates isolated process group, closes race window
2. **`tcsetpgrp(tty_fd, child_pgid)`** — makes child's group the exclusive foreground
3. **Reset signal handlers to `SIG_DFL`** in child before exec — undoes any `SIG_IGN` inherited from parent

After `waitpid()` returns in the parent:
1. Ignore `SIGTTOU` (parent is background, tcsetpgrp would trigger it)
2. `tcsetpgrp(tty_fd, parent_pgid)` — reclaim terminal
3. Restore terminal modes with `termios.tcsetattr()` (agent may have left raw mode)
4. Run cleanup (stop server)
5. `sys.exit(child_exit_code)` — propagate exit code

### Signal Routing Summary

| Signal | With fork+tcsetpgrp | Notes |
|--------|---------------------|-------|
| SIGINT (Ctrl+C) | → child only | Terminal sends to foreground pgrp |
| SIGTSTP (Ctrl+Z) | → child only | Terminal sends to foreground pgrp |
| SIGWINCH (resize) | → child only | Terminal sends to foreground pgrp |
| SIGCHLD | → parent | Kernel sends when child exits/stops |
| SIGTERM | → whichever process it targets | Parent should forward to child if received |
| waitpid() | Works from background | Pure kernel call, no terminal involvement |

### Platform Notes

- **macOS/Linux**: Full fork+exec+tcsetpgrp support. This is the primary path.
- **Windows**: `os.fork()` is not available. Fall back to `subprocess.Popen()` without process group isolation. Windows doesn't have POSIX job control, so the shared-pgrp issue is moot. Interactive agents on Windows use ConPTY which handles stdio inheritance differently.

---

## Decision 2: Agent-Specific Environment Variable Configuration

### Decision

Each coding agent requires different environment variables and potentially different wire formats. The wrapper must maintain a per-agent configuration mapping.

### Claude Code

**Binary**: `claude`
**Wire format**: Anthropic API (`/v1/messages`) — NOT OpenAI-compatible
**Key env vars**:

| Env Var | Value from Server | Purpose |
|---------|-------------------|---------|
| `ANTHROPIC_BASE_URL` | `server.openai_url` (without `/v1` suffix — adjusted to match Anthropic format) | API endpoint |
| `ANTHROPIC_API_KEY` | `server.api_key` or dummy value | Auth (local servers typically don't validate) |

**Wire format note**: Claude Code sends requests in Anthropic's Messages API format (`POST /v1/messages`), not OpenAI's chat completions format. Services like OpenRouter work because they accept the Anthropic wire format on their end. For local inference servers that only speak OpenAI format (llama.cpp, Ollama), a translation proxy (LiteLLM, Claude Code Router) would be needed between the server and Claude Code. This is outside the wrapper's scope — the wrapper only sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`; ensuring the endpoint speaks the right protocol is the user's or backend's responsibility.

### OpenCode

**Binary**: `opencode`
**Wire format**: Configurable via provider registry (models.dev); supports OpenAI-compatible via `@ai-sdk/openai-compatible`
**Key env vars**:

OpenCode does NOT read a base URL from a single env var. Configuration must be injected via:

| Env Var | Value | Purpose |
|---------|-------|---------|
| `OPENCODE_CONFIG_CONTENT` | JSON string with provider config | Inline config injection (highest precedence) |

The JSON config must define a custom provider pointing at the server's OpenAI-compatible endpoint:

```json
{
  "provider": {
    "anaconda": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "<server.openai_url>",
        "apiKey": "<server.api_key>"
      },
      "models": {
        "<model-name>": { "name": "<model-name>" }
      }
    }
  },
  "model": "anaconda/<model-name>"
}
```

### Extensibility

Other agents (for future consideration):

| Agent | Binary | Base URL Env Var | API Key Env Var | Wire Format |
|-------|--------|-----------------|-----------------|-------------|
| Aider | `aider` | `OPENAI_API_BASE` | `OPENAI_API_KEY` | OpenAI |
| Codex CLI | `codex` | `OPENAI_BASE_URL` | `OPENAI_API_KEY` | OpenAI |

### Rationale

Each agent has evolved independently and uses different configuration mechanisms. There is no universal standard. The wrapper must abstract this — each agent definition includes a function that takes a `Server` object and returns the environment dict to inject.

### Alternatives Considered

- **Single universal env var approach**: Rejected because Claude Code uses Anthropic format and OpenCode requires JSON config, not env vars.
- **Config file generation only**: Rejected because Claude Code works purely via env vars and doesn't need config files.

---

## Decision 3: Command Argument Parsing Strategy

### Decision

Use the `--` separator to delineate server options from agent arguments. Everything before `--` is parsed by the wrapper (model, `--backend`, `--at`, `--detach`, and extra `--key=value` server options). Everything after `--` is forwarded verbatim to the coding agent.

### Rationale

This follows the Unix convention (`--` ends option processing) and matches the user's original design. The existing `launch` command already uses `context_settings={"allow_extra_args": True, "ignore_unknown_options": True}` for extra server options — we extend this pattern.

### Parsing Flow

```
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --ctx-size=512 --detach -- --verbose --no-confirm
│                  │                                   │               │         │  │
│                  │                                   │               │         │  └─ agent args (forwarded verbatim)
│                  │                                   │               │         └─ separator
│                  │                                   │               └─ wrapper flag
│                  │                                   └─ server extra option
│                  └─ positional: model OR server/<id>
└─ subcommand
```

---

## Decision 4: Server Lifecycle Management

### Decision

The wrapper tracks whether it launched the server or connected to an existing one. Only servers launched by the wrapper are stopped on cleanup (unless `--detach`).

### Rationale

The existing `servers.create()` method already returns a `Server` object with a `_matched` attribute indicating whether it reused an existing server. The `launch` command already uses this pattern (line 345 of cli.py: `if server._matched: return`). We follow the same convention.

### Lifecycle Matrix

| Mode | Server Source | On Agent Exit | On `--detach` |
|------|-------------|---------------|---------------|
| `anaconda ai claude <model>` | New (launched by wrapper) | Stop + delete | Leave running |
| `anaconda ai claude <model>` | Matched (already running) | Leave running | Leave running |
| `anaconda ai claude server/<id>` | Existing (looked up by ID) | Leave running | Leave running |
