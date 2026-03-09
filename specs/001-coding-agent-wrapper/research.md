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
1. `tcsetpgrp(tty_fd, parent_pgid)` — reclaim terminal (SIGTTOU already masked)
2. Restore terminal modes with `termios.tcsetattr()` (agent may have left raw mode)
3. Run cleanup (stop server)
4. Restore all signal handlers to original values
5. `sys.exit(child_exit_code)` — propagate exit code

**Critical**: The parent must ignore `SIGINT`, `SIGTSTP`, `SIGTTIN`, and `SIGTTOU` for the entire duration from before `waitpid()` through cleanup completion. Only ignoring `SIGINT` is insufficient — when the child exits and the terminal's foreground process group is in transition, the parent (a background process) can receive `SIGTSTP` and get suspended by the shell, preventing cleanup from ever running. This matches the glibc manual §27.6.4 shell implementation where the parent shell masks all job-control signals while a foreground job is active.

### Signal Handling in the Parent

The parent must mask all four job-control and interrupt signals before entering `waitpid()` and keep them masked through cleanup:

| Signal | Parent action | Why |
|--------|--------------|-----|
| SIGINT | SIG_IGN | Terminal sends to foreground pgrp (child), but parent may receive stale delivery |
| SIGTSTP | SIG_IGN | During child exit/pgrp transition, shell can suspend the parent — prevents cleanup |
| SIGTTIN | SIG_IGN | Background process reading from terminal would stop the parent |
| SIGTTOU | SIG_IGN | Background process writing to terminal (or calling tcsetpgrp) would stop the parent |

Signals are restored to their original handlers **after** cleanup completes, just before returning the exit code.

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
**Wire format**: Anthropic API (`/v1/messages`)
**Key env vars**:

| Env Var | Value from Server | Purpose |
|---------|-------------------|---------|
| `ANTHROPIC_BASE_URL` | `server.url` (the raw base URL, e.g., `http://localhost:8080`) | API endpoint — Claude Code appends `/v1/messages` itself |
| `ANTHROPIC_API_KEY` | `server.api_key` or dummy value | Auth (local servers typically don't validate) |

All Anaconda AI backends expose the `/v1/messages` Anthropic-compatible route at their base URL, so `server.url` is the correct value. Note: do NOT use `server.openai_url` (which appends `/v1` for OpenAI format) — that would result in a double-pathed `http://localhost:8080/v1/v1/messages`.

**Model selection via `--model` CLI flag**: Claude Code accepts `--model <name>` (space-separated) to set the model for the session ([CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-reference)). The wrapper injects `--model <model-name>` so Claude Code's UI displays the correct model name. The local server ignores the `model` field in API requests (it serves whatever model is loaded), but passing it ensures correct display and future-proofs against servers that may validate it.

### OpenCode

**Binary**: `opencode`
**Wire format**: Configurable via provider registry (models.dev); supports OpenAI-compatible via `@ai-sdk/openai-compatible`

OpenCode requires two separate configuration mechanisms — one for the provider/endpoint and one for model selection:

#### 1. Provider Configuration via `OPENCODE_CONFIG_CONTENT`

OpenCode does NOT read a base URL from a single env var. Provider/endpoint configuration must be injected via:

| Env Var | Value | Purpose |
|---------|-------|---------|
| `OPENCODE_CONFIG_CONTENT` | JSON string with provider config | Inline config injection (highest config precedence, per `config.ts#L78-L179`) |

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

#### 2. Model Selection via `--model` CLI Flag

OpenCode's TUI accepts `--model` / `-m` as a CLI flag (`thread.ts#L71-L75`):

```
opencode --model=anaconda/<model-name>
```

**Model resolution priority** (from `local.tsx#L153-L173`):
1. `--model` CLI flag — **highest priority, unconditional**
2. `OPENCODE_CONFIG_CONTENT`'s `"model"` field — second priority, subject to `isModelValid()` check
3. Per-session recently used model
4. First model from first available provider

The config `"model"` field alone is fragile — if `isModelValid()` fails (e.g., provider not yet synced), it silently falls through to another model. The `--model` CLI flag bypasses this check on the `run` subcommand path entirely (`run.ts#L639-L646`).

**The wrapper MUST pass `--model=anaconda/<model-name>` as a CLI argument** in addition to `OPENCODE_CONFIG_CONTENT`. This matches OpenCode's own SDK behavior — `createOpencodeTui()` in `packages/sdk/js/src/v2/server.ts` sets both `OPENCODE_CONFIG_CONTENT` and `--model=` when a model is specified.

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

Use the `--` separator to delineate server options from agent arguments. A custom `AgentCommand(TyperCommand)` subclass overrides `parse_args()` to intercept `--` **before** click's parser consumes it. Agent arguments (after `--`) are stashed in `ctx.meta['agent_args']` and stripped from the arg list. Click then parses the remaining args normally — typed options (`--detach`, `--backend`, etc.) and extra server options (`--ctx-size=4096`) all work in any position.

### Rationale

Click/typer's parser silently consumes `--` when processing args. The `--` token is popped from `rargs` at `click/parser.py#L327-330` and never reaches `ctx.args`. This means any design that relies on finding `--` in `ctx.args` cannot work.

Two alternatives were explored and rejected:
- **`allow_interspersed_args=False`** preserves `--` in `ctx.args`, but breaks typed options after the positional argument — `anaconda ai claude MyModel --detach` would silently fail to parse `--detach`.
- **`sys.argv` post-processing** doesn't work with `CliRunner` in tests.

The `AgentCommand.parse_args()` approach is clean: it splits args on `--` at the earliest possible point (before click's parser runs), so click never sees `--` or the agent args. All typed options, extra server options, and agent arguments work as expected with no UX constraints on argument ordering.

### Alternatives Rejected

| Approach | Why Rejected |
|----------|--------------|
| Default click behavior | `--` silently consumed, never in `ctx.args` — cannot split server opts from agent args |
| `allow_interspersed_args=False` | `--` preserved, but typed options after positional stop working (e.g., `MyModel --detach` broken) |
| `click.UNPROCESSED` with `nargs=-1` | Grabs extra server options too when combined with `allow_extra_args` |
| Post-parse `sys.argv` | Fragile, doesn't work with test runners (`CliRunner`) |
| Drop `--` entirely, use `--agent-args="..."` | Ugly UX for agent arguments, non-standard |

### Evidence

- **click source** (`parser.py#L327-330`): `if arg == "--": return` — `--` is consumed and discarded
- **Google ADK** (`cli_tools_click.py`): Uses `allow_interspersed_args=False` but has no typed options after positional — not our case
- **Direct testing**: Confirmed `allow_interspersed_args=False` breaks `anaconda ai claude MyModel --backend x` (typed opt after positional)
- **`AgentCommand` approach**: Tested with all combinations — typed opts before/after positional, server opts, `--`, agent args all work correctly

### Implementation

```python
class AgentCommand(TyperCommand):
    def parse_args(self, ctx, args):
        if "--" in args:
            idx = args.index("--")
            ctx.meta["agent_args"] = list(args[idx + 1:])
            args = list(args[:idx])
        else:
            ctx.meta["agent_args"] = []
        return super().parse_args(ctx, args)
```

In `_run_wrapper()`:
```python
agent_args = ctx.meta.get("agent_args", [])  # Already extracted by AgentCommand
```

### Parsing Flow

```
anaconda ai claude MyModel/Q4_K_M --backend anaconda-desktop --ctx-size=80000 -- --tools=Edit,Read
│                  │               │                          │                │  │
│                  │               │                          │                │  └─ agent args (ctx.meta['agent_args'])
│                  │               │                          │                └─ separator (intercepted by AgentCommand)
│                  │               │                          └─ server extra option (ctx.args)
│                  │               └─ typed option (parsed by typer — works anywhere)
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
