# CLI Contract: Coding Agent Wrapper Commands

**Feature Branch**: `001-coding-agent-wrapper`

## Command: `anaconda ai claude`

```
anaconda ai claude <model-or-server> [--backend <backend>] [--at <site>] [--detach] [--json] [--<server-options>...] [-- <agent-args>...]
```

**Note**: Options can appear in any order. Use `--` to separate server options from agent arguments.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model-or-server` | positional | yes | Either a model identifier (e.g., `OpenHermes-2.5-Mistral-7B/Q4_K_M`) or `server/<server-id>` to connect to an existing server |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--backend` | string | config default | Select inference backend (ai-navigator, anaconda-desktop, ai-catalyst) |
| `--at` | string | config default | Site defined in config |
| `--detach` | flag | off (stop on exit) | Leave server running after agent exits. Opposite: `--rm` (default) |
| `--<key>=<value>` | extra args | none | Server configuration options (same as `anaconda ai launch`) |

### Separator

`--` separates wrapper/server options from coding agent arguments. Everything after `--` is forwarded verbatim to the `claude` binary.

### Exit Code

The wrapper exits with the same code as the `claude` process.

### Examples

```bash
# Launch server and run Claude Code
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M

# With server options and agent args
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --ctx-size=4096 --jinja -- --verbose

# Connect to existing server
anaconda ai claude server/abc123 -- --no-confirm

# Leave server running after exit
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --detach

# Use specific backend
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --backend ai-navigator --at mysite
```

### Error Cases

| Condition | Exit Code | Message |
|-----------|-----------|---------|
| `claude` binary not found | 1 | "Claude Code is not installed. Install with: npm install -g @anthropic-ai/claude-code" |
| Server ID not found | 1 | "Server '<id>' not found or not running" |
| Model not available | 1 | (delegated to existing `servers.create()` error handling) |
| Backend not reachable | 1 | (delegated to existing backend error handling) |

---

## Command: `anaconda ai opencode`

Identical contract to `anaconda ai claude` except:

| Difference | Value |
|-----------|-------|
| Binary invoked | `opencode` |
| Not-found message | "OpenCode is not installed. Install from: https://opencode.ai" |
| Env vars injected | `OPENCODE_CONFIG_CONTENT` (JSON) instead of `ANTHROPIC_*` |

```bash
# Launch server and run OpenCode
anaconda ai opencode OpenHermes-2.5-Mistral-7B/Q4_K_M

# With server options
anaconda ai opencode OpenHermes-2.5-Mistral-7B/Q4_K_M --ctx-size=4096 -- --theme dark
```
