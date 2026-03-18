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

# What actually gets executed (internally):
# claude --model OpenHermes-2.5-Mistral-7B/Q4_K_M --verbose
# with ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY set in the environment
```

### Auto-Injected CLI Args

The wrapper automatically prepends `--model <model-name>` to the arguments passed to `claude`. While the local inference server ignores the model field in API requests, passing `--model` ensures Claude Code's UI displays the correct model name and future-proofs against servers that may validate it. The user's arguments from after `--` are appended after the injected `--model` flag.

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
| Auto-injected CLI args | `--model=anaconda/<model-name>` (equals-separated, with `anaconda/` provider prefix) instead of `--model <model-name>` (space-separated, bare model name) |

The wrapper automatically prepends `--model=anaconda/<model-name>` to the arguments passed to `opencode`. This ensures the model is selected at the highest priority level (CLI flag), matching the pattern used by OpenCode's own SDK (`createOpencodeTui`). The user's arguments from after `--` are appended after the injected `--model` flag.

```bash
# Launch server and run OpenCode
anaconda ai opencode OpenHermes-2.5-Mistral-7B/Q4_K_M

# With server options
anaconda ai opencode OpenHermes-2.5-Mistral-7B/Q4_K_M --ctx-size=4096 -- --theme dark

# What actually gets executed (internally):
# opencode --model=anaconda/OpenHermes-2.5-Mistral-7B/Q4_K_M --theme dark
# with OPENCODE_CONFIG_CONTENT set in the environment
```
