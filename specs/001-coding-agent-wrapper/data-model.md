# Data Model: Coding Agent Wrapper Commands

**Feature Branch**: `001-coding-agent-wrapper`

## Entities

### AgentDefinition

Represents a supported coding agent with its configuration requirements.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name (e.g., "Claude Code") |
| `binary` | string | Executable name to find on PATH (e.g., "claude") |
| `build_env` | callable | Function `(Server, model_name) -> dict[str, str]` that produces the environment variables to inject |
| `build_agent_args` | callable | Function `(model_name) -> list[str]` that produces agent-specific CLI args to prepend (e.g., OpenCode's `--model` flag). Returns `[]` for agents that need no extra args. |
| `install_hint` | string | Human-readable installation instructions shown when binary is not found |

**Instances** (initial):
- Claude Code: binary=`claude`, env via `ANTHROPIC_BASE_URL` (from `server.url`) + `ANTHROPIC_API_KEY`, injects `--model <model-name>` CLI args (space-separated)
- OpenCode: binary=`opencode`, env via `OPENCODE_CONFIG_CONTENT` (JSON string using `server.openai_url`), injects `--model=anaconda/<model-name>` CLI arg

### Server (existing)

Already defined in `clients/base.py`. Key attributes used by the wrapper:

| Field | Type | Description |
|-------|------|-------------|
| `openai_url` | string | OpenAI-compatible endpoint URL with `/v1` appended (e.g., `http://localhost:8080/v1`) — used for OpenCode |
| `url` | string | Raw base URL without path suffix (e.g., `http://localhost:8080`) — used for Claude Code's `ANTHROPIC_BASE_URL` |
| `api_key` | string | API key for server authentication |
| `id` | string | Server identifier for lookup |
| `is_running` | bool | Whether the server is currently running |
| `_matched` | bool | Whether this server was reused (not newly created) |
| `config.model_name` | string | Model identifier the server is running |

### WrapperSession (runtime, not persisted)

Tracks the state of a single wrapper invocation.

| Field | Type | Description |
|-------|------|-------------|
| `agent` | AgentDefinition | Which coding agent is being run |
| `server` | Server | The inference server being used |
| `server_owned` | bool | True if this wrapper invocation launched the server (False if pre-existing or `server/<id>`) |
| `detach` | bool | True if `--detach` flag was passed |
| `child_pid` | int | PID of the forked child process running the agent |
| `original_pgid` | int | Parent's process group ID (for terminal reclaim) |
| `saved_termios` | list | Terminal state saved before agent launch (for restoration) |

## State Transitions

### Server Lifecycle (from wrapper's perspective)

```
[Not Running] --create()--> [Starting] --start()--> [Running] --agent exits--> [Cleanup Decision]
                                                                                      |
                                                              server_owned AND NOT detach --> [Stopped]
                                                              server_owned AND detach -------> [Running]
                                                              NOT server_owned --------------> [Running]
```

### Wrapper Process Lifecycle

```
[Parse Args] --> [Resolve Server] --> [Build Env] --> [Build Agent Args] --> [Fork] --> [Parent: Wait] --> [Cleanup] --> [Exit]
                      |                                                        |
                      |                                                  [Child: setpgid + tcsetpgrp + execvpe]
                      |
                 model arg --> create/match server
                 server/<id> --> lookup server by ID
```
