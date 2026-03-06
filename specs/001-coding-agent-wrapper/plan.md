# Implementation Plan: Coding Agent Wrapper Commands

**Branch**: `001-coding-agent-wrapper` | **Date**: 2026-03-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-coding-agent-wrapper/spec.md`

## Summary

Add `anaconda ai claude` and `anaconda ai opencode` wrapper commands that launch (or connect to) an Anaconda AI inference server and then exec a coding agent with the server's endpoint configured via environment variables. Uses `os.fork()` + `os.execvpe()` with POSIX job control (`setpgid` + `tcsetpgrp`) to give the agent full terminal control while retaining the parent's ability to clean up server resources after the agent exits.

## Technical Context

**Language/Version**: Python 3.10+ (project minimum from `pyproject.toml`)
**Primary Dependencies**: typer, rich, anaconda-cli-base, anaconda-auth (all existing); no new dependencies
**Storage**: N/A — no persistence beyond existing config
**Testing**: pytest with typer.testing.CliRunner; monkeypatch/mock for fork/exec/subprocess
**Target Platform**: macOS, Linux (primary); Windows (fallback via subprocess.Popen without process group isolation)
**Project Type**: CLI plugin (extends existing `anaconda ai` CLI)
**Performance Goals**: N/A — wrapper overhead is negligible compared to server startup time
**Constraints**: Must work with Python 3.10 (no `subprocess.Popen(process_group=)` which is 3.11+)
**Scale/Scope**: 2 new CLI commands, ~3 new source files, ~3 new test files

## Constitution Check

*No constitution file exists. Gate passes by default.*

## Project Structure

### Documentation (this feature)

```text
specs/001-coding-agent-wrapper/
├── plan.md              # This file
├── research.md          # Phase 0: architectural decisions
├── data-model.md        # Phase 1: entity definitions
├── quickstart.md        # Phase 1: usage guide
├── contracts/
│   └── cli.md           # Phase 1: CLI command contracts
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/anaconda_ai/
├── cli.py               # MODIFIED: add @app.command("claude") and @app.command("opencode")
├── agents.py            # NEW: AgentDefinition registry + per-agent env var builders
├── process.py           # NEW: fork+exec+tcsetpgrp process launcher + cleanup logic
├── clients/
│   └── base.py          # EXISTING: Server.openai_url, Server.api_key (read-only usage)
└── ...                  # EXISTING: unchanged

tests/
├── test_cli.py          # MODIFIED: add "claude" and "opencode" to SUBCOMMANDS list
├── test_agents.py       # NEW: test agent definitions, env var builders, binary lookup
└── test_process.py      # NEW: test fork+exec logic (mocked), cleanup behavior, signal handling
```

**Structure Decision**: Follows the existing flat module layout in `src/anaconda_ai/`. New functionality is split into two focused modules (`agents.py` for agent config, `process.py` for process management) rather than adding everything to `cli.py`, which is already 654 lines. The CLI commands in `cli.py` orchestrate these modules.

## Key Design Decisions

See [research.md](research.md) for full analysis. Summary:

1. **Process execution**: `os.fork()` + `os.execvpe()` with `setpgid` + `tcsetpgrp` for full TTY interactivity + post-exit cleanup. Windows fallback via `subprocess.Popen()`.
2. **Agent env vars**: Per-agent `build_env()` functions. Claude Code uses `ANTHROPIC_BASE_URL` (set to `server.url` — all backends serve `/v1/messages` at the base) + `ANTHROPIC_API_KEY`. OpenCode uses `OPENCODE_CONFIG_CONTENT` (inline JSON with `server.openai_url`).
3. **Argument parsing**: `--` separator via custom `AgentCommand(TyperCommand)` subclass. Click silently consumes `--` during parsing, so `AgentCommand.parse_args()` intercepts it first — stashing agent args in `ctx.meta['agent_args']` and stripping them before click runs. Typed options (`--detach`, `--backend`, `--at`, `--json`) work in any position. Extra server options go to `ctx.args`. Reuses `launch` command's extra_options pattern for server options.
4. **Server lifecycle**: Track `server_owned` flag. Only stop servers the wrapper launched (and only if not `--detach`).
