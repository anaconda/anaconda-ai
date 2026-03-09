# Quickstart: Coding Agent Wrapper Commands

**Feature Branch**: `001-coding-agent-wrapper`

## Prerequisites

- `anaconda-ai` installed (`conda install -c anaconda-cloud anaconda-ai`)
- A backend running (AI Navigator, Anaconda Desktop, or AI Catalyst)
- At least one coding agent installed:
  - Claude Code: `npm install -g @anthropic-ai/claude-code`
  - OpenCode: See https://opencode.ai for installation

## Usage

### Basic: Launch a model and start Claude Code

```bash
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M
```

This will:
1. Download the model if needed (progress shown)
2. Launch an inference server
3. Start Claude Code with the server configured
4. Stop the server when Claude Code exits

### Basic: Launch a model and start OpenCode

```bash
anaconda ai opencode OpenHermes-2.5-Mistral-7B/Q4_K_M
```

### Connect to an existing server

```bash
# First, find your running servers
anaconda ai servers

# Then connect to one
anaconda ai claude server/<server-id>
```

### Pass arguments to the coding agent

Use `--` to separate wrapper options from agent arguments:

```bash
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M -- --verbose --no-confirm
```

### Server configuration options

Pass server options before `--` (same syntax as `anaconda ai launch`):

```bash
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --ctx-size=4096 --jinja
```

### Keep server running after exit

```bash
anaconda ai claude OpenHermes-2.5-Mistral-7B/Q4_K_M --detach
```

## Development

### Running tests

```bash
make test
```

### Key source files

| File | Purpose |
|------|---------|
| `src/anaconda_ai/cli.py` | CLI command definitions (add new `@app.command()` here) |
| `src/anaconda_ai/agents.py` | Agent definitions and env var builders (new file) |
| `src/anaconda_ai/process.py` | fork+exec+tcsetpgrp process launcher (new file) |
| `tests/test_cli.py` | CLI smoke tests |
| `tests/test_agents.py` | Agent env var builder tests (new file) |
| `tests/test_process.py` | Process launcher tests (new file) |
