# Feature Specification: Coding Agent Wrapper Commands

**Feature Branch**: `001-coding-agent-wrapper`
**Created**: 2026-03-06
**Status**: Draft
**Input**: User description: "Write a coding agent wrapper (like 'anaconda ai claude' and 'anaconda ai opencode') that takes inputs similar to 'anaconda ai launch' to launch or connect to running servers and then pass any further arguments after '--' onto the coding agent (like claude and opencode). This wrapper will set any appropriate env vars so that the requested/launched server is utilized by the coding agent being run."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Launch Model and Run Coding Agent in One Command (Priority: P1)

A developer wants to start a coding session with a specific local model powering the AI. They run a single command that launches (or reuses) an inference server for their chosen model and immediately starts their preferred coding agent (e.g., Claude Code, OpenCode) with the server already configured as the backend. Arguments after `--` are forwarded directly to the coding agent.

**Why this priority**: This is the core value proposition — eliminating the manual multi-step workflow of launching a server, copying URLs, setting environment variables, and then starting a coding agent. Without this, the feature has no purpose.

**Independent Test**: Can be fully tested by running `anaconda ai claude <model> -- --help` and verifying that the coding agent receives the correct environment and arguments, and delivers specific value by reducing a multi-step workflow to a single command.

**Acceptance Scenarios**:

1. **Given** no server is running for the requested model, **When** the user runs `anaconda ai claude <model> -- <agent-args>`, **Then** a new server is launched for that model, the coding agent process is started with the server's endpoint and credentials injected as environment variables, and all arguments after `--` are forwarded to the coding agent.
2. **Given** the user runs the wrapper command, **When** the coding agent process exits, **Then** control returns to the user's terminal with the coding agent's exit code preserved.

---

### User Story 2 - Connect to Existing Server by ID (Priority: P2)

A developer already has an inference server running (started earlier or by another tool) and wants to connect their coding agent to it without launching a new one. They pass `server/<server-id>` as the positional argument (in place of a model name) and the wrapper looks up that server and configures the coding agent accordingly. For example: `anaconda ai claude server/abc123 -- <agent-args>`.

**Why this priority**: Supports workflows where servers are long-lived or shared, avoiding unnecessary server lifecycle overhead. Complements the primary launch flow.

**Independent Test**: Can be tested by first launching a server via `anaconda ai launch`, noting the server ID, then running `anaconda ai claude server/<id> -- --help` and verifying the coding agent connects to the pre-existing server.

**Acceptance Scenarios**:

1. **Given** a server is running with a known server ID, **When** the user runs `anaconda ai claude server/<server-id> -- <agent-args>`, **Then** the coding agent is started with that server's endpoint and credentials, and no new server is created.
2. **Given** the user provides a server ID that does not exist or is not running, **When** the user runs `anaconda ai claude server/<server-id> -- <agent-args>`, **Then** a clear error message is displayed indicating the server was not found or is not running.
3. **Given** the user passes a positional argument that does not start with `server/`, **When** the wrapper command is invoked, **Then** the argument is treated as a model identifier and the standard launch-or-reuse flow is used.

---

### User Story 3 - Server Lifecycle Matches Coding Agent Session (Priority: P2)

A developer wants their inference server to automatically stop when their coding agent session ends, so they don't leave orphaned servers consuming resources. Alternatively, they may want the server to persist after the session for reuse.

**Why this priority**: Resource management is critical for local development. Users should not need to manually track and stop servers after every coding session, but power users may want to detach.

**Independent Test**: Can be tested by running the wrapper, exiting the coding agent, and verifying the server is stopped (default) or left running (with detach flag).

**Acceptance Scenarios**:

1. **Given** the wrapper launched a new server (not pre-existing), **When** the coding agent process exits, **Then** the server is stopped and cleaned up by default.
2. **Given** the wrapper was told to detach the server, **When** the coding agent process exits, **Then** the server continues running.
3. **Given** the wrapper connected to a pre-existing server (not launched by this invocation), **When** the coding agent process exits, **Then** the server is left running regardless of detach settings.

---

### User Story 4 - Support Multiple Coding Agents (Priority: P3)

The system supports multiple coding agents (initially Claude Code and OpenCode) as separate subcommands, each with the appropriate environment variable mapping for that agent. Additional agents can be added in the future.

**Why this priority**: Multi-agent support differentiates this from a one-off script, but the core wrapper pattern is the same regardless of which agent is invoked. Supporting one agent fully delivers the primary value.

**Independent Test**: Can be tested by running each supported agent subcommand and verifying the correct agent-specific environment variables are set and the correct agent binary is invoked.

**Acceptance Scenarios**:

1. **Given** the user runs `anaconda ai claude <model> -- <args>`, **When** the coding agent launches, **Then** the `claude` binary is invoked with environment variables appropriate for Claude Code (OpenAI-compatible base URL and API key).
2. **Given** the user runs `anaconda ai opencode <model> -- <args>`, **When** the coding agent launches, **Then** the `opencode` binary is invoked with environment variables appropriate for OpenCode.
3. **Given** the user runs a wrapper command for an agent that is not installed on their system, **When** the command is invoked, **Then** a clear error message is displayed explaining which tool is missing and how to install it.

---

### User Story 5 - Pass Server Configuration Options (Priority: P3)

A developer needs to customize the inference server's configuration (e.g., context window size, enabling tool calling via Jinja templates) when launching it through the wrapper, just as they can with `anaconda ai launch`. They also need to select which backend or site to use via `--backend` and `--at` options, consistent with other `anaconda ai` commands.

**Why this priority**: Power users need server tuning and backend selection, but the feature is fully usable with default server settings and the default backend. This builds on the existing `anaconda ai launch` extra options pattern.

**Independent Test**: Can be tested by passing server options and backend/site flags, and verifying the launched server reflects those settings against the correct backend.

**Acceptance Scenarios**:

1. **Given** the user provides server configuration options to the wrapper command, **When** the server is launched, **Then** those options are applied to the server configuration.
2. **Given** the user provides both server options and coding agent arguments (after `--`), **When** the command is parsed, **Then** server options and agent arguments are correctly separated and applied to their respective targets.
3. **Given** the user provides `--backend` and/or `--at` options, **When** the wrapper command is invoked, **Then** the specified backend and site are used for server operations, consistent with how other `anaconda ai` commands handle these options.

---

### Edge Cases

- What happens when the model has not been downloaded yet? The server creation flow handles downloading automatically (existing behavior), so the wrapper should surface download progress to the user.
- What happens when the coding agent process is killed with SIGTERM or SIGKILL? The wrapper should still attempt server cleanup for servers it launched.
- What happens when the user interrupts with Ctrl+C during server startup (before the coding agent starts)? The partially started server should be cleaned up.
- What happens when multiple wrapper instances try to use the same model simultaneously? The existing server matching logic already handles this — both should connect to the same server.
- What happens when the inference server crashes while the coding agent is running? The coding agent will encounter connection errors; the wrapper should not mask these errors.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a subcommand for each supported coding agent (initially `claude` and `opencode`) under the `anaconda ai` command namespace.
- **FR-002**: Each wrapper subcommand MUST accept a positional argument that is either a model identifier (to launch/reuse a server) or a `server/<server-id>` reference (to connect to an existing server).
- **FR-003**: When the positional argument is a model identifier, the wrapper MUST launch an inference server for the specified model if one is not already running, reusing the existing server creation and matching logic.
- **FR-004**: When the positional argument matches the `server/<server-id>` pattern, the wrapper MUST look up that server by ID and use it directly, without launching a new server. If the server does not exist or is not running, the wrapper MUST display a clear error and exit.
- **FR-005**: The wrapper MUST set environment variables in the coding agent's process so the agent communicates with the launched/matched server. At minimum, the server's OpenAI-compatible base URL and an API key must be provided.
- **FR-006**: The wrapper MUST forward all arguments appearing after `--` to the coding agent process verbatim.
- **FR-007**: The wrapper MUST preserve the coding agent's exit code as its own exit code.
- **FR-008**: The wrapper MUST stop and clean up a server it launched when the coding agent exits, unless the user explicitly requested detach mode.
- **FR-009**: The wrapper MUST NOT stop a server that was already running before the wrapper was invoked.
- **FR-010**: The wrapper MUST display a clear error and exit if the specified coding agent binary is not found on the system PATH.
- **FR-011**: The wrapper MUST support the `--backend` and `--at` (site) options consistent with other `anaconda ai` commands.
- **FR-012**: The wrapper MUST support a `--detach` flag that leaves the server running after the coding agent exits.
- **FR-013**: The wrapper MUST support passing server configuration options (e.g., context size, Jinja flag) using the same mechanism as `anaconda ai launch`.
- **FR-014**: The wrapper MUST surface server download and startup progress to the user before launching the coding agent.
- **FR-015**: The wrapper MUST attempt server cleanup if the process is interrupted (e.g., Ctrl+C) during or after server startup, for servers it launched.

### Key Entities

- **Coding Agent**: An external tool (e.g., Claude Code, OpenCode) that provides AI-assisted coding capabilities and accepts an inference server endpoint via environment variables. Each agent has a known binary name and a mapping of environment variable names it expects.
- **Inference Server**: A running instance of a model server managed by the Anaconda AI backend, exposing an OpenAI-compatible endpoint. Represented by the existing `Server` object with `openai_url` and `api_key` attributes.
- **Wrapper Command**: A CLI subcommand that orchestrates the lifecycle of launching/connecting to an inference server and then executing a coding agent with the correct environment configuration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can go from zero (no server running) to an active coding agent session in a single command invocation.
- **SC-002**: The coding agent successfully communicates with the Anaconda AI inference server without any manual configuration by the user.
- **SC-003**: Server resources are automatically cleaned up after 100% of non-detached coding agent sessions where the wrapper launched the server.
- **SC-004**: All arguments after `--` are received by the coding agent identically to how they would be received if the user ran the agent directly.
- **SC-005**: Users familiar with `anaconda ai launch` can use the wrapper commands without consulting documentation, due to consistent option naming and behavior.
- **SC-006**: When a required coding agent tool is not installed, the user receives actionable guidance within the error message.

## Assumptions

- The coding agents (Claude Code and OpenCode) are installed separately by the user and available on the system PATH. The wrapper does not install them.
- Coding agents accept OpenAI-compatible server configuration via standard environment variables (e.g., `OPENAI_BASE_URL`, `OPENAI_API_KEY`, or agent-specific equivalents).
- The existing `AnacondaAIClient.servers.create()` method handles model downloading, server matching, and server creation — the wrapper delegates to this existing infrastructure.
- The wrapper runs the coding agent as a child process in the foreground, inheriting the user's terminal (stdin/stdout/stderr).
- The API key provided by the `Server` object is sufficient for authentication with the local inference server (no additional auth flow needed).
- Server configuration options (extra args before `--`) follow the same `--key=value` pattern as `anaconda ai launch`.
