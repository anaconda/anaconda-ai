# Tasks: Coding Agent Wrapper Commands

**Input**: Design documents from `specs/001-coding-agent-wrapper/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md

**Tests**: Included — the existing project uses pytest with typer.testing.CliRunner and has coverage enforcement.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Create new source files and establish the module structure

- [x] T001 [P] Create agent definition module with `AgentDefinition` dataclass, empty `AGENTS` registry dict, and stub signatures for `find_agent_binary()`, `build_env_claude()`, `build_env_opencode()` (bodies raise `NotImplementedError`) in `src/anaconda_ai/agents.py`
- [x] T002 [P] Create process launcher module with `run_agent_foreground()` implementing the fork+exec+tcsetpgrp pattern from research.md Decision 1 (including `_child_exec()`, `_parent_wait_and_cleanup()`, `_safe_tcsetpgrp()`, and Windows fallback via `subprocess.Popen`) in `src/anaconda_ai/process.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core wrapper logic that all user stories depend on — the shared `_run_wrapper()` helper and CLI command registration

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Add `_run_wrapper()` function scaffold in `src/anaconda_ai/cli.py` — define the function signature with parameters for `agent_name`, `ctx: typer.Context`, `model_or_server`, `site`, `backend`, `remove`, `as_json`. Include argument parsing for `ctx.args` extra options (reuse `launch` pattern lines 307-316) and the `server/` prefix detection. Leave the server creation, agent launch, and cleanup sections as `# TODO` placeholders. Follow the existing `launch` command's pattern for `--backend`/`--at`/`--rm/--detach` options and `extra_options` parsing.
- [x] T004 Add `@app.command("claude")` and `@app.command("opencode")` command definitions in `src/anaconda_ai/cli.py` that each call `_run_wrapper()` with the appropriate agent name from the AGENTS registry. Use `context_settings={"allow_extra_args": True, "ignore_unknown_options": True}` matching the `launch` command pattern.
- [x] T005 Add `"claude"` and `"opencode"` to the `SUBCOMMANDS` list in `tests/test_cli.py` to get `--help` smoke test coverage for both new commands.
- [x] T005B [P] Implement interrupt cleanup in `src/anaconda_ai/process.py` and `src/anaconda_ai/cli.py` — wrap server creation + agent launch in try/except `KeyboardInterrupt` so Ctrl+C during server startup (before fork) still cleans up owned servers. Follow the `launch` command's existing pattern (lines 340-351 of cli.py). (FR-015)

**Checkpoint**: Both commands are registered and respond to `--help`. The full wrapper flow is wired end-to-end. Individual behaviors are validated in the user story phases below.

---

## Phase 3: User Story 1 — Launch Model and Run Coding Agent (Priority: P1) 🎯 MVP

**Goal**: `anaconda ai claude <model> -- <agent-args>` launches a server, starts the agent with correct env vars, forwards args after `--`, and exits with the agent's exit code.

**Independent Test**: Run `anaconda ai claude <model> -- --help` and verify the agent receives correct environment and arguments.

### Tests for User Story 1

- [x] T006 [P] [US1] Write test in `tests/test_agents.py`: `test_build_env_claude()` — given a mock Server with `url="http://localhost:8080"` and `api_key="test-key"`, assert `build_env_claude()` returns `{"ANTHROPIC_BASE_URL": "http://localhost:8080", "ANTHROPIC_API_KEY": "test-key"}` (uses `server.url` directly — all backends serve `/v1/messages` at the base URL)
- [x] T007 [P] [US1] Write test in `tests/test_agents.py`: `test_build_env_opencode()` — given a mock Server, assert `build_env_opencode()` returns a dict with `OPENCODE_CONFIG_CONTENT` key containing valid JSON with the server's baseURL and apiKey in the provider config
- [x] T008 [P] [US1] Write test in `tests/test_agents.py`: `test_find_agent_binary_found()` — monkeypatch `shutil.which` to return `/usr/bin/claude`, assert `find_agent_binary("claude")` returns the path
- [x] T009 [P] [US1] Write test in `tests/test_agents.py`: `test_find_agent_binary_not_found()` — monkeypatch `shutil.which` to return `None`, assert `find_agent_binary("claude")` raises or returns None with the install hint from the AgentDefinition
- [x] T010 [P] [US1] Write test in `tests/test_process.py`: `test_run_agent_foreground_exit_code()` — mock `os.fork` to simulate child exiting with code 42, assert `run_agent_foreground()` returns 42
- [x] T011 [P] [US1] Write test in `tests/test_process.py`: `test_run_agent_foreground_env_passed()` — mock fork+exec, assert `os.execvpe` is called with the correct env dict merged into `os.environ`
- [x] T012 [P] [US1] Write test in `tests/test_process.py`: `test_run_agent_foreground_args_forwarded()` — mock fork+exec, call with `agent_args=["--verbose", "--no-confirm"]`, assert `os.execvpe` is called with `["claude", "--verbose", "--no-confirm"]`
- [x] T012B [P] [US1] Write test in `tests/test_process.py`: `test_windows_fallback()` — monkeypatch `os` to not have `fork` attribute (`delattr` or `hasattr` returns False), call `run_agent_foreground()`, assert `subprocess.Popen` is used instead of `os.fork`

### Implementation for User Story 1

- [x] T013 [US1] Implement `build_env_claude(server, model_name)` in `src/anaconda_ai/agents.py` — return dict with `ANTHROPIC_BASE_URL` set to `server.url` (the raw base URL — all backends serve the Anthropic `/v1/messages` route at their base) and `ANTHROPIC_API_KEY` set to `server.api_key` or `"not-needed"`
- [x] T014 [US1] Implement `build_env_opencode(server, model_name)` in `src/anaconda_ai/agents.py` — return dict with `OPENCODE_CONFIG_CONTENT` containing JSON per research.md Decision 2 OpenCode section
- [x] T015 [US1] Implement `find_agent_binary(binary_name)` in `src/anaconda_ai/agents.py` — use `shutil.which()`, return path or None
- [x] T016 [US1] Implement `run_agent_foreground(binary_path, agent_args, env, cleanup_fn)` in `src/anaconda_ai/process.py` — the full fork+exec+tcsetpgrp+waitpid+cleanup flow per research.md Decision 1 Implementation Protocol. Include Windows fallback using `subprocess.Popen` when `os.fork` is not available. The parent's `waitpid()` loop must NOT catch or suppress errors originating from the agent process (e.g., connection failures from a crashed server) — propagate whatever exit code the agent returns.
- [x] T017 [US1] Complete `_run_wrapper()` body in `src/anaconda_ai/cli.py` — fill in: `client.servers.create()` → `server.start(show_progress=not as_json, console=console)` (matching `launch` pattern at cli.py:326 to surface download/startup progress per FR-014) → show server info → `find_agent_binary()` (exit with install hint if not found, FR-010) → `build_env()` → `run_agent_foreground()` → `sys.exit(exit_code)`.
- [x] T018 [US1] Run `make test` and verify all US1 tests pass and `--help` smoke tests pass for both commands

**Checkpoint**: `anaconda ai claude <model>` works end-to-end. Server launches, agent starts with correct env, args forwarded, exit code preserved.

---

## Phase 4: User Story 2 — Connect to Existing Server by ID (Priority: P2)

**Goal**: `anaconda ai claude server/<id> -- <args>` connects to a running server by ID without launching a new one.

**Independent Test**: Launch a server via `anaconda ai launch`, then run `anaconda ai claude server/<id> -- --help` and verify connection.

### Tests for User Story 2

- [x] T019 [P] [US2] Write test in `tests/test_cli.py`: `test_parse_server_id()` — assert `"server/abc123"` is parsed as server ID `"abc123"`, and `"OpenHermes-2.5"` is parsed as a model name
- [x] T020 [P] [US2] Write test in `tests/test_cli.py`: `test_server_id_not_found()` — mock `client.servers.get()` to raise/return None for unknown ID, assert wrapper exits with error message "Server 'xyz' not found or not running"

### Implementation for User Story 2

- [x] T021 [US2] Add `server/<id>` positional argument parsing to `_run_wrapper()` in `src/anaconda_ai/cli.py` — detect `server/` prefix, extract ID, look up server via `client.servers.get(id)` or equivalent, validate it's running, skip `servers.create()` path. Set `server_owned = False`.
- [x] T022 [US2] Add error handling for server-not-found case — display "Server '<id>' not found or not running" and exit with code 1
- [x] T023 [US2] Run `make test` and verify all US2 tests pass alongside existing US1 tests

**Checkpoint**: Both `anaconda ai claude <model>` and `anaconda ai claude server/<id>` work. Server lookup path skips creation.

---

## Phase 5: User Story 3 — Server Lifecycle / Cleanup (Priority: P2)

**Goal**: Server auto-stops on agent exit (default) unless `--detach` or server was pre-existing.

**Independent Test**: Run wrapper, exit agent, verify server stopped. Run with `--detach`, exit, verify server still running.

### Tests for User Story 3

- [x] T024 [P] [US3] Write test in `tests/test_process.py`: `test_cleanup_called_for_owned_server()` — mock fork+exec, pass `cleanup_fn`, assert cleanup_fn is called after child exits
- [x] T025 [P] [US3] Write test in `tests/test_process.py`: `test_cleanup_not_called_for_detach()` — pass `cleanup_fn=None` (detach mode), assert no server stop/delete calls
- [x] T026 [P] [US3] Write test in `tests/test_process.py`: `test_cleanup_not_called_for_preexisting_server()` — set `server_owned=False`, assert no cleanup

### Implementation for User Story 3

- [x] T027 [US3] Implement cleanup logic in `_run_wrapper()` in `src/anaconda_ai/cli.py` — build a `cleanup_fn` that calls `server.stop()` + `server.delete()` only when `server_owned=True` and `detach=False`. Pass `cleanup_fn` to `run_agent_foreground()`. For `server_owned=False` or `detach=True`, pass `cleanup_fn=None`.
- [x] T028 [US3] Run `make test` and verify all US3 tests pass alongside US1 and US2 tests

**Checkpoint**: Server lifecycle is fully managed. Owned servers stop on exit, pre-existing servers are left alone, `--detach` leaves servers running.

---

## Phase 6: User Story 4 — Multiple Agent Support (Priority: P3)

**Goal**: Both `claude` and `opencode` commands work with agent-specific env var mappings. New agents can be added by registering an AgentDefinition.

**Independent Test**: Run both `anaconda ai claude <model>` and `anaconda ai opencode <model>` and verify each sets the correct agent-specific env vars.

### Tests for User Story 4

- [x] T030 [P] [US4] Write test in `tests/test_agents.py`: `test_agents_registry_has_claude_and_opencode()` — assert AGENTS dict contains `"claude"` and `"opencode"` keys with correct AgentDefinition fields
- [x] T031 [P] [US4] Write test in `tests/test_agents.py`: `test_agent_install_hints()` — assert each agent's `install_hint` is non-empty and contains a URL or package name

### Implementation for User Story 4

- [x] T032 [US4] Wire the AGENTS registry in `src/anaconda_ai/agents.py` — add complete `AgentDefinition` entries for `"claude"` (binary=`"claude"`, build_env=`build_env_claude`, install_hint with npm command) and `"opencode"` (binary=`"opencode"`, build_env=`build_env_opencode`, install_hint with URL). This connects the implementations from T013/T014/T015 into the lookup dict used by `_run_wrapper()`.
- [x] T033 [US4] Ensure `_run_wrapper()` in `src/anaconda_ai/cli.py` uses the AGENTS registry to look up agent config by command name, and that the error message for missing binary includes the agent-specific `install_hint`
- [x] T034 [US4] Run `make test` and verify all tests pass

**Checkpoint**: Both agent commands work with distinct env var mappings. Adding a third agent (e.g., aider) would only require a new AgentDefinition entry.

---

## Phase 7: User Story 5 — Server Configuration Options (Priority: P3)

**Goal**: Users can pass `--backend`, `--at`, and extra `--key=value` server options through the wrapper, matching `anaconda ai launch` behavior.

**Independent Test**: Run `anaconda ai claude <model> --backend ai-navigator --ctx-size=4096 -- --verbose` and verify server uses the specified backend and options.

### Tests for User Story 5

- [x] T035 [P] [US5] Write test in `tests/test_cli.py`: `test_extra_server_options_parsed()` — invoke wrapper with `--ctx-size=4096 --jinja`, assert these are passed to `servers.create(extra_options=...)` (mock the client)

### Implementation for User Story 5

- [x] T036 [US5] Ensure `_run_wrapper()` in `src/anaconda_ai/cli.py` parses `ctx.args` for extra `--key=value` server options using the identical pattern from the `launch` command (lines 307-316), and passes them as `extra_options` to `client.servers.create()`. Verify `--backend` and `--at` options are passed to `AnacondaAIClient(backend=backend, site=site)`.
- [x] T037 [US5] Run `make test` and verify all tests pass

**Checkpoint**: Full `anaconda ai launch` option parity. Backend selection, site selection, and extra server options all work through the wrapper.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Quality improvements across all user stories

- [x] T038 [P] Add type hints to all public functions in `src/anaconda_ai/agents.py` and `src/anaconda_ai/process.py`
- [x] T039 [P] Run `ruff check .` and fix any linting issues in new files
- [x] T040 Run full `make test` — verify all tests pass with coverage above project threshold
- [ ] T041 Validate quickstart.md examples work end-to-end (manual smoke test)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): No dependencies on other stories
  - US2 (P2): No dependencies on other stories (uses same `_run_wrapper()`)
  - US3 (P2): Logically builds on US1 (cleanup requires the launch flow) but uses the same code paths
  - US4 (P3): No dependencies on other stories (registry pattern is set up in Setup)
  - US5 (P3): No dependencies on other stories (option parsing is in `_run_wrapper()`)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1**: Can start after Phase 2 — delivers the complete MVP
- **US2**: Can start after Phase 2 — adds `server/<id>` parsing to the shared `_run_wrapper()`
- **US3**: Can start after Phase 2 — adds cleanup logic to `_run_wrapper()` and `run_agent_foreground()`
- **US4**: Can start after Phase 2 — populates agent registry (partially done in Setup)
- **US5**: Can start after Phase 2 — adds extra options parsing to `_run_wrapper()`

### Within Each User Story

- Tests are written first and should FAIL before implementation
- Agent definitions before process launcher
- Process launcher before CLI wiring
- Core implementation before error handling

### Parallel Opportunities

- T001 and T002 (Setup) can run in parallel — different files
- T006–T012 (US1 tests) can all run in parallel — different test functions
- T013, T014, T015 (US1 agent impls) can partially parallel — all in agents.py but independent functions
- US2, US3, US4, US5 can run in parallel once Phase 2 is complete (if team capacity allows)
- All test tasks within a story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (they all go in different test files/functions):
Task T006: "test_build_env_claude() in tests/test_agents.py"
Task T007: "test_build_env_opencode() in tests/test_agents.py"
Task T008: "test_find_agent_binary_found() in tests/test_agents.py"
Task T010: "test_run_agent_foreground_exit_code() in tests/test_process.py"
Task T011: "test_run_agent_foreground_env_passed() in tests/test_process.py"
Task T012: "test_run_agent_foreground_args_forwarded() in tests/test_process.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T002)
2. Complete Phase 2: Foundational (T003–T005)
3. Complete Phase 3: User Story 1 (T006–T018)
4. **STOP and VALIDATE**: `anaconda ai claude <model> -- --help` works end-to-end
5. Deploy/demo if ready — this alone delivers the core value proposition

### Incremental Delivery

1. Setup + Foundational → Commands registered, `--help` works
2. + US1 → Full launch+run flow works (MVP!)
3. + US2 → `server/<id>` connection works
4. + US3 → Auto-cleanup on exit works
5. + US4 → OpenCode agent works alongside Claude
6. + US5 → Backend/site/extra options work
7. Each story adds value without breaking previous stories

---

## Phase 9: Fix `--` Separator Parsing (Post-Implementation Discovery)

**Purpose**: Fix click/typer silently consuming `--` separator. Click's parser strips `--` at `parser.py#L327-330` before it reaches `ctx.args`. The fix uses a custom `AgentCommand(TyperCommand)` subclass that overrides `parse_args()` to intercept `--` and stash agent args in `ctx.meta['agent_args']` before click processes the remaining args. No UX constraints on argument ordering.

- [x] T042 Add `AgentCommand(TyperCommand)` subclass with `parse_args()` override in `src/anaconda_ai/cli.py`
- [x] T043 Update `claude` and `opencode` commands to use `cls=AgentCommand`
- [x] T044 Update `_run_wrapper()` to read `agent_args` from `ctx.meta['agent_args']` instead of scanning `ctx.args` for `--`
- [x] T045 [P] Write test: `test_separator_preserved_and_typed_options_parsed()` — typed opts after positional + server opts + `--` + agent args all work correctly
- [x] T046 [P] Write test: `test_separator_splits_server_and_agent_args()` — verify `extra_options` gets server opts and agent only gets args after `--`
- [x] T047 Update spec artifacts (research.md, plan.md, contracts/cli.md, quickstart.md) with final approach
- [x] T048 Run `make test` and verify all 40 tests pass

---

## Phase 10: Fix Parent Suspension During Cleanup (Post-Implementation Discovery)

**Purpose**: Fix parent process getting suspended (`SIGTSTP`) during child exit, preventing server cleanup. The parent only masked `SIGINT` while waiting — it also needed to mask `SIGTSTP`, `SIGTTIN`, and `SIGTTOU` to avoid being stopped by the shell during the foreground process group transition.

- [x] T049 Mask `SIGTSTP`, `SIGTTIN`, `SIGTTOU` (in addition to existing `SIGINT`) before `waitpid()` in `_parent_wait_and_cleanup()` in `src/anaconda_ai/process.py`
- [x] T050 Keep all four signals masked through terminal reclaim + cleanup; restore only after cleanup completes
- [x] T051 Remove redundant `SIGTTOU` ignore/restore block around `tcsetpgrp()` (now covered by the broader signal mask)
- [x] T052 Update research.md Decision 1 (Implementation Protocol + new Signal Handling in Parent section)
- [x] T053 Update plan.md Key Decision 1 with signal mask detail
- [x] T054 Run `make test` and verify all 40 tests pass

---

## Phase 11: Inject `--model` Flag for Coding Agents (Post-Implementation Discovery)

**Purpose**: Both OpenCode and Claude Code benefit from receiving the model name via `--model` CLI flag. OpenCode's TUI resolves the model with priority: `--model` CLI flag > config `"model"` field > session history > first available. The config `"model"` field (set via `OPENCODE_CONFIG_CONTENT`) is subject to `isModelValid()` checks that can silently fail. Claude Code uses `--model` for UI display and future-proofing — the local server ignores the model field but the UI shows it.

**Design**: Add a `build_agent_args()` callable to `AgentDefinition`. For OpenCode, it returns `["--model=anaconda/<model-name>"]` (equals-separated, with provider prefix). For Claude Code, it returns `["--model", model_name]` (space-separated, bare model name). The wrapper prepends these args before the user's `--` args when calling `run_agent_foreground()`.

- [x] T055 Add `build_agent_args` field to `AgentDefinition` dataclass in `src/anaconda_ai/agents.py` — `Callable[[str], List[str]]` that takes `model_name` and returns agent-specific CLI args to inject
- [x] T056 Implement `build_agent_args_opencode(model_name)` in `src/anaconda_ai/agents.py` — returns `["--model=anaconda/<model-name>"]`
- [x] T057 Implement `build_agent_args_claude(model_name)` in `src/anaconda_ai/agents.py` — returns `["--model", model_name]` (Claude Code accepts `--model` for UI display and future-proofing, space-separated with bare model name)
- [x] T058 Wire `build_agent_args` into AGENTS registry entries for both `"claude"` and `"opencode"` in `src/anaconda_ai/agents.py`
- [x] T059 Update `_run_wrapper()` in `src/anaconda_ai/cli.py` — call `agent.build_agent_args(model_name)` and prepend the result to `agent_args` before passing to `run_agent_foreground()`
- [x] T060 [P] Write test in `tests/test_agents.py`: `test_build_agent_args_opencode()` — assert returns `["--model=anaconda/MyModel"]` for model_name `"MyModel"`
- [x] T061 [P] Write test in `tests/test_agents.py`: `test_build_agent_args_claude()` — assert returns `["--model", "MyModel"]` for model_name `"MyModel"`
- [x] T062 [P] Write test in `tests/test_cli.py`: `test_opencode_injects_model_flag()` — mock `run_agent_foreground`, invoke `anaconda ai opencode MyModel/Q4_K_M`, assert `agent_args` starts with `["--model=anaconda/MyModel/Q4_K_M"]`
- [x] T063 [P] Write test in `tests/test_cli.py`: `test_claude_injects_model_flag()` — mock `run_agent_foreground`, invoke `anaconda ai claude MyModel/Q4_K_M`, assert `agent_args` starts with `["--model", "MyModel/Q4_K_M"]`
- [x] T064 Run `make test` and verify all tests pass

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- The project targets Python 3.10+ — `subprocess.Popen(process_group=)` is 3.11+ only, so we use `os.fork()` directly
- Windows fallback in `process.py` uses `subprocess.Popen()` without process group isolation
- All new code goes in `src/anaconda_ai/` — no new packages or entry point changes needed
- The `_run_wrapper()` shared function is the heart of the design — US2-US5 all add behavior to this function
