# QA automation guidelines (Playwright & TypeScript)

## Overview

These rules guide QA automation work in this repo (Playwright and TypeScript). **Automated rules** live in `eslint.config.base.mjs`, `eslint.config.mjs`, and `scripts/check-code-quality.sh`—follow ESLint and script output as the source of truth for anything they enforce.

## Scope and file application

### Files where rules apply

- **TypeScript/JavaScript test files**: `*.ts`, `*.js`, `*.spec.ts`, `*.page.ts`
- **Test directories**: `tests/` and subdirectories
- **Page Object Models**: `tests/pages/` (api/, ui/, shared/)
- **Test specifications**: `tests/specs/` (api/, e2e/, smoke/, regression/)
- **Test fixtures**: `tests/fixtures/`
- **Test utilities**: `tests/utils/`
- **Test configuration**: `tests/config/`
- **Test data**: TypeScript files in `tests/testdata/`
- **Playwright configuration**: `playwright.config.ts`
- **Related config**: `eslint.config.mjs`, `tsconfig.json` (when used for test automation)

### Files where rules do not apply

- **Python**: `*.py` (e.g. bot/, cli/, cookbook/, sdk/, machine-images/)
- **Shell**: `*.sh` (scripts/, machine-images/)
- **Other config**: `Makefile`, `package.json`, `serverless.yaml`, `*.yml`, `*.yaml`
- **Documentation**: `*.md`, README files
- **Build/deployment**: Files outside the test automation scope

### How rules are enforced

- **Guidelines**: Apply while editing in-scope files
- **Automated**: ESLint, Prettier, and `scripts/check-code-quality.sh` (run in CI / pre-commit as configured in the repo)

## General guidance (not fully expressed in ESLint / the shell script)

### Code organization and structure

- Use Anaconda Playwright Utils Page Object Model (POM) to organize fixtures, locators, actions, and assertions.
- Group related tests in logical folders and files; keep specs focused and reuse utilities for repeated flows.
- Apply SOLID and DRY sensibly; prioritize readability.
- **Avoid duplicate flows under different names** (e.g. multiple methods that do the same login with different names like signIn(), logIn()).

### Naming conventions

- **Files**: hyphen-separated names (e.g. `login-tests.spec.ts`, `user-management.page.ts`). **Automated:** `scripts/check-code-quality.sh` enforces lowercase hyphen-style basenames on `tests/specs/**/*.ts`, `tests/pages/**/*.ts`, and `src/**/*.ts` (`code-quality/file-naming`; runs on `--staged` for those paths even when no other test file is staged).
- **Methods/variables**: camelCase. **Automated:** `@typescript-eslint/naming-convention` in `eslint.config.base.mjs` (variables may also use `UPPER_CASE` / `PascalCase` for module constants and exported config objects—see rule in config).
- **Fixtures**: names that reflect scope and purpose.

### Playwright and test design

- Prefer robust selectors (`data-qa-id`, `data-testid`, or other stable hooks) over brittle CSS-only selectors when you have a choice.
- Keep each test self-contained; avoid depending on execution order of other tests.
- **Separation of concerns**: action methods perform interactions; assertion methods contain validation (including `expect()`).
- When using `test.skip()`, add a comment explaining why (the repo’s quality script checks for justification patterns).

### Security and data

- Do not hard-code credentials or secrets; use environment variables and patterns the team uses for config.

### Assertions

- Use clear string descriptions on tests.
- **Every `expect()` must include a message** (second argument) so failures are easy to triage in reports, e.g. `expect(autoFixObjectShorthand.bar, 'bar should be baz').toBe(bar);` — not enforced by ESLint alone; verify in review.

## Documentation and comments

- **JSDoc**: Required for **complex** methods per `scripts/check-jsdoc-complexity.js` (high complexity, many parameters, non-trivial logic). Simple actions (click, fill, straightforward `expect`) often do not need JSDoc—see that script’s behavior. In `npm run check:code-quality`, missing JSDoc is reported as **warning** severity (`code-quality/jsdoc-complexity`) and does not fail the step.
- Document non-obvious workarounds and special setup/teardown.

## Imports

- Obey ESLint import rules (`import/first`, `sort-imports`, etc.); see `eslint.config.base.mjs`.

## Error handling and logging

- Use try/catch where failures are expected and messages should be actionable.
- Prefer Playwright’s waiting APIs over arbitrary sleeps (also enforced via ESLint / restricted syntax for literals).

## Test directory layout (reference)

```
tests/
├── fixtures/
├── pages/
│   ├── api/
│   ├── ui/
│   └── shared/
├── test-plans/          # Markdown test plans (planner / generator input; not *.spec.ts)
├── specs/
│   ├── api/
│   ├── e2e/
│   ├── smoke/
│   └── regression/
├── testdata/
├── utils/
└── config/
```

## Before you commit

- **Consumer install root** — Quality and pre-commit CLIs resolve **`CONSUMER_ROOT`** via **`scripts/resolve-consumer-root.sh`**: the QA **npm package directory** where `npm run` was invoked (`package.json`, `lint-staged`, path scoping)—e.g. `functional_tests/`, `e2e/`, `packages/api/`, or repo root. This is **not** git root. In hoisted workspaces, `CONSUMER_ROOT` may not contain `node_modules/@anaconda/playwright-utils`; **`INIT_CWD`** selects the package dir over the hoist root. Package `scripts/` are resolved via **`SCRIPT_DIR`**, not `CONSUMER_ROOT`.
- **Git pre-commit** uses **`"precommit": "playwright-utils-precommit"`** in the **QA** `package.json` (Husky in that folder). The hook **does not run** on dev-only commits (no lint-staged, no report). When QA files are staged: **lint-staged** from **`CONSUMER_ROOT`**, then **`playwright-utils-commit-quality-report`** on **staged paths under `CONSUMER_REL` only**. For whole-repo fixes, use **`npm run format`** / **`npm run lint:fix`** or **`npm run quality:full`**.
- **`npm run quality:full`** — bin **`playwright-utils-quality-full`**: `format` + `lint:fix` + `lint` + code-quality from **`CONSUMER_ROOT`** (compact).
- **`npm run quality:report`** — bin **`playwright-utils-full-quality-report`**: steps [1]–[3] with banners, then **[4]** manual hints on the full tests tree under **CONSUMER_ROOT** via **`scan-manual-review-hints.sh --full-tests`** (**always**, even when [1]–[3] fail). Exit **0** if [1]–[3] pass; exit **1** otherwise. Bins follow `.bin` symlinks and `require.resolve('@anaconda/playwright-utils/package.json')` for companion scripts.

## Automated enforcement (where to look)

**Consumer `package.json`:** map npm script aliases to **`playwright-utils-*` bin names only**—never `bash ./scripts/...` from repo root. Bins are on `PATH` via `node_modules/.bin/` when `@anaconda/playwright-utils` is a direct dependency. **This library repo** is not a consumer install: npm does not link its own bins at the package root—maintainers use **`bash ./scripts/<name>.sh`** in `package.json` (not `npx playwright-utils-*`). See package README § Code quality checks for the full matrix and anti-patterns.

| Concern                                                                                                                       | Where it’s defined (package)                                                                                                  | Consumer `package.json`                                              |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Prettier, TypeScript, imports, Playwright rules, complexity warning, timeout/`setTimeout` literals, inline test-data patterns | `eslint.config.base.mjs`, `eslint.config.mjs` (consumer copies/extends)                                                       | `format`, `lint`, `lint:fix` (consumer-defined)                      |
| Large files, file naming (specs/pages/src basenames), `test.skip` justification, TODO + ticket, JSDoc complexity              | `scripts/check-code-quality.sh` (+ package-internal `check-jsdoc-complexity.js`)                                              | `check:code-quality` → **`playwright-utils-check-code-quality`**     |
| Consumer install root (`CONSUMER_ROOT`, `GIT_ROOT`, `CONSUMER_REL`)                                                           | package-internal **`resolve-consumer-root.sh`** (sourced by bins)                                                             | — (automatic)                                                        |
| Full-repo compact gate (`format` + `lint:fix` + `lint` + code-quality)                                                        | `scripts/full-quality.sh` / bin **`playwright-utils-quality-full`**                                                           | `quality:full` → **`playwright-utils-quality-full`**                 |
| Full-repo unified report ([1]–[3] banners + [4] manual hints on full tests tree under CONSUMER_ROOT, always; exit 0/1)        | `scripts/full-quality-report.sh` / bin **`playwright-utils-full-quality-report`**                                             | `quality:report` → **`playwright-utils-full-quality-report`**        |
| QA-scoped pre-commit (skip hook when no staged QA files)                                                                      | `scripts/precommit.sh` / bin **`playwright-utils-precommit`**                                                                 | `precommit` → **`playwright-utils-precommit`**                       |
| Pre-commit unified report (staged QA paths only)                                                                              | `scripts/commit-quality-report.sh` / bin **`playwright-utils-commit-quality-report`**                                         | **Do not wire** — invoked by **`playwright-utils-precommit`** only   |
| Manual-review hints (ad-hoc)                                                                                                  | `scripts/print-manual-review-hint.sh` / bin **`playwright-utils-print-manual-review-hint`** (+ package-internal scan scripts) | Optional: **`playwright-utils-print-manual-review-hint`** (consumer) |

Enumerated lists (what each gate checks, including rule IDs where applicable) are in **Quality gates catalog** below.

## Quality gates catalog

Use this section for onboarding: it separates **ESLint / Prettier**, **code quality scripts**, and **manual review** (plus optional heuristics).

### ESLint / Prettier

Authoritative machine config: [`eslint.config.base.mjs`](../../../../eslint.config.base.mjs) (shared/published as `@anaconda/playwright-utils/eslint`) and [`eslint.config.mjs`](../../../../eslint.config.mjs) (this repo). Below is a grouped summary; exact severities and edge cases are in those files.

| Category                                      | What is enforced                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Base presets**                              | `@eslint/js` recommended.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Formatting (Prettier + style)**             | `prettier/prettier`, `no-trailing-spaces`, `no-multiple-empty-lines`, `eol-last`.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **TypeScript**                                | `@typescript-eslint` recommended + `eslint-recommended` overrides; notable rules include `no-floating-promises`, `no-unused-vars` (with `_` ignore), `no-unused-expressions`, `prefer-nullish-coalescing`, `prefer-optional-chain`, `prefer-as-const`, `no-duplicate-enum-values`, `no-inferrable-types`, `require-await`, `await-thenable`, `no-misused-promises`; several `no-unsafe-*` and `no-explicit-any` as **warn**.                                                                                             |
| **Imports**                                   | `import/no-unresolved`, `import/named`, `import/default`, `import/no-absolute-path`, `import/no-self-import`, `import/first`, `import/no-mutable-exports`, `sort-imports` (with `ignoreDeclarationSort: true`).                                                                                                                                                                                                                                                                                                          |
| **General**                                   | `complexity` (warn, max 11), `no-console` (warn; allows `warn`, `error`, `info`), `no-debugger`, `no-alert`, `no-var`, `prefer-const`, `prefer-template`, `object-shorthand`, `no-lonely-if`, `no-useless-return`, `no-nested-ternary` (warn), `eqeqeq`, `no-throw-literal`, `curly`, `@typescript-eslint/naming-convention` (identifiers; relaxed object literal / type property / import names).                                                                                                                       |
| **JSDoc (plugin)**                            | Alignment/indentation are **off** in the base config (consumers); this repo turns them to **warn** in `eslint.config.mjs`.                                                                                                                                                                                                                                                                                                                                                                                               |
| **Playwright**                                | Spreads `eslint-plugin-playwright` `playwright-test` recommended rules, then sets explicit severities for e.g. `missing-playwright-await`, `no-focused-test`, `valid-expect`, `prefer-web-first-assertions`, `no-useless-await`, `no-page-pause`, `no-element-handle`, `no-eval`, `prefer-to-be`, `prefer-to-contain`, `prefer-to-have-length`, `require-top-level-describe`, `no-wait-for-timeout` (warn), `no-conditional-in-test` (warn), `no-force-option` (warn), and others as listed in `eslint.config.base.mjs`. |
| **Custom `no-restricted-syntax`**             | Blocks hard-coded timeout literals (`waitForTimeout`, `timeout` option literals, `setTimeout` delay literals) and patterns for **inline test-data objects** (variable names like `*Data` / `*TestData`) outside `tests/testdata/`. Files under `tests/testdata/**/*.ts` only get the timeout-related restricted-syntax rules (inline test-data patterns are allowed there).                                                                                                                                              |
| **Repo-only overrides** (`eslint.config.mjs`) | `jsdoc/check-alignment` and `jsdoc/check-indentation` as **warn**; `@typescript-eslint/explicit-module-boundary-types` **warn** for `src/**/*.ts` only.                                                                                                                                                                                                                                                                                                                                                                  |

### Code quality (`check-code-quality.sh` + `check-jsdoc-complexity.js`)

Runs on `tests/**/*.ts` (excludes `tests/scripts/fixtures`) for length, skip, TODO, and JSDoc; **file naming** additionally checks `tests/specs`, `tests/pages`, and `src` (see script). Pre-commit step [3] uses **`playwright-utils-check-code-quality --staged`** from **`CONSUMER_ROOT`** (staged paths filtered to **`CONSUMER_REL`**). Blocking vs non-blocking matches script severities (errors fail the step; warnings print but still allow exit 0 for that script). JSDoc complexity from `check-jsdoc-complexity.js` is **warning** output only (process exit **2**); `check-code-quality.sh` treats it as non-blocking alongside oversized files.

| Check                                                                                                                                                                                       | Severity | Rule id (in output)             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------- |
| TypeScript basename not lowercase hyphenated (under `tests/specs`, `tests/pages`, `src`)                                                                                                    | error    | `code-quality/file-naming`      |
| File longer than 1000 lines                                                                                                                                                                 | warning  | `code-quality/file-length`      |
| `test.skip` / `test.describe.skip` without a nearby justification (TODO/FIXME/issue/ticket patterns — see script)                                                                           | error    | `code-quality/test-skip`        |
| `TODO` comment without a ticket/issue pattern (e.g. `PROJ-123`, `#456`)                                                                                                                     | error    | `code-quality/todo-ticket`      |
| Complex methods missing JSDoc (cyclomatic complexity &gt; 5, or &gt; 3 parameters, or long body, complex name patterns, public/exported — per `scripts/check-jsdoc-complexity.js` `CONFIG`) | warning  | `code-quality/jsdoc-complexity` |

### Manual review (human + heuristics)

These items are only partially automatable; PR review and judgment apply.

#### Human checklist

After automated checks pass, verify on **changed** test/page code (detail also appears under **General guidance** above):

- **POM** — Actions vs assertions separated; reuse library utilities vs raw `page.*` where the codebase expects helpers.
- **Selectors** — Prefer stable hooks (`data-qa-id`, `data-testid`, or other stable hooks) over brittle CSS-only locators when you have a choice.
- **Secrets** — No hard-coded credentials in specs or page objects; use env/config or centralize under `tests/testdata/` (Hook [4] `manual-review/secrets` skips `tests/testdata/**/*.ts`, same as ESLint inline test-data override).
- **Duplication** — No parallel flows that do the same thing under different names (for example, two tests that each inline the same login → navigate → assert sequence instead of one shared helper, fixture, or page-object method). **Hook [4]** runs **`scripts/scan-duplicate-function-hints.js`** (via `scan-manual-review-hints.sh`) for duplicate **function/method bodies** — rule id **`manual-review/duplication`** (see **Duplicate function/method bodies** below). For test-level or partial overlap the AST scan does not cover, add `// MANUAL: Duplication — …` ( **`manual-review/tagged`** ).
- **Expect messages** — All assertions must pass a **message** as the second argument to `expect()` (Playwright/Jest-style), e.g. `expect(autoFixObjectShorthand.bar, 'bar should be baz').toBe(bar);`. Omitting it fails manual review.

#### Heuristic helpers (not a full audit)

[`scripts/scan-manual-review-hints.sh`](../../../../scripts/scan-manual-review-hints.sh) runs in pre-commit (`commit-quality-report.sh` section [4]) on **staged** `tests/**/*.ts` under **`CONSUMER_REL`** (explicit file arguments) and in **`npm run quality:report`** (`full-quality-report.sh` → [`print-manual-review-hint.sh`](../../../../scripts/print-manual-review-hint.sh) → **`--full-tests`**) on the **full** `CONSUMER_ROOT/tests/**/*.ts` tree (streamed `find`, excludes `scripts/fixtures`; duplicate scan uses the same file list via **`scan-duplicate-function-hints.js --files-from`**). The **`manual-review/secrets`** heuristic is skipped for `tests/testdata/**/*.ts` (aligned with ESLint: inline test-data objects are allowed there). Ad-hoc full-tree hints: run **`--full-tests`** from the QA install root (or via bins that set **`CONSUMER_ROOT`**). It emits **pointers only** —

| Heuristic                                                                                                                    | Rule id (in output)         |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Lines with `MANUAL:` in comments                                                                                             | `manual-review/tagged`      |
| Duplicate function/method bodies (AST; see below)                                                                            | `manual-review/duplication` |
| Possible hard-coded secret/credential patterns (excludes `tests/testdata/**/*.ts`, same as ESLint inline test-data override) | `manual-review/secrets`     |
| Possible brittle selector patterns (e.g. deep `nth-child` / chained `>`)                                                     | `manual-review/selector`    |
| No heuristic match for a file (reminder to use guidelines / skill)                                                           | `manual-review/no-match`    |

Use the **qa-automation-quality** Claude skill for a **failure-only** report that can include **manual/review** gaps against this doc.

#### Duplicate function/method bodies (`scripts/scan-duplicate-function-hints.js`)

TypeScript AST heuristic invoked from **`scripts/scan-manual-review-hints.sh`**. Emits **`manual-review/duplication`** (pre-commit section **[4]** is **WARN** only — does not block the commit if **[1]–[3]** pass).

|                                                       |                                                                                                                                               |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pre-commit (`commit-quality-report.sh` [4])**       | Staged **`tests/**/\*.ts`** only (explicit file list; not `src/\*\*`)                                                                         |
| **`quality:report` (`full-quality-report.sh` [4])**   | **`scan-manual-review-hints.sh --full-tests`** — same tree as below; no giant argv list                                                       |
| **CLI `scan-manual-review-hints.sh --full-tests`**    | All **`tests/**/\*.ts`** (excludes `scripts/fixtures`; not `src/\*\*`)                                                                        |
| **CLI `scan-duplicate-function-hints.js` (no paths)** | Default walk under `tests/` (skips `scripts/fixtures` and `demo-manual-dup-*` demos); **`--full-tests`** passes an explicit file list instead |
| **CLI with file args**                                | Exactly the paths given (`scan-manual-review-hints.sh <paths>` or `node scripts/scan-duplicate-function-hints.js <paths>`)                    |

**Scanned (named bodies at module or class-member scope; export not required):**

- Top-level `function` declarations
- Module-level `const` / `let` arrow functions and function expressions (block or expression body)
- Top-level `class` declarations (methods on the class are scanned)
- Nested classes via **property initializers** (`Inner = class { ... }` or `const Helper = class { ... }`) — not `class` declarations inside a class body (invalid in TypeScript)
- Class methods, constructors, and property arrow/function initializers
- Methods on module-level object literals (e.g. `export const helpers = { build() {} }`)
- `export default` class/function at module scope (statement-level `export default function/class`, or expression forms: arrows, `class` / `function` expressions, including parenthesized)

**Intentionally not scanned** (by design — reduces false positives in tests and callbacks; not a gap to fix in review):

- Anything declared **inside** a function, method, or `test()` / hook callback body (nested functions, nested classes, inline helpers)
- Accessors (`get` / `set`); unnamed declarations
- Coincidentally identical **short** bodies (&lt; 40 characters after normalize)
- Trivial literal-only bodies (e.g. `() => true`, `{ return [] }`)

Use `// MANUAL: Duplication — …` when duplicate logic lives only inside nested scopes or test steps.

**How bodies are compared:** Comments stripped; whitespace collapsed. Expression-bodied arrows and block bodies with a **single `return`** normalize to the same key (so `(x) => expr` matches `(x) => { return expr; }`). Multi-statement blocks compare as a full block. The **first** match in lexicographic file order is canonical; each later duplicate gets one warning. For `const` / property assignments, **`line:col` points at the binding** (identifier or property key), not the `=>` / `function` on the RHS.

**CLI / TSV output (intentional — not incomplete):**

| Mode                                            | Format                                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--tsv` (used by `scan-manual-review-hints.sh`) | Four tab-separated columns, **no header:** `file`, `line`, `col`, `message`          |
| Default (human)                                 | Grouped paths with `line:col  warning  message  manual-review/duplication`           |
| `scanDuplicateFunctionHints()` return value     | Structured `name`, `canonical`, etc. — for programmatic use only; not printed on CLI |

The **canonical** declaration (name and `path:line:col`) is embedded in **`message`** only. Extra TSV columns (e.g. `canonical_file`) are **not** planned: the only consumer is the bash hook, which displays the message for reviewers. Do not treat missing TSV columns as a bug in code review.

```mermaid
flowchart LR
  eslint[ESLint_Prettier]
  cq[check_code_quality_sh]
  manual[Human_checklist]
  hints[scan_manual_review_hints]
  precommit[precommit_hook]
  eslint --> precommit
  cq --> precommit
  hints --> precommit
  manual --> review[PR_review]
```

## Exceptions

Any intentional deviation should be:

1. Called out in a short comment with rationale
2. Reviewed in PR
3. Tracked as tech debt if temporary

Code should stay easy for humans to read and maintain.

## Hook: quick manual checklist

**Scripts** — `commit-quality-report.sh` [4]: staged `tests/**/*.ts` via explicit paths; `full-quality-report.sh` (`quality:report`) then `print-manual-review-hint.sh` [4] → **`scan-manual-review-hints.sh --full-tests`** for the full tree. All use **`scripts/scan-manual-review-hints.sh`** for heuristic pointers (`MANUAL:` comments, duplicate bodies, secrets/selectors), not a full semantic audit.

The **full human checklist** and **heuristic rule IDs** are documented under **Quality gates catalog → Manual review** above. Use the **qa-automation-quality** Claude skill for a **failure-only** report that includes **manual/review** gaps (in addition to automated output).
