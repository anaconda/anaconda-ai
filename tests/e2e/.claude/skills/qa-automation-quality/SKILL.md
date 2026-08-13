---
name: qa-automation-quality
description: Run full repo quality gates for Playwright TypeScript QA automation—consumer package.json maps npm scripts to playwright-utils-* bins (check:code-quality, quality:full, quality:report, precommit; optional staged/print-manual-review-hint). Do not bash ./scripts/*.sh or split precommit into lint-staged+commit-quality-report. Resolves CONSUMER_ROOT via resolve-consumer-root.sh (not git root). QA-scoped pre-commit skips dev-only commits. Use before commit or PR, CI failures, repo standards, ESLint/Prettier, test.skip/TODO/JSDoc, or "run all checks."
---

# QA automation quality

Orchestrate the repo’s quality workflow: **automated commands first**, then **manual/review** using the reference guidelines, then a **failure-only** report.

## Automated gates (run first)

From the **QA install root** (where `@anaconda/playwright-utils` is installed — e.g. `functional_tests/`, `e2e/`, or repo root for flat projects). Bins use **`scripts/resolve-consumer-root.sh`** to set **`CONSUMER_ROOT`**, not git root.

```bash
npm run quality:full
```

- **`npm run format`** — Prettier `--write` on `src/`, `tests/`, `example/`, config, JSON, and Markdown. When editing `.md` files, follow Prettier-safe patterns (see CLAUDE.md → Prettier-safe markdown patterns) to avoid `*` escaping: use plain backticks for glob paths, and avoid bold text ending immediately before a colon.
- **`npm run lint:fix`** then **`npm run lint`** — ESLint auto-fix then verify on `src/`, `tests/`, `example/`, `playwright.config.ts` (includes Prettier via `prettier/prettier` where configured).
- **`npm run check:code-quality`** — bin **`playwright-utils-check-code-quality`**: file length, `test.skip` justification, TODO + ticket, JSDoc complexity (**warning**, non-blocking; internal `check-jsdoc-complexity.js`—not consumer-wired).

**Consumer bin matrix and anti-patterns:** see package README § Code quality checks (all `playwright-utils-*` bins, consumer-owned `format`/`lint`/`lint:fix`, package-internal scripts).

**Onboarding / enumerated check lists:** see `references/qa-automation-guidelines.md` → **Quality gates catalog** (ESLint groupings, `code-quality/*` and `manual-review/*` rule IDs, human checklist vs heuristics).

**For the assistant using this skill:** There is no separate script for “manual” checks—you (or the user) read `references/qa-automation-guidelines.md` and apply judgment. **If `npm run lint` or `npm run check:code-quality` fails, lead the failure-only report with those errors** and do not imply the overall quality pass is done. Only after automated steps succeed is it appropriate to walk the manual/review bullets and report any gaps there. **`npm run quality:report`** always prints heuristic section **[4]** on the full tests tree under **CONSUMER_ROOT** even when [1]–[3] fail—use that output for duplication/secrets/selector pointers while fixing blocking errors (`manual-review/secrets` skips `tests/testdata/**/*.ts`, same as ESLint).

**Consumer install root:** Quality bins resolve the QA npm project from `node_modules/@anaconda/playwright-utils` (not git root). Use Husky and **`npm run precommit`** in the QA folder (`functional_tests/`, `e2e/`, etc.). See **`references/qa-automation-guidelines.md` → Before you commit**.

**Pre-commit (Husky):** use **`"precommit": "playwright-utils-precommit"`** in the QA **`package.json`** (Husky hook in that folder):

1. **Gate** — if no staged files under the QA install root, the hook **exits immediately** (no lint-staged, no report; dev-only commits are untouched).
2. **`lint-staged`** — runs from **`CONSUMER_ROOT`** only when QA files are staged; globs are relative to that folder (`tests/**/*.ts`, …).
3. **`playwright-utils-commit-quality-report`** — unified report on **staged QA files only**:
   - **Prettier** — `--check` on staged QA `*.json` / `*.md` / `*.ts`.
   - **ESLint** — verify staged QA `*.ts`.
   - **check-code-quality** — `--staged` under `CONSUMER_ROOT`.
   - **Manual review** — staged `tests/**/*.ts` (heuristic; `manual-review/secrets` skips `tests/testdata/**/*.ts`; see guidelines).

The hook exits **1** if lint-staged fails or if any automated section in the report fails; fix issues in **any order** from the printed sections. For **whole-repo** formatting/lint fixes without going through the hook, use `npm run format` or `npm run lint:fix`.

**Same pipeline without committing:** `npm run precommit` (stage files first; does not create a commit).

**Full repo before push:** `npm run quality:full` or **`playwright-utils-quality-full`** from the **QA install root** — `format` + `lint:fix` + `lint` + code-quality (writes files; no `lint-staged`).

**Full repo + manual hint:** `npm run quality:report` or **`playwright-utils-full-quality-report`**: steps [1]–[3] from `CONSUMER_ROOT`, then [4] on the full tests tree under **CONSUMER_ROOT** (**always**, even when [1]–[3] fail). Section [4] `manual-review/secrets` skips `tests/testdata/**/*.ts` (same as ESLint). Exit **0** if [1]–[3] pass; exit **1** otherwise.

**Semantic manual gaps:** only an assistant can compare code to `references/qa-automation-guidelines.md` fully. Use this skill for a **failure-only** report including manual/review where needed.

Skipped with:

```bash
git commit --no-verify
# or
git commit -n
```

## ESLint config layering

- **`eslint.config.base.mjs`** — Shared/published rules (`@anaconda/playwright-utils/eslint`).
- **`eslint.config.mjs`** — This repo: spreads `...base`, then adds repo-specific blocks. In flat config, **later** objects **override** the same rule keys from earlier objects.

## Manual / review (after automated gates pass)

For in-scope files, walk the checklist in:

@file references/qa-automation-guidelines.md

Focus on items **not** fully enforced by ESLint/scripts (POM structure, duplicate flows, selector habits, secrets, expect messages, etc.).

## Final report (failure-only)

Output **only** problems—do **not** list passing checks or per-section “Pass.”

| Section                    | Include                                                                          |
| -------------------------- | -------------------------------------------------------------------------------- |
| **ESLint / Prettier**      | Issues from `npm run lint` (file:line, rule/message). **Omit** if lint exited 0. |
| **check-code-quality**     | Issues from `npm run check:code-quality`. **Omit** if clean.                     |
| **Manual / review**        | Gaps vs `qa-automation-guidelines.md`. **Omit** if none.                         |
| **Exceptions / tech debt** | Only if calling out **intentional** deviations (short bullets).                  |

**After automated checks already passed:** In that situation there are **no automated failures**, so the **ESLint / Prettier** and **check-code-quality** rows are **empty—omit those sections entirely.** The report may only contain **Manual / review** (and optionally **Exceptions**). If manual review finds no gaps either, the report body is empty or a single short all-clear line.

**When automated checks failed** (e.g. `quality:full` or `quality:report` with failing [1]–[3], or `--no-verify`): include automated failures; for `quality:report`, heuristic [4] is already in the terminal—reference it when relevant.

If everything is clean across automated and manual, use **no body** or at most **one** short all-clear line—no enumerated success list.
