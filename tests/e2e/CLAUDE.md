# Playwright TypeScript Project

## Prerequisites

The Playwright agents (planner / generator / healer) shell out to `playwright-cli` for live browser interaction. It must be **globally installed** on each developer machine that uses the agents — run `npx anaconda-pw-setup` to install or verify it, or install manually: `npm install -g @playwright/cli`. Re-run setup after upgrading `@anaconda/playwright-utils` to check for a newer recommended version.

## Load This First (AI Assistants)

> **CRITICAL:** Before writing any test code, load `.claude/skills/anaconda-playwright-utils/SKILL.md`. This defines all 103 library functions, import patterns, CLI-to-library mappings, and example tests. For detailed function signatures, load the specific `references/*.md` files. In Cursor IDE, include `@.cursor/rules/` to map skills to this project.

## Project Structure

```
<project>/
├── playwright.config.ts          # Playwright configuration
├── tests/
│   ├── test-plans/               # Markdown test plans (planner output; not *.spec.ts)
│   ├── specs/                    # Test spec files (*.spec.ts)
│   ├── pages/                    # Page Object classes
│   ├── fixtures/
│   │   └── fixture.ts            # Custom fixtures for page objects
│   ├── testdata/                 # Test data files
│   └── storage-setup/            # Auth storage state setup
├── .claude/
│   ├── skills/
│   │   ├── anaconda-playwright-utils/  # Library API docs + references
│   │   └── playwright-cli/             # Browser automation CLI
│   └── agents/                         # Planner, generator, healer
└── .cursor/rules/                      # Cursor rules referencing skills
```

## Imports and Setup

### Path Aliases (tsconfig.json)

| Alias                | Resolves to                            |
| -------------------- | -------------------------------------- |
| `@pages/*`           | `tests/pages/*`                        |
| `@testdata/*`        | `tests/testdata/*`                     |
| `@fixture`           | `tests/fixtures/fixture` (single file) |
| `@playwright-config` | `playwright.config`                    |

### Singleton Page Pattern

Always import `test` from `@fixture`, never from `@playwright/test`. The fixture calls `setPage(page)` before every test — all library functions use this singleton internally. If `setPage` is not called, all library calls fail.

### Import Convention

One barrel import from `@anaconda/playwright-utils` for all utilities and constants. ESLint enforces sorted named imports.

```typescript
import {
  STANDARD_TIMEOUT,
  click,
  clickAndNavigate,
  expectElementToBeVisible,
  fill,
  getLocatorByTestId,
  gotoURL,
} from '@anaconda/playwright-utils';
```

### Config Files (project root)

- `playwright.config.ts` — spread `AnacondaConfigDefaults` and `AnacondaProjectDefaults` from `@anaconda/playwright-utils`
- `tsconfig.json` — strict mode, path aliases above
- `eslint.config.js` — extends `@anaconda/playwright-utils/eslint` (flat config)

## Page Object Model (3-File Pattern)

Every test uses three files: a **Page Object** class, a **Fixture** registration, and a **Spec** file. All actions and assertions live in the page object — specs only call page object methods.

### Page Object (`tests/pages/login-page.ts`)

```typescript
import {
  click,
  clickAndNavigate,
  expectElementToBeAttached,
  expectElementToBeVisible,
  fill,
  getLocator,
  getLocatorByPlaceholder,
  getLocatorByRole,
  gotoURL,
} from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { validUser, invalidUser } from '@testdata/user-testdata';

export class LoginPage {
  // Static selectors — raw CSS/XPath strings (no library call at class instantiation; tiers 3–6 + CSS compound scope)
  private readonly usernameInput = '#username';
  private readonly errorMessage = '[data-test="error-message"]';

  // Arrow functions — any library locator call (any tier); defers getPage() to test execution
  private readonly passwordInput = () => getLocator('#password').or(getLocatorByPlaceholder('Password'));
  private readonly loginButton = () => getLocatorByRole('button', { name: 'Login' });

  async navigateToLoginPage(): Promise<void> {
    await gotoURL(urlData.loginPageUrl);
  }

  async loginWithValidCredentials(username = validUser.username, password = validUser.pwd): Promise<void> {
    await fill(this.usernameInput, username);
    await fill(this.passwordInput(), password);
    await clickAndNavigate(this.loginButton());
  }

  async loginWithInvalidCredentials(username = invalidUser.username, password = invalidUser.pwd): Promise<void> {
    await fill(this.usernameInput, username);
    await fill(this.passwordInput(), password);
    await click(this.loginButton());
    await expectElementToBeVisible(this.errorMessage, 'Error message should appear for invalid credentials');
  }

  async verifyLoginSuccessful(): Promise<void> {
    await expectElementToBeAttached('[data-test="welcome"]', 'User should be logged in successfully');
  }

  async verifyLoginPageIsDisplayed(): Promise<void> {
    await expectElementToBeVisible(this.usernameInput, 'Login page should be displayed');
  }
}
```

### Fixture (`tests/fixtures/fixture.ts`)

```typescript
import { test as baseTest } from '@anaconda/playwright-utils';
import { LoginPage } from '@pages/login-page';
import { ProductsPage } from '@pages/products-page';

export const test = baseTest.extend<{
  loginPage: LoginPage;
  productsPage: ProductsPage;
}>({
  loginPage: async ({}, use) => {
    await use(new LoginPage());
  },
  productsPage: async ({}, use) => {
    await use(new ProductsPage());
  },
});
```

### Spec (`tests/specs/login.spec.ts`)

```typescript
import { test } from '@fixture';

test.describe('Login @smoke', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.navigateToLoginPage();
  });

  test('should login with valid credentials', async ({ loginPage }) => {
    await loginPage.loginWithValidCredentials();
    await loginPage.verifyLoginSuccessful();
  });

  test('should show error with invalid credentials', async ({ loginPage }) => {
    await loginPage.loginWithInvalidCredentials();
  });
});
```

### POM Rules

- **Spec files contain only page object method calls** — no `fill()`, `click()`, `expect*()`, or raw `expect()` in specs
- **Page objects own all actions and assertions** — action methods (verb+noun), `verify*` methods (assertions), `get*` methods (data retrieval)
- **Register every new page object** in `tests/fixtures/fixture.ts` before using it in specs
- **Wrap all tests** in a `test.describe` block with tags (`@smoke`, `@reg`)
- **Use `test.beforeEach`** for shared setup (navigation, login)
- **Store test data** in `tests/testdata/` — never hardcode values in page objects or specs

## Rules

### Library Usage

- **Always use `@anaconda/playwright-utils` functions** — never raw Playwright API (`page.click()`, `page.fill()`, `page.goto()`, `expect(locator)`)
- **`clickAndNavigate()`** for clicks that trigger page navigation; **`click()`** for same-page/AJAX actions
- **`fill()`** for inputs; **`pressSequentially()`** only for auto-search/autocomplete fields
- **Never add `waitForPageLoadState` after `clickAndNavigate`** — it already waits internally

### Assertions

- **Every assertion must include a descriptive error message** as the last argument
- **Hard assertions** (default) for critical checks; **soft assertions** (`{ soft: true }`) for non-critical — call `assertAllSoftAssertions(test.info())` immediately after each page object method that uses soft assertions

### Locators

- **Use `getLocatorByTestId()`** for `data-qa-id` for a **single standalone element** — never raw CSS `[data-qa-id="..."]` for a single element. CSS compound strings with `data-qa-id` ancestors are valid and preferred at 2+ ancestor levels (see `references/locators.md`). (`getLocatorByTestId()` targets the configured `testIdAttribute` — Anaconda projects set `use.testIdAttribute = 'data-qa-id'` in `playwright.config.ts`; any other `data-*` attribute must use a CSS selector instead.)
- **Always upgrade locators** — if a DOM snapshot reveals a `data-qa-id` or `data-*` attribute, use it instead of role/text locators
- **Never use `.nth()`, `.first()`, `.last()`** — disambiguate with ancestor scoping instead (see `references/locators.md`)

### Code Quality

- **No `console.log`** — use `logger` from `@anaconda/playwright-utils` in page objects only, never in specs
- **Import `test` from `@fixture`** — never from `@playwright/test`

## Locator Priority (9-Tier)

1. `data-qa-id` attributes (best) -> `getLocatorByTestId()` (`use.testIdAttribute = 'data-qa-id'` is configured in Anaconda projects)
2. Other `data-*` attributes (e.g. `data-testid`, `data-test`) -> CSS selector `[data-testid="..."]`
3. `id` attributes -> `#id`
4. `name` attributes -> `[name="..."]`
5. XPath with unique attributes -> `//button[@aria-label="Submit"]`
6. CSS with unique attributes -> `button[aria-label="Submit"]`
7. Playwright built-in (only when no stable selector) -> `getLocatorByRole()`, `getLocatorByLabel()`, `getLocatorByText()`
8. XPath structural (fragile) -> `//div[@class="form"][2]//button`
9. CSS structural (last resort) -> `.form-group:nth-child(2) button`

Full guide with scoping patterns: `.claude/skills/anaconda-playwright-utils/references/locators.md`

## Commands

```bash
npx playwright test                              # Run all tests
npx playwright test <spec-file>                  # Run specific file
npx playwright test --grep @smoke                # Run by tag
npx playwright test -g 'login'                   # Run by pattern
npx playwright test --project=chromium           # Run on specific browser
npx playwright test <spec-file> -j 3 --retries 2 # Parallel workers + retries
npx playwright test --ui                         # Open Playwright Inspector
npx playwright show-report                       # View HTML report
```

### Optional quality scripts (when defined in `package.json`)

Some projects (including library maintenance repos that mirror `@anaconda/playwright-utils` tooling) define:

| Script                       | Consumer wires (bin)                             | Purpose                                                                                                           |
| ---------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `format`, `lint`, `lint:fix` | Your Prettier/ESLint commands (not from package) | Required by `quality:full` / `quality:report`                                                                     |
| `check:code-quality`         | `playwright-utils-check-code-quality`            | Code-quality from install root                                                                                    |
| `check:code-quality:staged`  | `playwright-utils-check-code-quality --staged`   | Optional: staged QA paths only                                                                                    |
| `quality:full`               | `playwright-utils-quality-full`                  | Full-repo compact gate; needs `format`, `lint:fix`, `lint`                                                        |
| `quality:report`             | `playwright-utils-full-quality-report`           | Full-repo report [1]–[4]; [4] `manual-review/secrets` skips `tests/testdata/**/*.ts`; exit **0**/**1** on [1]–[3] |
| `precommit`                  | `playwright-utils-precommit`                     | QA-scoped Husky hook; do not split into lint-staged + commit-quality-report                                       |
| (ad-hoc)                     | `playwright-utils-print-manual-review-hint`      | Optional manual hints; [4] is already in `quality:report`                                                         |

**Consumers:** wire bins as above—do not use `bash ./scripts/*.sh` in your `package.json`. **Maintainers of `@anaconda/playwright-utils` itself** use `bash ./scripts/*.sh` in that repo (own bins are not in `node_modules/.bin/` at the package root). See README § Code quality checks.

If your project does not list these scripts, ignore this table. When the **qa-automation-quality** skill is installed, see its `references/qa-automation-guidelines.md` for detail.

### Prettier-safe markdown patterns

`npm run format` runs Prettier on all `**/*.md` files. Two patterns cause Prettier to escape `*`, corrupting the markdown — always use the safe form:

```text
# BAD: glob inside bold+backtick — Prettier escapes the inner *
**`path/**/*.ts`**

# GOOD: plain backticks only
`path/**/*.ts`
```

```text
# BAD: bold ending immediately before colon — Prettier escapes the closing **
**text**:

# GOOD: swap colon for em-dash, or restructure
**text** —
```

After editing any `.md` file, run `npm run format` and verify it shows `(unchanged)`.

## Skills and Agents

Installed via `npx anaconda-pw-setup` (flags: `--skills`, `--agents`, `--force`, `--force-claude`).

### Skills (`.claude/skills/`)

| Skill                                | When to load                                               |
| ------------------------------------ | ---------------------------------------------------------- |
| `anaconda-playwright-utils/SKILL.md` | **Always first** — all 103 functions, imports, CLI mapping |
| `references/actions.md`              | Click, fill, select, drag, upload, keyboard, alerts        |
| `references/assertions.md`           | All `expect*` assertion functions                          |
| `references/locators.md`             | Locator strategy, 9-tier priority, frames                  |
| `references/element-utils.md`        | Element data retrieval, state checks, waits                |
| `references/api-utils.md`            | API/HTTP request testing                                   |
| `references/page-utils.md`           | Navigation, multi-tab, page state                          |
| `references/browser-strategy.md`     | Token-efficient page exploration (Lite/Snapshot/Full)      |
| `playwright-cli/SKILL.md`            | Live browser interaction for selector capture              |

### Agents (`.claude/agents/`)

| Agent                       | Purpose                                                             |
| --------------------------- | ------------------------------------------------------------------- |
| `playwright-test-planner`   | Explores a URL and produces a test plan mapped to library functions |
| `playwright-test-generator` | Generates test code (POM + fixture + spec) from a plan or URL       |
| `playwright-test-healer`    | Debugs and fixes failing tests using live browser inspection        |

**Workflow:** Plan -> Generate -> Heal
