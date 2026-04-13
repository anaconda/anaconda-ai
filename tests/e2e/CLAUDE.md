# Playwright TypeScript Project

## Project Overview

A Playwright TypeScript end-to-end testing framework for Web (Desktop & Mobile), API, and Electron apps. Built on top of `@anaconda/playwright-utils` which provides simplified utility functions for actions, assertions, locators, elements, page management, and API requests.

**Repository**: `anaconda/playwright-ts-utils`
**Default test target**: https://www.saucedemo.com (configurable via `URL` env var or `.env` file)
**AI Tools**: Claude Code and Cursor (skills and agents shared via `.claude/` and `.cursor/rules/`). In **Cursor IDE**, whenever you use **`@.claude/skills/`**, include **`@.cursor/`** in the same context—see [AI Skills and Agents](#ai-skills-and-agents).

## Project Structure

```
playwright-ts-template/
├── playwright.config.ts          # Playwright configuration (projects, timeouts, reporters)
├── test-setup/
│   ├── page-setup.ts             # Sets page context via setPage() before each test
│   ├── global-setup.ts           # Runs before all tests (initialization hooks)
│   └── global-teardown.ts        # Runs after all tests (cleanup hooks)
├── tests/
│   ├── specs/                    # Test spec files (*.spec.ts)
│   ├── pages/                    # Page Object classes (class-based POM)
│   ├── fixtures/
│   │   └── fixture.ts            # Custom Playwright fixtures for page objects
│   ├── testdata/                 # Test data files
│   └── storage-setup/            # Authentication storage state setup
├── .claude/
│   ├── skills/
│   │   ├── anaconda-playwright-utils/ # API docs, locator strategy, browser strategy, function references
│   │   └── playwright-cli/            # Browser automation CLI commands and references
│   └── agents/                        # Agent workflows (planner, generator, healer)
└── .cursor/rules/                 # Cursor rules referencing .claude/ skills and agents via @file
```

## Key Conventions

### Imports and Path Aliases

Use TypeScript path aliases defined in `tsconfig.json`:

- `@pages/*` -> `tests/pages/*`
- `@testdata/*` -> `tests/testdata/*`
- `@fixture` -> `tests/fixtures/fixture`
- `@playwright-config` -> `playwright.config`

### Page Setup

Always import `test` from `@fixture` instead of `@playwright/test`. This ensures `setPage(page)` is called before each test (required by all `@anaconda/playwright-utils` functions) and provides page object class instances as fixtures.

```typescript
import { test } from '@fixture';
```

The fixture setup (`tests/fixtures/fixture.ts`) extends the base test from `@anaconda/playwright-utils` (which sets `setPage(page)` per test) and registers page object classes as Playwright fixtures.

### Use @anaconda/playwright-utils Functions

Always prefer `@anaconda/playwright-utils` utility functions over raw Playwright API calls.

**Imports:** The package re-exports actions, assertions, locators, element helpers, page helpers, and **constants** (e.g. `STANDARD_TIMEOUT`, `NAVIGATION_TIMEOUT`) from its main entry. **Use one combined import from `@anaconda/playwright-utils`** for everything you need from that library—do not split the same symbols across multiple import lines or duplicate imports from the root package. Subpath imports (e.g. `@anaconda/playwright-utils/action-utils`) are optional (e.g. for tree-shaking); if you use them, avoid mixing redundant root + subpath imports for the same helpers. ESLint in this repo enforces **sorted named imports** (e.g. `sort-imports` may place uppercase exports like `STANDARD_TIMEOUT` before lowercase names—run `npm run lint:fix` if needed).

```typescript
// DO: Single barrel import; utility functions + constants as needed
import {
  STANDARD_TIMEOUT,
  click,
  clickAndNavigate,
  expectElementToBeVisible,
  fill,
  getLocatorByRole,
  getLocatorByTestId,
  getLocatorByText,
  gotoURL,
  logger, // Winston-based logger — use sparingly in page objects only, never in spec files
} from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { userData } from '@testdata/user-testdata';

await gotoURL(urlData.loginPageUrl);
await fill('#username', userData.username);
await clickAndNavigate(getLocatorByRole('button', { name: 'Login' }));
await expectElementToBeVisible('.dashboard', 'Dashboard should be visible after login');

// DON'T: Use raw Playwright API
await page.goto('https://example.com');
await page.locator('#username').fill('user');
await page.getByRole('button', { name: 'Login' }).click();
```

### Page Object Model (Class-based)

All page objects use class-based POM in `tests/pages/`. Classes use `@anaconda/playwright-utils` functions internally and are registered as Playwright fixtures in `tests/fixtures/fixture.ts`.

**Page object class** (`tests/pages/login-page.ts`):

```typescript
import {
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
import { userData } from '@testdata/user-testdata';

export class LoginPage {
  // Static selectors — plain strings for simple CSS/XPath
  private readonly usernameInput = '#user-name';
  // Dynamic locators — arrow functions for chained or compound locators
  private readonly passwordInput = () => getLocator('#password').or(getLocatorByPlaceholder('Password'));
  private readonly loginButton = () => getLocatorByRole('button', { name: 'Login' });

  async navigateToLoginPage(): Promise<void> {
    await gotoURL(urlData.loginPageUrl);
  }

  async loginWithValidCredentials(): Promise<void> {
    await fill(this.usernameInput, userData.username);
    await fill(this.passwordInput(), userData.pwd);
    await clickAndNavigate(this.loginButton());
    await expectElementToBeAttached(this.usernameInput, 'User should be logged in');
  }

  async verifyLoginPageIsDisplayed(): Promise<void> {
    await expectElementToBeVisible(this.usernameInput, 'Login page should be displayed');
  }
}
```

**Fixture registration** (`tests/fixtures/fixture.ts`):

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

**Spec file** (`tests/specs/products.spec.ts`):

```typescript
import { test } from '@fixture';

test.describe('Products page @smoke', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.navigateToLoginPage();
    await loginPage.loginWithValidCredentials();
  });

  test('should display products page', async ({ productsPage }) => {
    await productsPage.verifyProductsPageIsDisplayed();
  });

  test('should show product count', async ({ productsPage }) => {
    await productsPage.verifyProductCount(6);
  });
});
```

When creating new page objects, add them as fixtures in `tests/fixtures/fixture.ts`.

### Locator Strategy

Follow the 9-tier priority order in `.claude/skills/anaconda-playwright-utils/references/locators.md` (best to worst):

1. `data-qa-id` attributes (best) -> `getLocatorByTestId()` — **never** raw CSS `[data-qa-id="..."]` for a single element
2. `data-testid` (configured testIdAttribute) -> `getLocatorByTestId()` — CSS only for other `data-*` like `[data-product-id="..."]`
3. `id` attributes -> `#id` or `[id="..."]`
4. `name` attributes -> `[name="..."]`
5. XPath with unique attributes -> `//button[@aria-label="Submit"]`
6. CSS with unique attributes -> `button[aria-label="Submit"]`, `input[type="email"]`
7. Playwright built-in locators (only when no stable selector exists) -> `getLocatorByRole()`, `getLocatorByLabel()`, `getLocatorByPlaceholder()`, `getLocatorByText()`
8. XPath structural (fragile) -> `//div[@class="form"][2]//button`
9. CSS structural (fragile, last resort) -> `.form-group:nth-child(2) button`

**Mandatory upgrade rule:** If a DOM snapshot reveals a `data-qa-id` or any `data-*` attribute, always use it — never keep a role/text locator when a stable attribute exists.

**Multiple-element matches:** Action functions already filter hidden elements internally. If multiple visible elements still match, find a more specific locator using ancestor scoping — never use `.nth()`, `.first()`, or `.last()`:

```typescript
// ❌ Avoid — index breaks silently when the page changes
getLocatorByText('Pending').nth(2);

// ✅ Simple 1-level scoping — prefer getLocatorByTestId chaining
getLocatorByTestId('channel-list').locator('[data-qa-id="channel-item"]');
// ✅ Complex scoping (2+ levels or mixed types) — CSS compound is preferred
('[data-qa-id="channel-list"] [data-qa-id="pending-btn"]');
// ✅ XPath ancestor scope
('//tr[@data-qa-id="latest-row"]//button[@aria-label="Pending"]');
```

### Common Anti-patterns

```typescript
// ❌ Role/text locator when snapshot shows data-qa-id — always upgrade
private readonly signIn = () => getLocatorByRole('link', { name: 'Sign In' }); // ❌
private readonly signIn = () => getLocatorByTestId('sign-in-link');             // ✅

// ❌ Raw CSS for a single data-qa-id or data-testid element — always use getLocatorByTestId
private readonly releaseType = '[data-qa-id="release-type"]';   // ❌
private readonly submitBtn   = '[data-testid="submit-btn"]';    // ❌
private readonly releaseType = () => getLocatorByTestId('release-type'); // ✅
private readonly submitBtn   = () => getLocatorByTestId('submit-btn');   // ✅

// ❌ Omit error messages from assertions — always include a descriptive message
await expectElementToBeVisible(this.header());                                          // ❌
await expectElementToBeVisible(this.header(), 'Header should be visible after login');  // ✅

// ❌ waitForPageLoadState after clickAndNavigate — redundant, already done internally
await clickAndNavigate(this.loginButton());
await waitForPageLoadState({ waitUntil: 'load' }); // ❌ remove this line
```

### Action and Assertion Reference

- **Actions**: `.claude/skills/anaconda-playwright-utils/references/actions.md` - click, fill, select, check, hover, drag, upload, alerts
- **Assertions**: `.claude/skills/anaconda-playwright-utils/references/assertions.md` - visibility, text, value, attribute, page URL/title, soft assertions
- **Full API**: `.claude/skills/anaconda-playwright-utils/SKILL.md` - complete function signatures and CLI-to-library mapping

### Test Patterns

- **Always wrap tests in a `test.describe` block** — every spec file must have exactly one top-level `test.describe` containing all its tests
- Use `test.describe.configure({ mode: 'parallel' });` for parallel execution within a spec
- Use `test.beforeEach` for navigation setup
- Use tags like `@smoke`, `@reg` in describe/test names for filtering
- Use `clickAndNavigate()` when a click triggers page navigation; `click()` for AJAX/same-page actions

## Common Commands

```bash
# Run all tests
npm run test

# Run in chromium headless
npm run test:chromium -- <spec-file>

# Run in chromium headed (visible browser)
npm run test:chromium-headed -- <spec-file>

# Run specific test by name
npm run test:chromium-headed -- -g 'test name'

# Run smoke tests
npm run test:smoke

# Run with retries and workers
npm run test:chromium -- <spec-file> -j 3 --retries 2

# View HTML report
npm run report

# Lint
npm run lint
npm run lint:fix

# Format
npm run format

# UI mode
npm run ui

# Record tests with codegen
npm run record
```

## Configuration

- **Playwright config**: `playwright.config.ts` - projects: `setup`, `chromium` (headed), `chromiumheadless`
- **TypeScript**: `tsconfig.json` - strict mode, ES6 target, CommonJS modules
- **ESLint**: `eslint.config.js` - extends `@anaconda/playwright-utils/eslint` shared config (flat config format)
- **Husky**: Pre-commit hooks for lint-staged (ESLint + Prettier)
- **Timeouts**: Imported from `@anaconda/playwright-utils` (`TEST_TIMEOUT`, `EXPECT_TIMEOUT`, `ACTION_TIMEOUT`, `NAVIGATION_TIMEOUT`)

## AI Skills and Agents

This project includes AI skills and agent workflows in `.claude/` for automated test development. Cursor rules in `.cursor/rules/` reference the same skills and agents via `@file` directives, so both Claude Code and Cursor share the same knowledge base.

### Cursor IDE: always include `.cursor/` with `.claude/skills/`

When you use **Cursor** for test generation, refactors, or any task that relies on **skills under `.claude/`** (i.e. **`@.claude/skills/`** and every skill inside it), **always add `@.cursor/` in the same context**—for example **`@.cursor`**, **`@.cursor/rules/`**, or the relevant **`.mdc`** rule files. That pairing should be the default for **all** skills in **`@.claude/skills/`**, not only one folder.

**`.claude/skills/`** holds the detailed references and workflows for each skill; **`.cursor/rules/`** maps them onto _this_ repository (globs, conventions, `@file` links into the same skill docs, and project-wide guidance such as **`project.mdc`** → this **`CLAUDE.md`**). Using **skills plus Cursor rules together** typically yields better-aligned output than either alone.

**Install/update skills and agents**: `npx anaconda-pw-setup` (or `--skills` / `--agents` individually, `--force` to update, `--force-claude` to update CLAUDE.md)

### Skills (`.claude/skills/`)

- **anaconda-playwright-utils**: Complete API reference for all 6 utility modules. **Always reference this skill** when writing or modifying test code.
  - **SKILL.md** — Quick reference tables with all 103 functions, import patterns, example test, CLI-to-library mapping (38 common translations)
  - **references/actions.md** — Action functions for user interactions
    - Click actions: click, clickAndNavigate, doubleClick, clickByJS
    - Input actions: fill, fillAndEnter, fillAndTab, pressSequentially, clear, clearByJS
    - Selection: selectByValue, selectByText, selectByIndex
    - Other: check, uncheck, hover, focus, dragAndDrop, uploadFiles, downloadFile
    - Alerts: acceptAlert, dismissAlert, getAlertText
  - **references/assertions.md** — Assertion functions for test validation
    - Element state: expectElementToBeVisible, expectElementToBeHidden, expectElementToBeAttached, expectElementToBeChecked, expectElementToBeEnabled, expectElementToBeDisabled, expectElementToBeEditable
    - Text: expectElementToHaveText, expectElementToContainText
    - Value: expectElementToHaveValue, expectElementValueToBeEmpty
    - Attribute: expectElementToHaveAttribute, expectElementToContainAttribute
    - Count: expectElementToHaveCount
    - Page: expectPageToHaveURL, expectPageToContainURL, expectPageToHaveTitle
    - Soft assertions: Pass `{ soft: true }` option, call `assertAllSoftAssertions(testInfo)` at end
    - **Best practice:** Use assertions ONLY in page objects (verify\* methods), NOT in spec files
  - **references/locators.md** — Locator strategy and finding elements
    - **Locator priority** (9 tiers, best to worst): (1) data-qa-id → (2) data-testid/data-\* → (3) id → (4) name → (5) XPath with unique attrs → (6) CSS with unique attrs → (7) Playwright built-in locators (role/text) → (8) XPath structural → (9) CSS structural
    - Functions: getLocator, getVisibleLocator, getLocatorByTestId, getLocatorByText, getLocatorByRole, getLocatorByLabel, getLocatorByPlaceholder, getAllLocators
    - Frame functions: getFrame, getFrameLocator, getLocatorInFrame
    - **Key concept:** Visible by default (`onlyVisible: true`)
  - **references/element-utils.md** — Element data retrieval, state checks, and waits (16 functions)
    - **Data retrieval**: getText, getAllTexts, getInputValue, getAllInputValues, getAttribute, getLocatorCount
    - **Conditional checks** (return boolean): isElementAttached, isElementVisible, isElementHidden, isElementChecked
    - **Wait functions**: waitForElementToBeVisible, waitForElementToBeHidden, waitForElementToBeAttached, waitForElementToBeDetached, waitForElementToBeStable, waitForFirstElementToBeAttached
    - **Critical:** Use element-utils for data extraction and conditionals, NOT for assertions. Use assert-utils for assertions.
  - **references/api-utils.md** — API request functions for HTTP testing
    - Functions: getAPIRequestContext, getRequest, postRequest, putRequest, patchRequest, deleteRequest
    - Returns Playwright's APIResponse object (methods: ok(), status(), json(), text(), headers())
    - Uses page's request context (shares cookies/storage with browser)
    - Common patterns: authentication headers, JSON/form data, response validation
  - **references/page-utils.md** — Page management, navigation, and multi-tab handling
    - **Singleton pattern**: getPage, setPage, getContext (setPage called automatically by fixture)
    - **Multi-tab**: getAllPages, switchPage (1-based index), switchToDefaultPage, closePage
    - **Navigation**: gotoURL, getURL, waitForPageLoadState, reloadPage, goBack
    - **Utilities**: wait (use sparingly), getWindowSize, saveStorageState
  - **references/browser-strategy.md** — Token optimization strategy for page exploration
    - **Tier 1 (Lite):** WebFetch - 200-1000 tokens, static HTML reconnaissance
    - **Tier 2 (Snapshot):** playwright-cli snapshot - 500-2000 tokens, interactive discovery with DOM snapshots
    - **Tier 3 (Full Browser):** playwright-cli actions - 50-200 tokens per action, real browser interaction for selector capture
    - **Decision rules:** Start with Lite, escalate to Snapshot if dynamic content, escalate to Full if interactive exploration needed

- **playwright-cli**: Browser automation CLI for interactive page exploration, snapshots, form filling, screenshots, and debugging. Use `playwright-cli` commands to explore a page before writing tests.

**Skill usage rules:**

1. **Always reference `anaconda-playwright-utils` skill** when generating or modifying test code
2. **Use specific reference files** for detailed function signatures and usage patterns:
   - Writing actions → reference `actions.md`
   - Writing assertions → reference `assertions.md`
   - Finding elements → reference `locators.md`
   - Reading element data or conditionals → reference `element-utils.md`
   - API testing → reference `api-utils.md`
   - Navigation or multi-tab → reference `page-utils.md`
3. **Follow locator strategy** from `locators.md` (9-tier priority, prefer data-qa-id over role/text)
4. **Apply browser strategy** from `browser-strategy.md` when exploring pages (start with WebFetch, escalate as needed)
5. **Element-utils vs assert-utils distinction:**
   - element-utils → Data extraction, conditionals (if/while), variable assignments
   - assert-utils → Test assertions (expect\*), only in page objects not spec files
6. **Page-utils for multi-tab:** Use switchPage with 1-based index, always switch after opening new tabs
7. **API-utils for HTTP:** Shares page request context, use for API validation in UI tests
8. **CLI-to-library mapping:** Translate playwright-cli generated code using SKILL.md mapping table (38 common patterns)

### Agents (`.claude/agents/`)

All agents follow the browser strategy in `.claude/skills/anaconda-playwright-utils/references/browser-strategy.md` and use `@anaconda/playwright-utils` functions when writing test code.

- **playwright-test-planner**: Explores a web application (WebFetch first, then playwright-cli for interactive discovery) and creates comprehensive test plans in `specs/` directory with steps mapped to `@anaconda/playwright-utils` functions.
- **playwright-test-generator**: Generates Playwright test code from test plans or from a prompt/URL. Uses playwright-cli to capture real selectors, translates to `@anaconda/playwright-utils` functions. Generated tests should follow this project's class-based POM and fixture conventions.
- **playwright-test-healer**: Debugs and fixes failing Playwright tests. Runs tests, analyzes errors, uses playwright-cli for live debugging, and applies fixes using `@anaconda/playwright-utils` patterns.

### Workflow

1. **Plan**: Use the test planner agent to explore a URL and create a test plan
2. **Generate**: Use the test generator agent to create test code from the plan or from a URL
3. **Heal**: Use the test healer agent to fix any failing tests
