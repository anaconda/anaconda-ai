# Test Generation

Generate Playwright test code automatically as you interact with the browser.

## How It Works

Every action you perform with `playwright-cli` generates corresponding Playwright TypeScript code.
This code appears in the output and can be copied directly into your test files.

## Example Workflow

```bash
# Start a session
playwright-cli open https://example.com/login

# Take a snapshot to see elements
playwright-cli snapshot
# Output shows: e1 [textbox "Email"], e2 [textbox "Password"], e3 [button "Sign In"]

# Fill form fields - generates code automatically
playwright-cli fill e1 "user@example.com"
# Ran Playwright code:
# await page.getByRole('textbox', { name: 'Email' }).fill('user@example.com');

playwright-cli fill e2 "password123"
# Ran Playwright code:
# await page.getByRole('textbox', { name: 'Password' }).fill('password123');

playwright-cli click e3
# Ran Playwright code:
# await page.getByRole('button', { name: 'Sign In' }).click();
```

## Building a Test File

The CLI generates raw Playwright code. Translate each line to `@anaconda/playwright-utils` functions and organize everything into the project's class-based Page Object Model. The five steps below follow a single sign-in flow end-to-end.

### Step 1 — Capture actions with the CLI

Interact with the page; the CLI records every action as raw Playwright code.

```bash
playwright-cli open https://example.com
playwright-cli click e3   # clicks "Sign In" link
# Ran Playwright code:
# await page.getByRole('link', { name: 'Sign In' }).click();

playwright-cli snapshot   # check the sign-in page that just loaded
# [region, data-qa-id="sign-in-header"] Sign In
```

**Raw code generated:**

```typescript
await page.goto('https://example.com');
await page.getByRole('link', { name: 'Sign In' }).click(); // role locator — fragile
await expect(page.getByRole('heading')).toContainText('Sign In'); // text assertion — fragile
```

### Step 2 — Upgrade locators using the DOM snapshot

The CLI defaults to role/text locators (tier 7). **This upgrade is mandatory, not optional** — if the snapshot reveals a `data-qa-id` or any `data-*` attribute, use it. Do not keep the CLI-generated role/text locator when a stable attribute exists.

```
Snapshot reveals:
  [link, data-qa-id="sign-in-link"]   → getLocatorByTestId('sign-in-link')   tier 1  ← use this
  [region, data-qa-id="sign-in-header"] → getLocatorByTestId('sign-in-header') tier 1  ← use this
  [input, id="username"]              → '#username'                           tier 3  ← use this
  [input, name="password"]            → '[name="password"]'                   tier 4  ← use this
  (no stable attr on button)          → getLocatorByRole('button', ...)       tier 7  ← only now
```

Locator field conventions inside a class (full 9-tier priority in `references/locators.md`):

- **Static** — `private readonly field = 'css-or-xpath'` — plain string for tiers 3–6 (`#id`, `[name="x"]`, XPath/CSS with stable attrs). Passed directly to utility calls.
- **Dynamic** — `private readonly field = () => getLocatorFn(...)` — arrow function for tiers 1–2 (`getLocatorByTestId`), tier 7 built-ins, or `.or()` compound locators. Arrow functions are lazy — they resolve after `setPage()` is called.
- **Never** `.first()` / `.last()` / `.nth(N)` — see below.

#### When the same locator matches multiple elements

Action functions already enforce `onlyVisible: true` — hidden duplicates are filtered automatically. If multiple **visible** elements still match, find a more specific locator:

1. Look for a `data-qa-id` / `id` / `data-*` on the target or a stable ancestor, then scope:
   ```typescript
   // Instead of: getLocatorByText('Pending').nth(2)
   '[data-qa-id="channel-list"] [data-qa-id="pending-btn"]'; // CSS ancestor scope
   '//tr[@data-qa-id="latest-row"]//button[@aria-label="Pending"]'; // XPath ancestor scope
   ```
2. If no ancestor attribute exists, write a custom XPath using structural context + stable attributes
3. `.nth()` / `.first()` / `.last()` are last resort only — add a comment if you must use one

### Step 3 — Create page object classes

Page objects live in `tests/pages/`. Use destructured imports from `@anaconda/playwright-utils` — never namespace objects (`ActionUtils.xxx`) or raw Playwright API calls.

```typescript
// tests/pages/home-page.ts
import { clickAndNavigate, getLocatorByTestId, gotoURL } from '@anaconda/playwright-utils';

export class HomePage {
  // CLI: page.getByRole('link', { name: 'Sign In' }) — snapshot showed data-qa-id instead
  private readonly signInButton = () => getLocatorByTestId('sign-in-link'); // tier 1

  async goToHomePage(): Promise<void> {
    await gotoURL('https://example.com');
  }

  async clickSignIn(): Promise<void> {
    // clickAndNavigate (not click) — this link loads a new page
    await clickAndNavigate(this.signInButton());
  }
}
```

```typescript
// tests/pages/sign-in-page.ts
import { expectElementToBeVisible, getLocatorByTestId } from '@anaconda/playwright-utils';

export class SignInPage {
  // CLI: page.getByRole('heading') — snapshot showed data-qa-id instead
  private readonly pageHeader = () => getLocatorByTestId('sign-in-header'); // tier 1

  async verifySignInPageLoaded(): Promise<void> {
    // expectElementToBeVisible (not text assertion) — verifies the page loaded, not copy
    await expectElementToBeVisible(this.pageHeader(), 'Sign-in page header should be visible after navigation');
  }
}
```

### Step 4 — Register classes in the fixture file

Extend the base `test` fixture so every spec receives page object instances automatically. The base fixture also calls `setPage(page)` before each test, which is required by all `@anaconda/playwright-utils` functions.

```typescript
// tests/fixtures/fixture.ts
import { test as baseTest } from '@anaconda/playwright-utils';
import { HomePage } from '@pages/home-page';
import { SignInPage } from '@pages/sign-in-page';

export const test = baseTest.extend<{
  homePage: HomePage;
  signInPage: SignInPage;
}>({
  homePage: async ({}, use) => {
    await use(new HomePage());
  },
  signInPage: async ({}, use) => {
    await use(new SignInPage());
  },
});

export const expect = test.expect;
```

### Step 5 — Write the spec file

Import `test` from `@fixture` (not `@playwright/test`). Destructure page object fixtures. Spec files only call page object methods — no actions or assertions directly in specs.

```typescript
// tests/specs/sign-in.spec.ts
import { test } from '@fixture';

test.describe('Sign-in flow @smoke', () => {
  test('goes to sign-in page correctly', async ({ homePage, signInPage }) => {
    await homePage.goToHomePage();
    await homePage.clickSignIn();
    await signInPage.verifySignInPageLoaded();
  });
});
```

See the full CLI-to-library mapping table in `.claude/skills/anaconda-playwright-utils/SKILL.md`.

## Best Practices

### 1. Prefer Stable Locators Over Role/Text

The CLI often generates role- or text-based locators. **Replace them with stable selectors** (tiers 1–6) before adding to a page object. Role/text locators (tier 7) are a last resort — they break when copy or locale changes.

```typescript
// CLI generates (tier 7 — fragile, text can change)
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByText('Order complete').click();

// Translate to stable selectors first (tiers 1–3 preferred)
private readonly submitButton = () => getLocatorByTestId('submit-order');  // tier 1 — data-qa-id
private readonly successMessage = '[data-test="complete-header"]';         // tier 2 — data-*
private readonly usernameInput = '#user-name';                             // tier 3 — id
```

See the full 9-tier priority in `references/locators.md`.

### 2. Explore Before Recording

Take snapshots to understand the page structure before recording actions:

```bash
playwright-cli open https://example.com
playwright-cli snapshot
# Review the element structure
playwright-cli click e5
```

### 3. Choose the correct Click Action

The CLI always generates `page.locator(...).click()`. When translating, pick the correct library function based on whether the click triggers a page navigation:

| Scenario                                                         | Use                  |
| ---------------------------------------------------------------- | -------------------- |
| Click opens a dropdown, modal, or accordion (same page)          | `click()`            |
| Click adds to cart, submits AJAX form, toggles state (same page) | `click()`            |
| Click navigates to a new page (link, form redirect)              | `clickAndNavigate()` |

```typescript
// CLI generates (always .click() — doesn't know if navigation follows)
await page.locator('#add-to-cart').click();
await page.locator('#login-button').click();

// Translate: ask "does this load a new page?"
await click(this.addToCartButton()); // no — stays on product page
await clickAndNavigate(this.loginButton()); // yes — redirects to dashboard
```

Using `click()` when navigation follows will cause the next action to run before the new page has loaded. `clickAndNavigate()` waits for three things in order: the `framenavigated` event, the page load state, and the clicked element becoming stale/hidden — confirming the old page is fully gone before the test continues.

See full signatures in `references/actions.md`.

### 4. Add Assertions Manually

Generated code captures actions but not assertions. Add expectations in `verify*` methods inside the page object using stable selectors and a descriptive error message:

```typescript
// Generated action (translated to library)
await clickAndNavigate(this.submitButton());

// Add assertion in page object — stable selector + error message
await expectElementToBeVisible(this.successMessage, 'Success message should be visible after form submission');
```
