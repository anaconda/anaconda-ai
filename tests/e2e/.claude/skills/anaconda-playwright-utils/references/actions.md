# Action Utils Reference

Source: `src/playwright-utils/utils/action-utils.ts`

## Overview

Action utils provide functions for interacting with page elements. All action functions:

- Accept `string | Locator` as the first `input` parameter
- Enforce visibility by default (`onlyVisible: true`)
- Support `stable: true` option to wait for element position stability
- Return `Promise<void>` unless noted otherwise

**Important:**

- ✅ All actions go in page object methods, not spec files
- ✅ Never use `console.log()` or `console.error()` in tests
- ✅ If logging is needed, use `logger` from `@anaconda/playwright-utils` minimally in page objects only
- ✅ Spec files should call page object methods, never call actions directly

## Click Actions

### `click(input, options?: ClickOptions)`

Clicks on a visible element. Supports all Playwright click options plus `onlyVisible` and `stable`.

**Use this for:** AJAX calls, same-page interactions, dropdown toggles, modal opens — anything that doesn't navigate to a new page.

```typescript
await click('#add-to-cart-button'); // AJAX call, stays on same page
await click('.menu-toggle'); // Opens dropdown on same page
```

### `clickAndNavigate(input, options?: ClickOptions)`

Clicks and waits for page navigation (framenavigated event + load state). **Use this instead of `click` when the click triggers a full page navigation.**

**Use this for:** Links that navigate to different pages, form submissions that redirect, any click that loads a new page.

```typescript
await clickAndNavigate('a[href="/checkout"]'); // Navigate to checkout page
await clickAndNavigate('#login-button'); // Navigate to dashboard after login
```

**When to use which:**

- **`click()`** = No page navigation (stays on same page)
- **`clickAndNavigate()`** = Page navigation occurs (new page loads)

**`clickAndNavigate` already handles all post-navigation waiting** — it waits for the `framenavigated` event, the page load state, and the clicked element to go stale/hidden. Never add `waitForPageLoadState` after it:

```typescript
// ❌ Redundant — waitForPageLoadState is already done internally
await clickAndNavigate(this.loginButton());
await waitForPageLoadState({ waitUntil: 'load' });

// ✅ Correct
await clickAndNavigate(this.loginButton());
```

### `doubleClick(input, options?: DoubleClickOptions)`

Double-clicks on a visible element.

```typescript
await doubleClick('.text-field'); // Select all text
```

### `clickByJS(input, options?: TimeoutOption)`

Clicks using JavaScript `el.click()`. Bypasses visibility checks. Use only when standard click fails.

```typescript
// Only use if click() fails due to overlapping elements or CSS issues
await clickByJS('.hidden-button', { timeout: SMALL_TIMEOUT });
```

## Input Actions

### `fill(input, value: string, options?: FillOptions)`

Fills an input field, replacing existing content. **Use this as the default for all form field filling.**

```typescript
await fill('#username', userData.userName);
await fill('#password', userData.pwd);
```

### `fillAndEnter(input, value: string, options?: FillOptions)`

Fills an input field then presses Enter. Use for search fields and forms that submit on Enter.

```typescript
await fillAndEnter('#search-input', searchData.query); // Fills and presses Enter
```

### `fillAndTab(input, value: string, options?: FillOptions)`

Fills an input field then presses Tab. Use to move to the next field.

```typescript
await fillAndTab('#first-name', userData.firstName); // Fills and moves to next field
```

### `pressSequentially(input, value: string, options?: PressSequentiallyOptions)`

Types value character by character, simulating keyboard press events. **Use this only when testing features that require character-by-character typing** like auto-search, autocomplete, or autofill.

**Use `fill()` by default, use `pressSequentially()` only for:**

- Auto-search suggestions (need to trigger on each character)
- Autocomplete features (need intermediate results)
- Autofill testing (need to see progressive changes)

```typescript
// Default: Use fill() for normal form filling
await fill('#email', userData.email);

// Special case: Use pressSequentially() for auto-search
await pressSequentially('#search', searchData.query); // Triggers API calls per character
// Server responds to 'j', then 'ja', then 'jav', then 'java'
```

### `clear(input, options?: ClearOptions)`

Clears an input field.

```typescript
await clear('#input-field'); // Clears existing content
```

### `clearByJS(input, options?: TimeoutOption)`

Clears input using JavaScript, dispatches `input` and `change` events. Use only when `clear()` doesn't work.

```typescript
// Only if clear() fails to trigger change events
await clearByJS('.special-input', { timeout: SMALL_TIMEOUT });
```

## Keyboard Actions

### `pressPageKeyboard(key: string, options?: PressSequentiallyOptions)`

Presses a key on the page (not on a specific element). Example: `pressPageKeyboard('Escape')`.

### `pressLocatorKeyboard(input, key: string, options?: PressSequentiallyOptions)`

Presses a key on a specific element. Example: `pressLocatorKeyboard('#search', 'Enter')`.

## Selection Actions

### `selectByValue(input, value: string, options?: SelectOptions)`

Selects a dropdown option by its `value` attribute.

### `selectByValues(input, values: string[], options?: SelectOptions)`

Multi-select by `value` attributes.

### `selectByText(input, text: string, options?: SelectOptions)`

Selects a dropdown option by its visible text (label).

### `selectByIndex(input, index: number, options?: SelectOptions)`

Selects a dropdown option by its zero-based index.

## Checkbox/Radio

### `check(input, options?: CheckOptions)`

Checks a checkbox or radio button. **Use this as the default for checkboxes/radios.**

```typescript
await check('#agree-terms-checkbox');
await check('input[name="subscription"]'); // Select radio option
```

### `uncheck(input, options?: CheckOptions)`

Unchecks a checkbox. **Use this to deselect checkboxes.**

```typescript
await uncheck('#optional-newsletter');
```

**When `check()/uncheck()` are unreliable:** If you find these functions behave inconsistently, use `click()` as a fallback. This allows you to click the checkbox like a user would:

```typescript
// If check() is unreliable:
await click('#checkbox-label'); // Click the label or checkbox directly
```

## Mouse Actions

### `hover(input, options?: HoverOptions)`

Hovers over an element.

### `focus(input, options?)`

Focuses on an element.

### `dragAndDrop(input, dest, options?: DragOptions)`

Drags element `input` and drops on element `dest`. Both accept `string | Locator`.

## File Actions

### `downloadFile(input, savePath: string, options?: ClickOptions): Promise<string>`

Clicks the element to trigger a download, saves to `savePath`, returns the suggested filename. Works with remote browsers.

### `uploadFiles(input, path: UploadValues, options?: UploadOptions)`

Uploads files to a file input element.

## Other

### `scrollLocatorIntoView(input, options?: TimeoutOption)`

Scrolls the element into view if it is not currently visible in the viewport.

**Parameters:**

- `input` — CSS/XPath selector string or Locator
- `options.timeout` — Maximum wait time (default: ACTION_TIMEOUT)

**When to use:** Use before interacting with elements that are outside the viewport (e.g., elements far down a long page, or inside a scrollable container). Most action functions scroll automatically, but `scrollLocatorIntoView` is useful when you need to scroll without performing any other action — for example, to trigger lazy-loaded content or verify visual presence in a scrollable area.

**Example:**

```typescript
// Scroll a footer element into view before asserting visibility
await scrollLocatorIntoView('[data-qa-id="footer-links"]');
await expectElementToBeVisible('[data-qa-id="footer-links"]', 'Footer links should be visible after scroll');

// In a page object
async verifyLazyLoadedContent(): Promise<void> {
  await scrollLocatorIntoView(this.lazySection());
  await expectElementToBeVisible(this.lazySection(), 'Lazy section should load after scrolling into view');
}
```

## Alert/Dialog Actions

### `acceptAlert(input, options?): Promise<string>`

Clicks element, accepts the resulting dialog (optionally with prompt text), returns dialog message.
Options: `{ promptText?: string, timeout?: number }`

### `dismissAlert(input, options?): Promise<string>`

Clicks element, dismisses the resulting dialog, returns dialog message.

### `getAlertText(input, options?): Promise<string>`

Clicks element, gets dialog message text, dismisses it, returns the message.

## Page Object Model Examples

### Login Form Example

**Page Object (tests/pages/login-page.ts):**

```typescript
import {
  gotoURL,
  fill,
  fillAndEnter,
  click,
  clickAndNavigate,
  expectElementToBeVisible,
  expectPageToHaveURL,
} from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { adminUserData, nonAdminUserData } from '@testdata/user-testdata';

export class LoginPage {
  private readonly usernameField = '#username';
  private readonly passwordField = '#password';
  private readonly loginButton = '#login-button';
  private readonly errorMessage = '.error-message';

  async navigateToLogin() {
    await gotoURL(urlData.loginPageUrl);
    await expectElementToBeVisible(this.usernameField, {
      message: 'Login page should display username field',
    });
  }

  async login(username: string, password: string) {
    await fill(this.usernameField, username);
    await fill(this.passwordField, password);
    await clickAndNavigate(this.loginButton);
    await expectPageToHaveURL(/dashboard/, {
      message: `User should be redirected to dashboard after login`,
    });
  }

  async loginWithEnter(username: string, password: string) {
    await fill(this.usernameField, username);
    // Use fillAndEnter to submit on password field
    await fillAndEnter(this.passwordField, password);
    await expectPageToHaveURL(/dashboard/, {
      message: 'Dashboard should load after entering password with Enter key',
    });
  }

  async verifyErrorMessage(expectedError: string) {
    await expectElementToBeVisible(this.errorMessage, {
      message: `Error message should be displayed: ${expectedError}`,
    });
  }

  async loginAsAdmin() {
    await this.login(adminUserData.email, adminUserData.pwd);
  }

  async loginAsUser() {
    await this.login(nonAdminUserData.email, nonAdminUserData.pwd);
  }
}
```

**Fixture (tests/fixtures/fixture.ts):**

```typescript
import { test as base } from '@anaconda/playwright-utils';
import { LoginPage } from '@pages/login-page';
import { userData } from '@testdata/user-testdata';

export const test = base.extend({
  loginPage: async ({}, use) => {
    await use(new LoginPage());
  },
});
```

**Spec File (tests/specs/login.spec.ts):**

```typescript
import { test } from '@fixture';

test.describe('Login @smoke', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.navigateToLogin();
  });

  test('should login with valid credentials', async ({ loginPage }) => {
    await loginPage.login(userData.email, userData.pwd);
  });

  test('should login as admin', async ({ loginPage }) => {
    await loginPage.loginAsAdmin();
  });

  test('should login with enter key', async ({ loginPage }) => {
    await loginPage.loginWithEnter(userData.email, userData.pwd);
  });
});
```

### Checkout Form Example

**Page Object (tests/pages/checkout-page.ts):**

```typescript
import {
  fill,
  fillAndTab,
  selectByValue,
  check,
  click,
  clickAndNavigate,
  expectElementToBeVisible,
} from '@anaconda/playwright-utils';

export class CheckoutPage {
  async fillShippingAddress(firstName: string, lastName: string, address: string, city: string) {
    await fill('#first-name', firstName);
    await fill('#last-name', lastName);
    await fill('#address', address);
    await fillAndTab('#city', city); // Fill and move to next field
  }

  async selectShippingMethod(method: string) {
    await selectByValue('#shipping-method', method);
  }

  async agreeToTermsAndCheckout() {
    await check('#agree-terms');
    await check('#newsletter-signup');
    await clickAndNavigate('#place-order-button');
    await expectElementToBeVisible('.order-confirmation', {
      message: 'Order confirmation should display after successful checkout',
    });
  }

  async completeCheckout(firstName: string, lastName: string, address: string, city: string, shippingMethod: string) {
    await this.fillShippingAddress(firstName, lastName, address, city);
    await this.selectShippingMethod(shippingMethod);
    await this.agreeToTermsAndCheckout();
  }
}
```

**Fixture (tests/fixtures/fixture.ts):**

```typescript
import { test as base } from '@anaconda/playwright-utils';
import { CheckoutPage } from '@pages/checkout-page';

export const test = base.extend({
  checkoutPage: async ({}, use) => {
    await use(new CheckoutPage());
  },
});
```

**Spec File (tests/specs/checkout.spec.ts):**

```typescript
import { test } from '@fixture';
import { userData } from '@testdata/user-testdata';
enum Shipping {
  STANDARD = 'standard',
  EXPRESS = 'express',
}

test.describe('Checkout @reg', () => {
  test('should complete checkout with standard shipping', async ({ checkoutPage }) => {
    await checkoutPage.fillShippingAddress(userData.firstname, userData.lastName, userData.address, userData.city);
    await checkoutPage.selectShippingMethod(Shipping.STANDARD);
    await checkoutPage.agreeToTermsAndCheckout();
  });

  test('should complete checkout with express shipping', async ({ checkoutPage }) => {
    await checkoutPage.fillShippingAddress(userData.firstname, userData.lastName, userData.address, userData.city);
    await checkoutPage.selectShippingMethod(Shipping.EXPRESS);
    await checkoutPage.agreeToTermsAndCheckout();
  });
});
```

## Option Types

All option types extend Playwright's native options with:

- `onlyVisible?: boolean` - Default `true` for action functions
- `stable?: boolean` - Wait for element to stop moving before acting

```typescript
type ClickOptions = PlaywrightClickOptions & VisibilityOption & StabilityOption & LoadstateOption;
type FillOptions = PlaywrightFillOptions & VisibilityOption & StabilityOption;
// ... similar pattern for all option types
```
