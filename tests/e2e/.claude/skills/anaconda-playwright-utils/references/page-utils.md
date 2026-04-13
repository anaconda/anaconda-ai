# Page Utils Reference

Source: `src/playwright-utils/utils/page-utils.ts`

## Overview

Page utils provide functions for page management, navigation, and multi-tab handling. All functions work with the library's singleton page pattern — they internally call `getPage()` to access the current page instance, so you don't need to pass `page` to every function call.

## Page Instance Management

### `getPage(): Page`

Returns the current `Page` instance (singleton). This is called internally by all utility functions.

**Usage:**

```typescript
const page = getPage();
// Use only for advanced Playwright API access when utility functions don't cover your use case
```

### `setPage(pageInstance: Page): void`

Sets the current `Page` instance. **Automatically called by the fixture before each test — you don't need to use this directly.**

**Always use the fixture pattern:**

```typescript
import { test } from '@fixture'; // Automatically calls setPage(page) before each test
import { urlData } from '@testdata/urls-testdata'; // imports url data

test('example', async () => {
  // setPage() already called by fixture - just use the utilities
  await gotoURL(urlData.homePageUrl);
  await click('#button');
  await expectPageToHaveURL(/example/, { message: 'Should remain on the example domain after clicking button' });
});
```

**Do NOT manually call `setPage()`** — The fixture handles page setup automatically for all tests.

### `getContext(): BrowserContext`

Returns the browser context associated with the current page.

**Usage:**

```typescript
const context = getContext();
const cookies = await context.cookies();
const pages = await context.pages();
```

### Singleton Pattern Explanation

The library maintains a module-level `page` variable. This design eliminates the need to pass `page` to every function. The fixture handles page setup automatically:

**Standard Playwright (pass page to every function):**

```typescript
await page.goto(url);
await page.locator(sel).click();
await expect(page.locator(sel)).toBeVisible();
```

**With anaconda-playwright-utils + fixture (no setPage needed):**

```typescript
// Fixture calls setPage(page) automatically
await gotoURL(url);
await click(sel);
await expectElementToBeVisible(sel, 'Element should be visible after action');
```

This makes test code cleaner, easier to read, and eliminates the need for manual page setup.

## Multi-Tab Management

### `getAllPages(): Page[]`

Returns an array of all pages (tabs) in the current browser context.

**Example:**

```typescript
const pages = getAllPages();
logger.info(`Total tabs open: ${pages.length}`);
```

### `switchPage(winNum: number, options?): Promise<void>`

Switches to a different page (tab) by its **1-based index** and makes it the current page.

**Parameters:**

- `winNum` — 1-based index (1 = first tab, 2 = second tab, etc.)
- `options.loadState` — Load state to wait for after switching (default: `'load'`)

**Important:** Always use 1-based indexing, not 0-based.

**Example:**

```typescript
// User action opens a new tab (second tab created)
await click('a[target="_blank"]');

// Switch to the new tab (index 2)
await switchPage(2);

// Now all utility functions work on the new tab
await expectPageToHaveURL(/new-page/, { message: 'Should have navigated to new page in second tab' });
await click('#content-button');

// Switch back to the first tab
await switchPage(1);
```

### `switchToDefaultPage(): Promise<void>`

Switches back to the first page (index 1). Convenience function equivalent to `switchPage(1)`.

**Example:**

```typescript
// After working on multiple tabs, return to the original
await switchToDefaultPage();
```

### `closePage(winNum?: number): Promise<void>`

Closes a page (tab) by its 1-based index. If no index is provided, closes the current page.

**Parameters:**

- `winNum` — Optional 1-based index. If omitted, closes the current page.

**Example:**

```typescript
// Close the second tab
await closePage(2);

// Close the current tab
await closePage();
```

**Important:** After closing a page, the library automatically switches to the default page (index 1) if there are remaining pages. Call `switchPage()` explicitly only if you need to land on a specific tab.

## Navigation Functions

### `gotoURL(path: string, options?): Promise<Response | null>`

Navigates to the specified URL. Waits for the default load state before returning.

**Parameters:**

- `path` — URL or path to navigate to
- `options.timeout` — Navigation timeout (default: NAVIGATION_TIMEOUT)
- `options.waitUntil` — Load state to wait for: `'load'` | `'domcontentloaded'` | `'networkidle'` | `'commit'`
- `options.referer` — Referer header value

**Example:**

```typescript
await gotoURL(urlData.exampleUrl);
await gotoURL(urlData.exampleUrl, { waitUntil: 'networkidle' });
await gotoURL('/relative/path'); // Relative to base URL
```

### `getURL(options?): Promise<string>`

Returns the current page URL. Optionally waits for a load state first.

**Example:**

```typescript
const currentURL = await getURL();

// Wait for network idle before getting URL (useful after navigation)
const finalURL = await getURL({ waitUntil: 'networkidle' });
```

### `waitForPageLoadState(options?): Promise<void>`

Waits for a specific page load state.

**Parameters:**

- `options.waitUntil` — Load state to wait for (default: from constants)
- `options.timeout` — Wait timeout

**Example:**

```typescript
// Wait for all network requests to complete
await waitForPageLoadState({ waitUntil: 'networkidle' });

// Wait for DOM to be interactive
await waitForPageLoadState({ waitUntil: 'domcontentloaded' });
```

### `reloadPage(options?): Promise<void>`

Reloads the current page.

**Example:**

```typescript
await reloadPage();
await reloadPage({ waitUntil: 'domcontentloaded' });
```

### `goBack(options?): Promise<void>`

Navigates to the previous page in browser history.

**Example:**

```typescript
// Use goBack() directly for browser history navigation:
await goBack();
```

## Utility Functions

### `wait(ms: number): Promise<void>`

Waits for a specified number of milliseconds. Use sparingly — prefer explicit waits for specific conditions.

**Example:**

```typescript
await wait(1000); // Wait 1 second
```

**Better alternatives:**

```typescript
// Instead of arbitrary wait(), use explicit waits
await waitForElementToBeVisible('.modal'); // Wait for visibility
await waitForPageLoadState(); // Wait for page load
await expectElementToBeVisible('.content', 'Content should be visible after page load'); // Assert with auto-retry
```

### `getWindowSize(): Promise<{ width: number; height: number }>`

Returns the current browser window size in pixels.

**Example:**

```typescript
const { width, height } = await getWindowSize();

if (width < 768) {
  // Handle mobile viewport
}
```

### `saveStorageState(path?: string): Promise<StorageState>`

Saves the current browser storage state (cookies, localStorage, sessionStorage) to a file. Returns the storage state object.

**Parameters:**

- `path` — Optional file path. If provided, saves to file. If omitted, returns state without saving.

**Returns:** Storage state object with structure:

```typescript
{
  cookies: Array<{ name, value, domain, path, ... }>,
  origins: Array<{ origin, localStorage: Array<{ name, value }>, ... }>
}
```

**Example:**

```typescript
// Save to file
await saveStorageState('./auth-state.json');

// Get state without saving
const state = await saveStorageState();
logger.info(`Cookies captured: ${state.cookies?.length ?? 0}`);
```

**Usage with authentication (save auth state for reuse):**

```typescript
import { test } from '@fixture';
import { gotoURL, fill, clickAndNavigate, saveStorageState } from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { userData } from '@testdata/user-testdata';

// tests/auth.setup.ts — Run once to save auth state
test('authenticate and save state', async () => {
  // setPage() called automatically by fixture
  await gotoURL(urlData.loginPageUrl);
  await fill('#username', userData.userName);
  await fill('#password', userData.pwd);
  await clickAndNavigate('#login-button');

  // Save auth state
  await saveStorageState('./.auth/user-auth.json');
});

// playwright.config.ts — Reuse auth state in other tests
{
  name: 'authenticated',
  use: { storageState: './.auth/user-auth.json' },
}

// tests/specs/dashboard.spec.ts — Uses saved auth automatically
import { test } from '@fixture';

test('view dashboard', async ({ dashboardPage }) => {
  // Already authenticated because storageState is loaded
  await dashboardPage.verifyDashboardLoaded();
});
```

## Multi-Tab Workflow Example (Page Object Model)

**Page Object (tests/pages/dashboard-page.ts):**

```typescript
import {
  gotoURL,
  click,
  switchPage,
  closePage,
  switchToDefaultPage,
  expectPageToHaveURL,
  expectElementToBeVisible,
  getText,
  fill,
} from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { reviewData } from '@testdata/product-review-testdata';

export class DashboardPage {
  async navigateToDashboard() {
    await gotoURL(urlData.homePageUrl);
    await expectElementToBeVisible('.dashboard', {
      message: 'Dashboard should be visible after navigation',
    });
  }

  async openProductInNewTab() {
    await click('a[target="_blank"]');
    await switchPage(2); // Switch to new tab
  }

  async verifyProductPageLoaded() {
    await expectPageToHaveURL(/product-2/, {
      message: 'Product page should be loaded when switching to tab 2',
    });
    await expectElementToBeVisible('.product-page', {
      message: 'Product page content should be visible',
    });
  }

  async getProductDetails() {
    const itemId = await getText('.item-id');
    const price = await getText('.price');
    return { itemId, price };
  }

  async submitProductReview(comment: string) {
    await fill('#comment', comment);
    await click('#submit-button');
    await expectElementToBeVisible('.success-message', {
      message: 'Success message should appear after submitting review',
    });
  }

  async closeProductTab() {
    await closePage(2);
  }

  async returnToDashboard() {
    await switchToDefaultPage();
    await expectPageToHaveURL('https://example.com', {
      message: 'Should return to dashboard (tab 1) with correct URL',
    });
    await expectElementToBeVisible('.dashboard', {
      message: 'Dashboard should be visible when returning to tab 1',
    });
  }

  async comparePricesInMultipleTabs() {
    await this.navigateToDashboard();
    await this.openProductInNewTab();
    await this.verifyProductPageLoaded();
    const details = await this.getProductDetails();
    await this.submitProductReview(reviewData.review);
    await this.closeProductTab();
    await this.returnToDashboard();
    return details;
  }
}
```

**Fixture (tests/fixtures/fixture.ts):**

```typescript
import { test as base } from '@anaconda/playwright-utils';
import { DashboardPage } from '@pages/dashboard-page';

export const test = base.extend({
  dashboardPage: async ({}, use) => {
    await use(new DashboardPage());
  },
});
```

**Spec File (tests/specs/multi-tab.spec.ts):**

```typescript
import { test } from '@fixture';
import { reviewData } from '@testdata/product-review-testdata';

test.describe('Multi-tab product workflow @reg', () => {
  test.beforeEach(async ({ dashboardPage }) => {
    await dashboardPage.navigateToDashboard();
  });

  test('should compare prices across tabs', async ({ dashboardPage }) => {
    await dashboardPage.comparePricesInMultipleTabs();
  });

  test('should view product and leave review', async ({ dashboardPage }) => {
    await dashboardPage.openProductInNewTab();
    await dashboardPage.verifyProductPageLoaded();
    await dashboardPage.submitProductReview(reviewData.review);
    await dashboardPage.closeProductTab();
    await dashboardPage.returnToDashboard();
  });
});
```

**Key POM Benefits:**

- ✓ Spec file is clean and readable (just method calls)
- ✓ All implementation details in page object
- ✓ Easy to reuse methods across multiple tests
- ✓ Easier to maintain (change logic once, affects all tests)
- ✓ Methods can be composed (like `comparePricesInMultipleTabs()`)
- ✓ Logging is centralized and consistent

## Option Types

```typescript
type GotoOptions = {
  timeout?: number;
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' | 'commit';
  referer?: string;
};

type NavigationOptions = {
  timeout?: number;
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' | 'commit';
};

type SwitchPageOptions = {
  loadState?: WaitForLoadStateOptions; // Default: 'load'
};
```

## Common Patterns

### Navigation vs Same-Page Interactions

Use `clickAndNavigate()` when a click triggers a full page navigation. Use `click()` for same-page interactions only. Never add waiting conditions after either.

```typescript
// clickAndNavigate: handles framenavigated + load state + element staleness automatically
await clickAndNavigate('#submit-button'); // form submit → new page
await clickAndNavigate(getLocatorByRole('link', { name: 'Dashboard' })); // nav link

// click(): same-page interactions only — no navigation, no explicit waits after
await click('#dropdown-toggle'); // opens dropdown
await click(getLocatorByRole('tab', { name: 'Settings' })); // switches tab (AJAX)
```

### Handle Multiple Tabs in Workflow

```typescript
export class ComparisonPage {
  async comparePrices() {
    // Tab 1: Get first product price
    await gotoURL(urlData.productPageUrl);
    const price1 = await getText('.price');

    // Open second tab
    await click('a[target="_blank"]');
    await switchPage(2);

    // Tab 2: Get second product price
    await expectPageToHaveURL(/product-2/, {
      message: 'Second product page should load in tab 2',
    });
    const price2 = await getText('.price');

    // Compare
    const isCheaper = parseInt(price1) < parseInt(price2);

    // Cleanup
    await closePage(2);
    await switchToDefaultPage();

    return { price1, price2, isCheaper };
  }
}
```

### Save and Restore Authentication

Auth setup is done in a dedicated setup spec that runs once. All other tests then consume the saved state via `playwright.config.ts`.

```typescript
// tests/storage-setup/auth.setup.ts — runs once to capture auth state
import { test } from '@fixture'; // fixture auto-calls setPage(page)
import { gotoURL, fill, clickAndNavigate, saveStorageState } from '@anaconda/playwright-utils';
import { urlData } from '@testdata/urls-testdata';
import { userData } from '@testdata/user-testdata';

test('authenticate and save state', async () => {
  await gotoURL(urlData.loginPageUrl);
  await fill('#username', userData.userName);
  await fill('#password', userData.pwd);
  await clickAndNavigate('#login-button');
  await saveStorageState('./.auth/user-auth.json');
});
```

```typescript
// playwright.config.ts — apply saved state to all authenticated tests
{
  name: 'authenticated',
  use: { storageState: './.auth/user-auth.json' },
  dependencies: ['setup'],
}
```

```typescript
// tests/specs/protected.spec.ts — already authenticated, no login needed
import { test } from '@fixture';

test.describe('Protected pages @smoke', () => {
  test('should access protected content', async ({ protectedPage }) => {
    await protectedPage.verifyProtectedContentIsDisplayed();
  });
});
```
