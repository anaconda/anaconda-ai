# Assert Utils Reference

Source: `src/playwright-utils/utils/assert-utils.ts`

## Using Assertions in Spec Files

**Do not use assertions in spec files.** Assertions are for building and verifying behaviour inside page objects (e.g. `verify*` methods). Spec files should only orchestrate steps and call those methods so the test reads like a clear, readable scenario.

**Important Guidelines:**

- ✅ All assertions go in page object methods (verify*, check* methods)
- ✅ All test data goes in `tests/testdata/` folder and imported
- ✅ Never use `console.log()` in any test file
- ✅ If rare logging needed, use `logger` from `@anaconda/playwright-utils` in page objects only (never in specs)
- ✅ Spec files should read like scenarios - only method calls, no raw utilities

### Good Example — Readable Spec (No Assertions in Spec)

The spec reads like a test plan; all assertions live in page object classes.

**Spec file** — plain English, no assertion utils:

```typescript
import { test } from '@fixture';

test.describe('Checkout flow @smoke', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.navigateToLoginPage();
    await loginPage.loginWithValidCredentials();
  });

  test('should complete full checkout flow', async ({ productsPage, cartPage, checkoutPage }) => {
    await productsPage.verifyProductsPageIsDisplayed();
    await productsPage.addToCartByProductNumber(1);
    await cartPage.verifyMiniCartCount('1');
    await checkoutPage.goToCart();
    await checkoutPage.fillCheckoutInfo();
    await checkoutPage.clickContinue();
    await checkoutPage.clickFinish();
    await checkoutPage.verifyOrderComplete();
  });
});
```

**Page file** — assertions live here with descriptive messages:

```typescript
// tests/pages/sauce-demo-products-page.ts
import { expectElementToBeVisible, expectElementToBeHidden } from '@anaconda/playwright-utils';

export class SauceDemoProductsPage {
  private readonly productsContainer = '#inventory_container';

  async verifyProductsPageIsDisplayed(): Promise<void> {
    await expectElementToBeVisible(this.productsContainer, {
      timeout: SMALL_TIMEOUT,
      message: 'Logged in user should see Products',
    });
  }

  async verifyProductsPageIsNotDisplayed(): Promise<void> {
    await expectElementToBeHidden(this.productsContainer, 'Products should not be displayed');
  }
}
```

```typescript
// tests/pages/sauce-demo-checkout-page.ts
import { expectElementToContainText } from '@anaconda/playwright-utils';

export class SauceDemoCheckoutPage {
  private readonly orderCompleteMessage = '[data-test="complete-header"]';

  async verifyOrderComplete(): Promise<void> {
    await expectElementToContainText(this.orderCompleteMessage, /thank you for your order/i, {
      message: 'Checkout complete message should be displayed',
    });
  }
}
```

## Overview

All assertion functions:

- Accept `string | Locator` as the `input` parameter
- Support `soft` option for soft assertions that don't stop the test
- Support `timeout` option to override the default expect timeout
- Support `message` option (or a string shorthand) for descriptive failure messages
- Auto-retry until the condition is met or timeout is reached

## Soft Assertions vs Hard Assertions

### Hard Assertions (Default)

**Use hard assertions for critical checks** — the test fails immediately when the assertion fails.

```typescript
import { test } from '@fixture';

test('critical flow', async () => {
  await expectElementToBeVisible('.critical-element', 'Critical element must be visible to proceed'); // Hard assertion — fails immediately
  // Test stops here if assertion fails
});
```

### Soft Assertions

**Use soft assertions for non-critical checks** — the test continues even if the assertion fails, and fails at the end if any soft assertion failed. Put soft assertions inside page object `verify*` methods, just like hard assertions.

```typescript
// In the page object class:
async verifyOptionalFeatures(): Promise<void> {
  await expectElementToBeVisible('.optional-banner', { soft: true, message: 'Banner should display (non-critical)' });
  await expectElementToHaveText('.secondary-message', 'Info', { soft: true, message: 'Secondary message should say Info' });
  await expectPageToHaveURL(/dashboard/, { message: 'Should be on dashboard' });
}
```

```typescript
// In the spec file — call assertAllSoftAssertions after the page object method:
import { test } from '@fixture';
import { assertAllSoftAssertions } from '@anaconda/playwright-utils';

test.describe('Dashboard optional features @reg', () => {
  test('should display optional features', async ({ dashboardPage }) => {
    await dashboardPage.verifyOptionalFeatures();
    // Fail immediately after all soft checks are done (rather than at test end)
    assertAllSoftAssertions(test.info());
  });
});
```

**When to use each:**

- **Hard assertions** = Critical functionality (login, checkout, core features)
- **Soft assertions** = Nice-to-have features, optional UI elements, analytics tracking

## Element Assertions

| Function                                          | Description                            |
| ------------------------------------------------- | -------------------------------------- |
| `expectElementToBeVisible(input, options?)`       | Element is in DOM and visible          |
| `expectElementToBeHidden(input, options?)`        | Element is not in DOM or hidden        |
| `expectElementToBeAttached(input, options?)`      | Element is in DOM (may not be visible) |
| `expectElementToBeInViewport(input, options?)`    | Element is visible in viewport         |
| `expectElementNotToBeInViewport(input, options?)` | Element is not in viewport             |
| `expectElementToBeChecked(input, options?)`       | Checkbox/radio is checked              |
| `expectElementNotToBeChecked(input, options?)`    | Checkbox/radio is not checked          |
| `expectElementToBeDisabled(input, options?)`      | Element is disabled                    |
| `expectElementToBeEnabled(input, options?)`       | Element is enabled                     |
| `expectElementToBeEditable(input, options?)`      | Element is editable                    |

## Text Assertions

| Function                                               | Description                     |
| ------------------------------------------------------ | ------------------------------- |
| `expectElementToHaveText(input, text, options?)`       | Text equals value (exact match) |
| `expectElementNotToHaveText(input, text, options?)`    | Text does NOT equal value       |
| `expectElementToContainText(input, text, options?)`    | Text contains value (substring) |
| `expectElementNotToContainText(input, text, options?)` | Text does NOT contain value     |

`text` accepts `string | RegExp | Array<string | RegExp>`. Options extend with `ignoreCase?: boolean` and `useInnerText?: boolean`.

## Value Assertions

| Function                                            | Description                         |
| --------------------------------------------------- | ----------------------------------- |
| `expectElementToHaveValue(input, text, options?)`   | Input has the specified value       |
| `expectElementToHaveValues(input, texts, options?)` | Multi-select has specified values   |
| `expectElementValueToBeEmpty(input, options?)`      | Input/editable element is empty     |
| `expectElementValueNotToBeEmpty(input, options?)`   | Input/editable element is not empty |

## Attribute & Count Assertions

| Function                                                        | Description                              |
| --------------------------------------------------------------- | ---------------------------------------- |
| `expectElementToHaveAttribute(input, attr, value, options?)`    | Attribute equals value                   |
| `expectElementToContainAttribute(input, attr, value, options?)` | Attribute contains value                 |
| `expectElementToHaveCount(input, count, options?)`              | Number of matching elements equals count |

## Page Assertions

| Function                                         | Description                       |
| ------------------------------------------------ | --------------------------------- |
| `expectPageToHaveURL(urlOrRegExp, options?)`     | Page URL matches exactly          |
| `expectPageToContainURL(url, options?)`          | Page URL contains string          |
| `expectPageToHaveTitle(titleOrRegExp, options?)` | Page title matches                |
| `expectPageSizeToBeEqualTo(count, options?)`     | Number of open pages equals count |

## Alert Assertions

| Function                                        | Description                                        |
| ----------------------------------------------- | -------------------------------------------------- |
| `expectAlertToHaveText(input, text, options?)`  | Clicks element, asserts alert text equals value    |
| `expectAlertToMatchText(input, text, options?)` | Clicks element, asserts alert text matches pattern |

## Page Object Model Assertion Examples

### Dashboard Page with Verification Methods

**Page Object (tests/pages/dashboard-page.ts):**

```typescript
import {
  expectElementToBeVisible,
  expectElementToHaveText,
  expectElementToHaveCount,
  expectPageToHaveURL,
  expectPageToHaveTitle,
  logger,
} from '@anaconda/playwright-utils';

export class DashboardPage {
  async verifyDashboardLoaded() {
    // All assertions in page object, not in spec
    await expectPageToHaveURL(/dashboard/, {
      message: 'User should be on dashboard page',
    });
    await expectPageToHaveTitle('Dashboard', {
      message: 'Page title should be Dashboard',
    });
    await expectElementToBeVisible('.dashboard-header', {
      message: 'Dashboard header should be visible',
    });
  }

  async verifyWelcomeMessage(username: string) {
    await expectElementToHaveText('.welcome-message', `Welcome, ${username}`, {
      message: `Welcome message should greet user as ${username}`,
    });
  }

  async verifyUserListDisplayed() {
    await expectElementToBeVisible('.user-list', {
      message: 'User list should be visible',
    });
    const userCount = 5;
    await expectElementToHaveCount('.user-list-item', userCount, {
      message: `User list should have exactly ${userCount} items`,
    });
  }

  async verifyEmptyState() {
    await expectElementToBeVisible('.empty-state', {
      message: 'Empty state message should be visible',
    });
    await expectElementToHaveText('.empty-message', 'No items found', {
      message: 'Empty state should display correct message',
    });
  }

  async verifyCriticalElements() {
    // Combine multiple assertions for readability
    await expectElementToBeVisible('.header', {
      message: 'Header must be visible for navigation',
    });
    await expectElementToBeVisible('.navigation', {
      message: 'Navigation menu must be visible',
    });
    await expectElementToBeVisible('.content-area', {
      message: 'Main content area must be visible',
    });
  }

  async verifyOptionalFeatures() {
    // Use soft assertions for non-critical checks
    await expectElementToBeVisible('.analytics-widget', {
      soft: true,
      message: 'Analytics widget should display (non-critical)',
    });
    await expectElementToBeVisible('.suggested-items', {
      soft: true,
      message: 'Suggested items should display (non-critical)',
    });
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

**Spec File (tests/specs/dashboard.spec.ts):**

```typescript
import { test } from '@fixture';
import { assertAllSoftAssertions } from '@anaconda/playwright-utils';

test.describe('Dashboard @smoke', () => {
  test.beforeEach(async ({ dashboardPage }) => {
    await dashboardPage.navigateToDashboard();
  });

  test('should load dashboard correctly', async ({ dashboardPage }) => {
    await dashboardPage.verifyDashboardLoaded();
    await dashboardPage.verifyWelcomeMessage('John Doe');
    await dashboardPage.verifyCriticalElements();
  });

  test('should display user list', async ({ dashboardPage }) => {
    await dashboardPage.verifyUserListDisplayed();
  });

  test('should display optional features', async ({ dashboardPage }) => {
    await dashboardPage.verifyOptionalFeatures();
    assertAllSoftAssertions(test.info());
  });
});
```

### API Response Verification Page Object

**Page Object (tests/pages/api/user-api-verifier.ts):**

```typescript
import { expect, getRequest, postRequest } from '@anaconda/playwright-utils';
import { BASE_URL } from 'playwright.config';

export class UserAPIVerifier {
  async verifyUserExists(userId: number) {
    const response = await getRequest(`${BASe_URL}/users/${userId}`);
    await expect(response).toBeOK();

    const user = await response.json();
    expect(user, {
      message: `User ${userId} should exist in API`,
    }).toHaveProperty('id', userId);
    expect(user, {
      message: 'User should have name property',
    }).toHaveProperty('name');
    expect(user, {
      message: 'User should have email property',
    }).toHaveProperty('email');
  }

  async verifyUserCreated(userData: Record<string, unknown>): Promise<void> {
    const response = await postRequest(`${this.baseURL}/users`, { data: userData });
    expect(response.status(), {
      message: 'POST request should return 201 Created status',
    }).toBe(201);

    const newUser = await response.json();
    expect(newUser, {
      message: 'Created user should have id property',
    }).toHaveProperty('id');
    expect(newUser.name, {
      message: `User name should match submitted data: ${userData.name}`,
    }).toBe(userData.name as string);
  }

  async verifyUserList() {
    const response = await getRequest(`${this.baseURL}/users`);
    await expect(response).toBeOK();

    const users = await response.json();
    expect(Array.isArray(users), {
      message: 'API should return array of users',
    }).toBeTruthy();
    expect(users.length, {
      message: 'User list should not be empty',
    }).toBeGreaterThan(0);

    return users;
  }

  async verifyUserNotFound(userId: number) {
    const response = await getRequest(`${this.baseURL}/users/${userId}`);
    expect(response.status(), {
      message: `User ${userId} should return 404 Not Found`,
    }).toBe(404);
  }
}
```

**Fixture (tests/fixtures/fixture.ts):**

```typescript
import { test as base } from '@anaconda/playwright-utils';
import { UserAPIVerifier } from '@pages/api/user-api-verifier';

export const test = base.extend({
  userAPI: async ({}, use) => {
    await use(new UserAPIVerifier());
  },
});
```

**Spec File (tests/specs/api/user-api.spec.ts):**

```typescript
import { test } from '@fixture';
import { testUsers } from '@testdata/users';

test.describe('User API @smoke', () => {
  test('should verify user exists', async ({ userAPI }) => {
    await userAPI.verifyUserExists(1);
  });

  test('should create a new user', async ({ userAPI }) => {
    await userAPI.verifyUserCreated(testUsers.validUser);
  });

  test('should return user list', async ({ userAPI }) => {
    await userAPI.verifyUserList();
  });
});
```

## Test Data Organization

**Always store test data in `tests/testdata/` folder, not inline in spec files:**

```typescript
// ✓ GOOD: Import from testdata folder
// tests/testdata/users.ts
export const testUsers = {
  validUser: { name: 'John Doe', email: 'john@example.com' },
  adminUser: { name: 'Admin', email: 'admin@example.com' },
};

// tests/specs/api/user-api.spec.ts
import { test } from '@fixture';
import { testUsers } from '@testdata/users';

// ✓ GOOD: Page object method returns data; spec only orchestrates, no raw assertions
test.describe('User API @smoke', () => {
  test('verify user created', async ({ userAPI }) => {
    await userAPI.verifyUserCreated(testUsers.validUser);
  });
});

// ✗ WRONG: Inline test data
test.describe('User API @smoke', () => {
  test('verify user created', async ({ userAPI }) => {
    await userAPI.verifyUserCreated({
      name: 'John Doe',
      email: 'john@example.com',
    });
  });
});
```

This keeps specs clean, reuses data across tests, and separates concerns.

## Key POM Assertion Rules

✓ **Do:** Put all assertions in page object methods
✓ **Do:** Use descriptive method names like `verifyDashboardLoaded()`
✓ **Do:** Combine related assertions into single methods for readability
✓ **Do:** Add descriptive error messages to assertions
✓ **Do:** Return data from verification methods if needed by tests
✓ **Do:** Store test data in `tests/testdata/` and import it

✗ **Don't:** Put assertions directly in spec files
✗ **Don't:** Repeat assertion patterns across multiple tests
✗ **Don't:** Mix assertions with actions without clear method names
✗ **Don't:** Use inline test data (use testdata folder instead)

## Option Types

```typescript
type ExpectOptions = TimeoutOption & SoftOption & MessageOrOptions;
// TimeoutOption = { timeout?: number }
// SoftOption = { soft?: boolean }
// MessageOrOptions = string | { message?: string }

type ExpectTextOptions = { ignoreCase?: boolean; useInnerText?: boolean };
```
