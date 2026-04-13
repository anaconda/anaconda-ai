# Element Utils Reference

Source: `src/playwright-utils/utils/element-utils.ts`

## Overview

Element utils provide functions for retrieving data from elements and checking element states. These functions are for:

- **Data extraction** — Use in variable assignments and conditionals (reading text, input values, attributes, counts)
- **State checks** — Use in if/while statements to conditionally execute logic
- **Wait functions** — Use for explicit synchronization when element state changes are expected

**Do NOT use these for assertions** — Use `assert-utils` instead. Assertions are for test validation; these functions are for runtime data extraction and conditional logic.

**Logging note:** Avoid logging in page objects. Use element-utils for data extraction only. If logging is needed in rare cases, import `logger` from `@anaconda/playwright-utils` and use it minimally at page object level (never in spec files).

## Data Retrieval Functions

### `getText(input, options?): Promise<string>`

Returns the trimmed inner text of an element. Use for reading visible text content.

```typescript
const text = await getText('.page-title');
// Use for data extraction and variable assignment, not logging

const text2 = await getText(getLocatorByRole('heading')); // Pass Locator directly
```

### `getAllTexts(input, options?): Promise<string[]>`

Returns trimmed inner text of all matching elements. Automatically waits for the first element to be attached.

```typescript
const allLabels = await getAllTexts('.form-label');
// Use for data extraction, assignments, and conditionals
```

### `getInputValue(input, options?): Promise<string>`

Returns the trimmed input value of a form element. Use for reading input field values.

```typescript
const username = await getInputValue('#username-input');
const email = await getInputValue(getLocatorByLabel('Email'));
```

### `getAllInputValues(input, options?): Promise<string[]>`

Returns input values of all matching form elements.

```typescript
const values = await getAllInputValues('input[type="checkbox"]:checked');
```

### `getAttribute(input, attributeName, options?): Promise<string | null>`

Returns the trimmed attribute value, or `null` if the attribute doesn't exist.

```typescript
const ariaLabel = await getAttribute('.button', 'aria-label');
const dataId = await getAttribute('#item-1', 'data-item-id');

// Useful for conditional logic
if (!ariaLabel) {
  logger.warn('Button missing aria-label — consider adding one for accessibility');
}
```

### `getLocatorCount(input, options?): Promise<number>`

Returns the count of matching elements. Returns 0 on error instead of throwing.

```typescript
const itemCount = await getLocatorCount('.cart-item');
if (itemCount === 0) {
  logger.info('Cart is empty — no items to process');
}
```

## Conditional Check Functions

These functions return boolean values for use in conditional statements. They catch errors and return `false` instead of throwing, making them safe for use in conditionals.

### `isElementAttached(input, options?): Promise<boolean>`

Checks if an element is attached to the DOM. Returns `false` instead of throwing on timeout.

```typescript
if (await isElementAttached('.modal')) {
  // Modal exists in DOM - conditional logic
  await click('.modal .close-button');
}
```

### `isElementVisible(input, options?): Promise<boolean>`

Checks if an element is attached to the DOM and visible. Returns `false` instead of throwing.

```typescript
if (await isElementVisible('.error-message')) {
  const errorText = await getText('.error-message');
  // Conditional logic - use in page object methods
}
```

### `isElementHidden(input, options?): Promise<boolean>`

Checks if an element is hidden or not present in the DOM. Returns `false` instead of throwing.

```typescript
// Use in conditional logic — e.g. skip an action when element is already gone
if (!(await isElementHidden('.loading-spinner'))) {
  // spinner still visible — wait for it to disappear
  await waitForElementToBeHidden('.loading-spinner');
}
```

### `isElementChecked(input, options?): Promise<boolean>`

Checks if a checkbox or radio button element is checked. Returns `false` instead of throwing.

```typescript
if (await isElementChecked('#agree-checkbox')) {
  await click('#submit-button');
}
// Use for conditional logic in page object methods
```

**Usage pattern for data extraction and conditionals:**

```typescript
// Extract data based on conditions
const itemCount = await getLocatorCount('.cart-item');
const cartTotal = await getText('.cart-total');

if (itemCount > 0) {
  await click('.checkout-button');
  await expectElementToBeVisible('.payment-form', {
    message: 'Payment form should appear',
  }); // Assert in page object
}
// Use conditional logic for runtime decisions
```

## Wait Functions

Use these for explicit synchronization when element state changes are expected. Unlike the check functions above, these throw on timeout.

### `waitForElementToBeVisible(input, options?): Promise<void>`

Waits for an element to be visible. Throws on timeout.

```typescript
await waitForElementToBeVisible('.modal', { timeout: SMALL_TIMEOUT });
// Continue execution only after modal is visible
```

### `waitForElementToBeHidden(input, options?): Promise<void>`

Waits for an element to be hidden or detached from the DOM. Throws on timeout.

```typescript
// Wait for loading spinner to disappear
await waitForElementToBeHidden('.loading-spinner', { timeout: STANDARD_TIMEOUT });
// Now safe to interact with page
```

### `waitForElementToBeAttached(input, options?): Promise<void>`

Waits for an element to be attached to the DOM. Throws on timeout.

```typescript
await waitForElementToBeAttached('.dynamic-content');
```

### `waitForElementToBeDetached(input, options?): Promise<void>`

Waits for an element to be detached from the DOM (removed from HTML). Throws on timeout.

```typescript
// Modal was closed and removed from DOM
await waitForElementToBeDetached('.modal');
```

### `waitForElementToBeStable(input, options?): Promise<boolean>`

Waits for an element's position to stop changing (3 consecutive stable position checks). Returns `false` if max wait time exceeded.

```typescript
// Wait for animation to finish before clicking
await waitForElementToBeStable('.animated-button');
await click('.animated-button');
```

### `waitForFirstElementToBeAttached(input, options?): Promise<void>`

Internal helper function — waits for the first matching element to be attached to DOM. Used internally by `getAllTexts` and `getLocatorCount`.

**Usage pattern with waits (in a page object method):**

```typescript
async verifyContentLoadedAfterLoadMore(): Promise<void> {
  await click('.load-more-button');

  // Wait for loading indicator to appear and disappear
  await waitForElementToBeVisible('.loading-spinner');
  await waitForElementToBeHidden('.loading-spinner');

  // Assert after dynamic content has loaded
  await expectElementToBeVisible('.new-content', 'New content should appear after loading spinner hides');
  const newItems = await getLocatorCount('.content-item');
  expect(newItems, { message: 'At least one content item should be visible after load' }).toBeGreaterThan(0);
}
```

## Option Types

All option types are defined in `src/playwright-utils/types/optional-parameter-types.ts`:

```typescript
type TimeoutOption = {
  timeout?: number; // Default: SMALL_TIMEOUT (5000ms)
};

type LocatorWaitOptions = TimeoutOption & {
  waitForLocator?: boolean; // Default: true — whether to wait for first element to attach
};
```

Import types for TypeScript:

```typescript
import { TimeoutOption, LocatorWaitOptions } from '@anaconda/playwright-utils';
```

## Wait vs Check vs Assert: Quick Reference

| Function Type | Example                       | Behavior on Timeout | Use Case                                               |
| ------------- | ----------------------------- | ------------------- | ------------------------------------------------------ |
| **Wait**      | `waitForElementToBeVisible()` | Throws error        | Explicit synchronization when state change is expected |
| **Check**     | `isElementVisible()`          | Returns `false`     | Conditional logic (if/while statements)                |
| **Assert**    | `expectElementToBeVisible()`  | Fails test          | Test validation (only in page objects)                 |

**Choose based on your use case:**

```typescript
// ✓ Correct: Check for conditional logic
if (await isElementVisible('.optional-form')) {
  await fill('.optional-form', 'data');
}

// ✓ Correct: Wait for expected state change
await click('.toggle-button');
await waitForElementToBeVisible('.expanded-content');

// ✓ Correct: Assert in page object verify method (class-based, with message)
async verifyContentLoaded(): Promise<void> {
  await expectElementToBeVisible('.content', 'Content should be visible after dynamic load');
}

// ✗ Wrong: Don't use wait in conditionals (wastes time)
if (await waitForElementToBeVisible('.optional-item')) {
  // Slow!
  // ...
}

// ✗ Wrong: Don't use assert in specs (should be in page objects)
test('spec file', async () => {
  await expectElementToBeVisible('.item'); // Move to page object!
});
```

## Page Object Model Examples

### Using Element-Utils for Data Extraction

**Page Object (tests/pages/product-page.ts):**

```typescript
import {
  // element-utils
  getText,
  getAllTexts,
  getInputValue,
  getAttribute,
  getLocatorCount,
  isElementVisible,
  isElementHidden,
  waitForElementToBeVisible,
  // assert-utils
  expectElementToBeVisible,
} from '@anaconda/playwright-utils';

export class ProductPage {
  async getProductTitle(): Promise<string> {
    return await getText('.product-title');
  }

  async getProductPrice(): Promise<string> {
    return await getText('.product-price');
  }

  async getAllProductTitles(): Promise<string[]> {
    return await getAllTexts('.product-item .title');
  }

  async getCartItemCount(): Promise<number> {
    return await getLocatorCount('.cart-item');
  }

  async isProductInStock(): Promise<boolean> {
    return await isElementVisible('.in-stock-badge');
  }

  async waitForPriceUpdateAfterDiscount(): Promise<string> {
    await waitForElementToBeVisible('.discount-price', { timeout: SMALL_TIMEOUT });
    return await getText('.discount-price');
  }

  async getProductDataForComparison(): Promise<{ title: string; price: string; sku: string | null; rating: string }> {
    const title = await getText('.product-title');
    const price = await getText('.product-price');
    const sku = await getAttribute('.product-sku', 'value');
    const rating = await getText('.product-rating');
    return { title, price, sku, rating };
  }

  async getSearchResults(): Promise<string[]> {
    const resultCount = await getLocatorCount('.search-result-item');
    if (resultCount === 0) {
      return [];
    }
    return await getAllTexts('.search-result-item');
  }

  async verifyProductDataLoaded(): Promise<void> {
    const title = await getText('.product-title');
    await expectElementToBeVisible('.product-title', `Product title "${title}" should be visible`);
    await expectElementToBeVisible('.product-price', 'Product price should be visible');
  }
}
```

**Fixture (tests/fixtures/fixture.ts):**

```typescript
import { test as base } from '@anaconda/playwright-utils';
import { ProductPage } from '@pages/product-page';

export const test = base.extend({
  productPage: async ({}, use) => {
    await use(new ProductPage());
  },
});
```

**Spec File (tests/specs/product.spec.ts):**

```typescript
import { test } from '@fixture';

test('verify product page is loaded', async ({ productPage }) => {
  // All assertions are inside the page object verify method
  await productPage.verifyProductDataLoaded();
});

test('wait for dynamic price update', async ({ productPage }) => {
  // Page object handles wait + extraction internally
  await productPage.waitForPriceUpdateAfterDiscount();
});
```

## Summary: Element-Utils Usage in POM

| Purpose               | Use In              | Example                                          |
| --------------------- | ------------------- | ------------------------------------------------ |
| **Data extraction**   | Page object methods | `getText()`, `getAllTexts()`, `getAttribute()`   |
| **Get counts**        | Page object methods | `getLocatorCount()` to verify list sizes         |
| **Conditional logic** | Page object methods | `isElementVisible()` to handle optional elements |
| **Wait for changes**  | Page object methods | `waitForElementToBeVisible()` to sync state      |
| **Assertions**        | Page object methods | Result of `getText()` in expectation             |
| **Spec files**        | Never directly      | Only call page object methods                    |
