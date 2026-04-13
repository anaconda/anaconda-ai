# Locator Utils Reference

Source: `src/playwright-utils/utils/locator-utils.ts`

## Locator Strategy Priority

When choosing locators for test code, follow this priority order (best to worst). **Prefer unique CSS or XPath with stable attributes over text-based locators** so that when a check fails you can tell quickly whether the element is missing (bug) or the copy changed (new functionality / locale).

> **Mandatory upgrade rule:** If a DOM snapshot or page inspection reveals a `data-qa-id` or any `data-*` attribute on an element, use it. Never keep a role- or text-based locator when a stable attribute exists — even if the CLI generated one.

### 1. `data-qa-id` attributes (Best)

Purpose-built for testing by QA and developers. Never changes due to styling or refactoring.

```typescript
// HTML: <button data-qa-id="submit-order">Place Order</button>
await click(getLocatorByTestId('submit-order'));

// HTML: <input data-qa-id="email-input">
await fill(getLocatorByTestId('email-input'), userData.email);
```

### 2. `data-testid` and other `data-*` attributes

`data-testid` is Playwright's default `testIdAttribute`. Use `getLocatorByTestId()` for it just like `data-qa-id`. For other custom `data-*` attributes that are not the configured `testIdAttribute`, use a CSS selector.

```typescript
// HTML: <button data-testid="submit-order">Place Order</button>
await click(getLocatorByTestId('submit-order')); // ✅ getLocatorByTestId for data-testid

// HTML: <div data-product-id="shoes-001">...</div>
await click('[data-product-id="shoes-001"]'); // ✅ CSS for non-testId data-* attributes

// HTML: <h2 data-test="complete-header">Thank you for your order</h2>
const orderCompleteMessage = () => getLocator('[data-test="complete-header"]');
await expectElementToContainText(orderCompleteMessage(), /thank you for your order/i, {
  message: 'Checkout complete message should be displayed',
});
```

### 3. `id` attributes

Stable when IDs are meaningful and developer-controlled. Avoid auto-generated IDs.

```typescript
// Good: semantic ID
await fill('#search-input', 'playwright');

// Bad: auto-generated ID (changes every render)
// await fill('#input-7f3a2b', 'playwright');  // DON'T use this
```

### 4. `name` attributes

Reliable for form elements. Often stable across releases.

```typescript
// HTML: <input name="email" type="email">
await fill('[name="email"]', userData.email);

// HTML: <select name="country">...</select>
await selectByText('[name="country"]', 'United States');
```

### 5. XPath with unique attributes

Use when no data-_ or id is available. Target **stable attributes** (e.g. `data-test`, `aria-_`, `type`), not text.

```typescript
// Good: XPath with stable attributes
await click('//button[@aria-label="Close dialog"]');
await click('//input[@type="email"]');

// Good: ancestor scoping with stable attributes
await click('//div[@data-section="billing"]//button[@type="submit"]');
```

### 6. CSS with unique attributes

Use stable attribute selectors so the locator does not depend on copy or locale.

```typescript
// Good: attribute-based CSS
await click('button[aria-label="Close dialog"]');
await fill('input[type="email"]', userData.email);

// Good: scoped by stable parent
await click('.billing-section button[type="submit"]');

// Good: data-test (e.g. Sauce Demo checkout complete)
const orderCompleteMessage = () => getLocator('[data-test="complete-header"]');
await expectElementToContainText(orderCompleteMessage(), /thank you for your order/i);
```

### 7. Playwright built-in locators (role / text) — use only when no stable selector exists

Text- and role-based locators are **flaky**: they change with copy, locale, and country. If the only way to find an element is by its text, a failure does not tell you whether the element is missing (bug) or the wording changed (new feature / i18n). Prefer **data-qa-id**, **data-testid**, **data-\***, **id**, or **unique CSS/XPath** first.

When you must use role or text:

```typescript
// By ARIA role + accessible name
await click(getLocatorByRole('button', { name: 'Submit' }));
await fill(getLocatorByRole('textbox', { name: 'Email' }), userData.email);

// By label text (form fields)
await fill(getLocatorByLabel('Email address'), userData.email);

// By placeholder text
await fill(getLocatorByPlaceholder('Search...'), serachData.query);

// By visible text — avoid for assertions; use stable selector + assert text separately
await click(getLocatorByText('Add to cart'));
```

**Assertions:** Prefer a **stable locator** for the element and assert the **text in the assertion**. That way a failure shows "expected text X, got Y" (copy change) vs element not found (bug).

```typescript
// Prefer: stable selector + text in assertion
const orderCompleteMessage = () => getLocator('[data-test="complete-header"]');
await expectElementToContainText(orderCompleteMessage(), /thank you for your order/i);

// Avoid: locating by text — fails ambiguously if copy or locale changes
// const orderCompleteMessage = () => getLocatorByRole('heading', { name: /thank you for your order/i });
```

### 8. XPath (structural)

Positional XPath. Fragile — use as last resort.

```typescript
// Fragile: depends on DOM structure
await click('//div[@class="form-group"][2]//button');
```

### 9. CSS (structural)

Positional CSS. Equally fragile.

```typescript
// Fragile: depends on DOM order
await click('.form-group:nth-child(2) button');
```

### What to Avoid

Locators that rely on values likely to change between runs:

```typescript
// DON'T: auto-generated IDs
await click('#ember-1234');
await click('#react-select-3-option-0');

// DON'T: dynamic index numbers or counts
await click('//tr[42]/td[3]/button');

// DON'T: timestamp or session-dependent values
await click('[data-id="item-1710612345"]');

// DON'T: deeply nested structural paths
await click('div > div > div > ul > li:nth-child(3) > a');

// DON'T: class names from CSS frameworks (change on rebuild)
await click('.css-1a2b3c');
await click('.MuiButton-root-123');
```

## Key Concept: Visible by Default

`getVisibleLocator()` (and by extension all action functions) filters to only visible elements by default. This prevents accidentally interacting with hidden duplicates.

```typescript
// Returns locator filtered to visible elements
const loc = getVisibleLocator('#submit');

// Equivalent to:
const loc = getLocator('#submit', { onlyVisible: true });

// To include hidden elements:
const loc = getLocator('#submit', { onlyVisible: false });
```

## When Multiple Elements Match

All action functions enforce `onlyVisible: true` internally, so hidden duplicates are already filtered at call time. If multiple elements **still** match after that, the fix is a **more specific locator** — not a visibility wrapper and not an index.

**Step 1 — look for a unique attribute on the target or a stable ancestor, then scope:**

```typescript
// ❌ Never — index breaks silently when the page changes
private readonly pendingButton = () => getLocatorByText('Pending').nth(2);

// ✅ Simple 1-level scoping — prefer getLocatorByTestId chaining
private readonly channelItem = () =>
  getLocatorByTestId('channel-list').locator('[data-qa-id="channel-item"]');

// ✅ Complex scoping (2+ ancestor levels or mixed locator types) — CSS compound is preferred
private readonly pendingButton = '[data-qa-id="channel-list"] [data-qa-id="pending-btn"]';

// ✅ XPath ancestor scope (stable attribute on row + aria-label on button)
private readonly pendingButton = '//tr[@data-qa-id="latest-row"]//button[@aria-label="Pending"]';
```

**When to use CSS compound vs `getLocatorByTestId` chaining:**

- **Single element** — always `getLocatorByTestId()`: `private readonly x = () => getLocatorByTestId('x')`
- **1-level ancestor scope** — `getLocatorByTestId` chaining is preferred: `getLocatorByTestId('parent').locator('[data-qa-id="child"]')`
- **2+ ancestor levels or mixed types** — CSS compound is preferred over deeply nested chains:

```typescript
// ❌ Over-engineered — hard to read
private readonly pendingButton = () =>
  getLocatorByTestId('dashboard')
    .locator(getLocatorByTestId('channel-list').locator('[data-qa-id="pending-btn"]'));

// ✅ CSS compound — flat, readable, and maintainable
private readonly pendingButton = '[data-qa-id="channel-list"] [data-qa-id="pending-btn"]';
```

**Step 2 — if no stable attribute exists on the target, write a custom XPath using structural context + stable ancestor attributes:**

```typescript
// ✅ XPath scoped by nearest ancestor with a stable attribute
private readonly submitButton = '//div[@id="billing-section"]//button[@type="submit"]';
```

**`.nth()` / `.first()` / `.last()` are last resort only.** If you must use one, add a comment explaining why no unique locator was possible — it is a signal to revisit when the element gains a stable attribute.

## Locator Functions

### `getLocator(input: string | Locator, options?: LocatorOptions): Locator`

Core function. If `input` is a string, creates a locator via `page.locator(input)`. If `input` is already a Locator, returns it as-is. When `onlyVisible: true`, appends `visible=true` filter.

### `getVisibleLocator(input: string | Locator, options?: LocatorOptions): Locator`

Same as `getLocator` but defaults to `onlyVisible: true`. This is what action functions use internally.

### `getLocatorByTestId(testId: string | RegExp): Locator`

Uses `page.getByTestId()`. The test ID attribute is configured in `playwright.config.ts` (defaults to `data-testid`).

### `getLocatorByText(text: string | RegExp, options?: GetByTextOptions): Locator`

Uses `page.getByText()`. Finds elements by their text content.

### `getLocatorByRole(role: GetByRoleTypes, options?: GetByRoleOptions): Locator`

Uses `page.getByRole()`. Finds elements by ARIA role.

```typescript
const btn = getLocatorByRole('button', { name: 'Submit' });
const link = getLocatorByRole('link', { name: /learn more/i });
```

### `getLocatorByLabel(text: string | RegExp, options?: GetByRoleOptions): Locator`

Uses `page.getByLabel()`. Finds form elements by their associated label.

### `getLocatorByPlaceholder(text: string | RegExp, options?: GetByPlaceholderOptions): Locator`

Uses `page.getByPlaceholder()`. Finds input elements by placeholder text.

### `getAllLocators(input: string | Locator, options?): Promise<Locator[]>`

Returns all matching locators as an array. Waits for at least the first element to be attached before resolving.

Options include `waitForLocator?: boolean` (default `true`) and `timeout?: number`.

## Frame Functions

### `getFrame(frameSelector: FrameOptions, options?): Frame | null`

Gets a Frame by name or URL. Throws if not found unless `{ force: true }`.

```typescript
const frame = getFrame({ name: 'my-iframe' });
const frame = getFrame({ url: /embed/ });
```

### `getFrameLocator(frameInput: string | FrameLocator): FrameLocator`

Gets a FrameLocator from a selector or existing FrameLocator.

### `getLocatorInFrame(frameInput, input): Locator`

Gets a locator for an element inside a frame.

```typescript
const btn = getLocatorInFrame('#my-iframe', '#submit-btn');
await click(btn); // Works with action utils
```

## Option Types

```typescript
type LocatorOptions = PlaywrightLocatorOptions & { onlyVisible?: boolean };
type LocatorWaitOptions = { waitForLocator?: boolean } & TimeoutOption;
type GetByTextOptions = PlaywrightGetByTextOptions & { onlyVisible?: boolean };
type GetByRoleTypes = PlaywrightGetByRoleTypes & { onlyVisible?: boolean };
type GetByRoleOptions = PlaywrightGetByRoleOptions & { onlyVisible?: boolean };
type GetByPlaceholderOptions = PlaywrightGetByPlaceholderOptions & { onlyVisible?: boolean };
type FrameOptions = PlaywrightFrameOptions; // { name?: string, url?: string | RegExp }
```
