# API Utils Reference

Source: `src/playwright-utils/utils/api-utils.ts`

## Overview

API utils provide simplified HTTP request functions wrapping Playwright's `APIRequestContext`. These functions use the page's request context, so they automatically share cookies, authentication tokens, and storage with the browser session — making them ideal for testing API interactions alongside UI tests.

All request functions return Playwright's `APIResponse` object, which provides methods for checking status, parsing body, and accessing headers.

## Request Context

### `getAPIRequestContext(): APIRequestContext`

Returns the `APIRequestContext` from the current page. Equivalent to `page.request` in Playwright.

**Usage:**

```typescript
const context = getAPIRequestContext();
// Use for advanced scenarios not covered by the helper functions
// (e.g., complex retry logic, custom interceptors)
```

## HTTP Request Functions

All request functions return `Promise<APIResponse>`. Options are passed directly to Playwright's native APIRequestContext methods.

### `getRequest(url, options?): Promise<APIResponse>`

Performs an HTTP GET request.

**Example:**

```typescript
const response = await getRequest('https://api.example.com/users/1');
await expect(response).toBeOK();

const user = await response.json();
logger.info(`Retrieved user: ${user.name}`);
```

### `postRequest(url, options?): Promise<APIResponse>`

Performs an HTTP POST request.

**Example:**

```typescript
const response = await postRequest('https://api.example.com/users', {
  data: {
    name: 'John Doe',
    email: 'john@example.com',
  },
});

expect(response.status(), { message: 'POST should return 201 Created' }).toBe(201);
const newUser = await response.json();
logger.info(`Created user ID: ${newUser.id}`);
```

### `putRequest(url, options?): Promise<APIResponse>`

Performs an HTTP PUT request (replace entire resource).

**Example:**

```typescript
const response = await putRequest('https://api.example.com/users/1', {
  data: {
    name: 'Jane Doe',
    email: 'jane@example.com',
  },
});
```

### `patchRequest(url, options?): Promise<APIResponse>`

Performs an HTTP PATCH request (partial update).

**Example:**

```typescript
const response = await patchRequest('https://api.example.com/users/1', {
  data: { email: 'newemail@example.com' },
});
```

### `deleteRequest(url, options?): Promise<APIResponse>`

Performs an HTTP DELETE request.

**Example:**

```typescript
const response = await deleteRequest('https://api.example.com/users/1');
expect(response.status()).toBe(204); // No content
```

## Response Handling

All request functions return Playwright's `APIResponse` object with the following methods:

```typescript
const response = await getRequest(url);

// Status info
response.ok(); // boolean: true if status 200-299
response.status(); // number: HTTP status code

// Body parsing
await response.json(); // Parse body as JSON object
await response.text(); // Parse body as plain text
await response.body(); // Get body as Buffer

// Headers
response.headers(); // Get headers as object
response.headersArray(); // Get headers as array of [name, value] pairs
```

## Common Patterns

### Response Status Validation

```typescript
import { expect, getRequest } from '@anaconda/playwright-utils';

const response = await getRequest('https://api.example.com/users/1');

// Best practice: Use await expect() for async response validation
await expect(response).toBeOK();

const data = await response.json();
expect(data).toHaveProperty('id');
expect(data).toHaveProperty('name');
```

Alternative error handling pattern:

```typescript
const response = await getRequest('/api/users/1');

if (!response.ok()) {
  const error = await response.text();
  throw new Error(`API error: ${response.status()} - ${error}`);
}

const data = await response.json();
```

### Authentication Headers

```typescript
const token = 'your-bearer-token';

const response = await getRequest('/api/protected', {
  headers: {
    Authorization: `Bearer ${token}`,
  },
});
```

### Request Body Formats

```typescript
// JSON body (default)
await postRequest('/api/users', {
  data: { name: 'John', age: 30 },
  headers: { 'Content-Type': 'application/json' },
});

// Form data
await postRequest('/api/form', {
  form: {
    username: 'john',
    password: 'secret',
  },
});

// Multipart form (file upload)
const fs = require('fs');
await postRequest('/api/upload', {
  multipart: {
    file: fs.createReadStream('document.pdf'),
    description: 'Important document',
    category: 'reports',
  },
});
```

### Response Validation Pattern (Page Object)

```typescript
// tests/pages/api/items-api.ts
import { expect, getRequest, logger } from '@anaconda/playwright-utils';

export class ItemsAPI {
  private readonly baseURL = 'https://api.example.com';

  async verifyItemsExist(): Promise<void> {
    const response = await getRequest(`${this.baseURL}/items`);
    await expect(response).toBeOK();

    const items = await response.json();
    expect(Array.isArray(items), { message: 'Items response should be an array' }).toBeTruthy();
    expect(items.length, { message: 'Items list should not be empty' }).toBeGreaterThan(0);
    expect(items[0], { message: 'Each item should have an id field' }).toHaveProperty('id');
    expect(items[0], { message: 'Each item should have a name field' }).toHaveProperty('name');

    logger.info(`Verified ${items.length} items exist`);
  }
}
```

Fixture and spec:

```typescript
// tests/fixtures/fixture.ts
import { test as base } from '@anaconda/playwright-utils';
import { ItemsAPI } from '@pages/api/items-api';

export const test = base.extend<{ itemsAPI: ItemsAPI }>({
  itemsAPI: async ({}, use) => {
    await use(new ItemsAPI());
  },
});

// tests/specs/api/items-api.spec.ts
import { test } from '@fixture';

test('verify items API returns data', async ({ itemsAPI }) => {
  await itemsAPI.verifyItemsExist();
});
```

### Error Handling

```typescript
import { logger, postRequest } from '@anaconda/playwright-utils';

const response = await postRequest('/api/users', {
  data: { email: 'invalid' },
});

if (response.status() === 400) {
  const errors = await response.json();
  logger.warn(`Validation errors returned: ${JSON.stringify(errors)}`);
}
```

## Integration with Page Objects

API utils work well with page object patterns for separating API testing concerns from UI. Always use `async/await` and import from proper modules:

```typescript
// tests/pages/api/user-api.ts
import { deleteRequest, expect, getRequest, logger, postRequest } from '@anaconda/playwright-utils';

export class UserAPI {
  private readonly baseURL = 'https://api.example.com';

  async getUser(id: number) {
    const response = await getRequest(`${this.baseURL}/users/${id}`);
    await expect(response).toBeOK();
    const user = await response.json();
    logger.info(`Retrieved user: ${user.name}`);
    return user;
  }

  async createUser(userData: Record<string, unknown>) {
    const response = await postRequest(`${this.baseURL}/users`, {
      data: userData,
    });
    expect(response.status(), { message: 'POST /users should return 201 Created' }).toBe(201);
    const newUser = await response.json();
    logger.info(`Created user with ID: ${newUser.id}`);
    return newUser;
  }

  async deleteUser(id: number): Promise<void> {
    const response = await deleteRequest(`${this.baseURL}/users/${id}`);
    expect(response.status(), { message: `DELETE /users/${id} should return 204 No Content` }).toBe(204);
    logger.info(`Deleted user ID: ${id}`);
  }

  async verifyUserExists(email: string) {
    const response = await getRequest(`${this.baseURL}/users?email=${email}`);
    await expect(response).toBeOK();
    const users = await response.json();
    expect(Array.isArray(users), { message: 'Users response should be an array' }).toBeTruthy();
    expect(users.length, { message: `At least one user with email ${email} should exist` }).toBeGreaterThan(0);
    logger.info(`Verified user exists: ${email}`);
  }
}
```

**Usage in specs:**

```typescript
import { test } from '@fixture';
import { UserAPI } from '@pages/api/user-api';

test.describe('User API @smoke', () => {
  test('should create and verify user', async ({ userAPI, userPage }) => {
    // Create user via API
    const newUser = await userAPI.createUser({
      name: 'John Doe',
      email: 'john@example.com',
    });

    // Verify in UI via page object — no raw utility calls in specs
    await userPage.verifyUserProfile(newUser.id, 'John Doe');

    // Cleanup via API
    await userAPI.deleteUser(newUser.id);
  });
});
```

## Sharing Cookies and Auth with Browser

API requests automatically use the same request context as the browser, so cookies and authentication persist:

```typescript
// tests/pages/auth-api-page.ts
import { clickAndNavigate, expect, fill, getRequest, gotoURL } from '@anaconda/playwright-utils';

export class AuthAPIPage {
  async loginAndVerifyProfile(): Promise<void> {
    // Login in UI — setPage is handled automatically by the fixture
    await gotoURL('https://example.com/login');
    await fill('#username', 'user');
    await fill('#password', 'pass');
    await clickAndNavigate('#login-button');

    // API requests now include auth cookies automatically
    const response = await getRequest('https://api.example.com/profile');
    await expect(response).toBeOK(); // Works because cookies are shared
  }
}
```

## Option Types

Options are typed as Playwright's native API request options:

```typescript
// For getRequest
Parameters < APIRequestContext['get'] > [1];

// For postRequest
Parameters < APIRequestContext['post'] > [1];

// For putRequest
Parameters < APIRequestContext['put'] > [1];

// For patchRequest
Parameters < APIRequestContext['patch'] > [1];

// For deleteRequest
Parameters < APIRequestContext['delete'] > [1];
```

See [Playwright APIRequestContext documentation](https://playwright.dev/docs/api/class-apirequestcontext) for complete option details including timeout, headers, auth, retry logic, and more.
