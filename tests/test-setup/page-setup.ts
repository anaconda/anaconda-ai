/**
 * page-setup.ts: Sets up the initial state of a page before each test.
 * Exports a base test object with a beforeEach hook for page context.
 */

import { test as baseTest, expect } from '@playwright/test';
import { setPage } from '@anaconda/playwright-utils';

export const test = baseTest.extend<{ testHook: void }>({
  testHook: [
    async ({ page }, use) => {
      setPage(page);
      await use();
    },
    { auto: true },
  ],
});

export { expect };
