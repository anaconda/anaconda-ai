import { test as baseTest } from '@anaconda/playwright-utils';
import { CLI } from '@pages/cli/cli';

export const test = baseTest.extend<{
  cli: CLI;
}>({
  cli: async ({}, use) => {
    await use(new CLI());
  },
});

export const expect = test.expect;
