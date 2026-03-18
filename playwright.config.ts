/**
 * playwright.config.ts: Configures the Playwright test runner for CLI/e2e tests.
 * See https://playwright.dev/docs/test-configuration for more details.
 */

import { AnacondaConfigDefaults, AnacondaProjectDefaults } from '@anaconda/playwright-utils';
import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: '.env' });

// Base URL from .env (align with cliCommands default)
export const BASE_URL = process.env.BASE_URL ?? process.env.URL ?? 'https://qa.anaconda-sandbox.com';
export const STORAGE_STATE_PATH = path.join(__dirname, 'playwright/.auth');
const bearerToken = process.env.BEARER_TOKEN ?? '';

export default defineConfig({
  ...AnacondaConfigDefaults,
  globalSetup: './tests/test-setup/global-setup.ts',
  testDir: './tests/e2e',
  use: {
    ...AnacondaProjectDefaults,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    baseURL: BASE_URL,
    extraHTTPHeaders: {
      Authorization: `Bearer ${bearerToken}`,
    },
  },

  projects: [
    {
      name: 'chromium',
      use: {
        viewport: null,
        launchOptions: {
          args: ['--disable-web-security', '--start-maximized'],
          slowMo: 0,
          headless: false,
        },
      },
    },
    {
      name: 'chromiumheadless',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1600, height: 1000 },
        launchOptions: {
          args: ['--disable-web-security'],
          slowMo: 0,
          headless: true,
        },
      },
    },
  ],
});
