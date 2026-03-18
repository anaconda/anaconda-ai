/**
 * global-setup.ts: Sets up global state before all tests start.
 * Runs the Anaconda AI package install before any tests.
 */

import { AnacondaAiCli } from '../e2e/pages/cli/anaconda-ai';

async function setupAnacondaAi(cli: AnacondaAiCli): Promise<void> {
  // Install the Anaconda AI package
  const installResult = await cli.runInstallAiPackageCommand();
  cli.verifyInstallAiPackageCommand(installResult);

  // Activate the Anaconda AI package environment
  const activateResult = await cli.runActivateAiPackageEnvCommand();
  cli.verifyActivateAiPackageEnvCommand(activateResult);

  // Add the Self-Hosted site as the default site
  const addResult = await cli.runAddAiPackageEnvToSandboxCommand();
  cli.verifyAddAiPackageEnvToSandboxCommand(addResult);

  // Configure the Anaconda AI package environment to use the AI Catalyst backend
  const configureResult = await cli.runConfigureAiPackageEnvToUseSandboxCommand();
  cli.verifyConfigureAiPackageEnvToUseSandboxCommand(configureResult);

  // Verify user is authenticated
  const authWhoamiResult = await cli.runAuthWhoamiCommand();
  cli.verifyAuthWhoamiCommand(authWhoamiResult);
}

export default async (): Promise<void> => {
  try {
    await setupAnacondaAi(new AnacondaAiCli());
  } catch (e) {
    throw new Error(`Global setup failed: ${e instanceof Error ? e.message : e}`, { cause: e });
  }
};
