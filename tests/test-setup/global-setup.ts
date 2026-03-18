/**
 * global-setup.ts: Sets up global state before all tests start.
 * Runs the Anaconda AI package install before any tests.
 */

import { AnacondaAiCli } from '../e2e/pages/cli/anaconda-ai';

export default async (): Promise<void> => {
  const anacondaAiCli = new AnacondaAiCli();
  // Install the Anaconda AI package
  const installResult = await anacondaAiCli.runInstallAiPackageCommand();
  anacondaAiCli.verifyInstallAiPackageCommand(installResult);

  // Activate the Anaconda AI package environment
  const activateResult = await anacondaAiCli.runActivateAiPackageEnvCommand();
  anacondaAiCli.verifyActivateAiPackageEnvCommand(activateResult);

  // Add the Anaconda AI package environment to the AI 
  const addResult = await anacondaAiCli.runAddAiPackageEnvToSandboxCommand();
  anacondaAiCli.verifyAddAiPackageEnvToSandboxCommand(addResult);

  // Configure the Anaconda AI package environment to use the AI Catalyst backend
  const configureResult = await anacondaAiCli.runConfigureAiPackageEnvToUseSandboxCommand();
  anacondaAiCli.verifyConfigureAiPackageEnvToUseSandboxCommand(configureResult);

  // Verify user is authenticated
  const authWhoamiResult = await anacondaAiCli.runAuthWhoamiCommand();
  anacondaAiCli.verifyAuthWhoamiCommand(authWhoamiResult);
};
