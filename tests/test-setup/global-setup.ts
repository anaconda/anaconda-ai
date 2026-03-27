/**
 * global-setup.ts: Sets up global state before all tests start.
 * Runs the Anaconda AI package install before any tests.
 */
import { AnacondaAiSetupCli } from '../e2e/pages/cli/anaconda-ai-setup';
import { baseDomain, SELF_HOSTED_SITE_NAME } from '@testdata/site-data';

async function setupAnacondaAi(cli: AnacondaAiSetupCli): Promise<void> {
  console.log('Running global setup…');

  // Install the Anaconda AI package
  const installResult = await cli.runInstallAiPackageCommand();
  cli.verifyInstallAiPackageCommand(installResult);

  // Activate the Anaconda AI package environment
  const activateResult = await cli.runActivateAiPackageEnvCommand();
  cli.verifyActivateAiPackageEnvCommand(activateResult);

  // List sites: if site exists run modify, otherwise run add
  const sitesListResult = await cli.runSitesListCommand();
  cli.verifySitesListCommand(sitesListResult);

  if (cli.isSiteNameListed(sitesListResult, SELF_HOSTED_SITE_NAME)) {
    const modifyResult = await cli.runModifySiteCommand(baseDomain, SELF_HOSTED_SITE_NAME);
    cli.verifyModifySiteCommand(modifyResult);
  } else {
    const addResult = await cli.runAddSiteCommand(baseDomain, SELF_HOSTED_SITE_NAME);
    cli.verifyAddSiteCommand(addResult);
  }

  // Configure the Anaconda AI package environment to use the AI Catalyst backend
  const configureResult = await cli.runConfigureAiPackageEnvToUseSandboxCommand();
  cli.verifyConfigureAiPackageEnvToUseSandboxCommand(configureResult);

  // Verify user is authenticated
  const authWhoamiResult = await cli.runAuthWhoamiCommand();
  cli.verifyAuthWhoamiCommand(authWhoamiResult);
}

export default async (): Promise<void> => {
  try {
    await setupAnacondaAi(new AnacondaAiSetupCli());
  } catch (e) {
    throw new Error(`Global setup failed: ${e instanceof Error ? e.message : e}`, { cause: e });
  }
};
