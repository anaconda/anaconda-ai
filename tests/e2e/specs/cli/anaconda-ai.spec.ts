import { test } from '@fixture';

test.describe('Anaconda AI CLI Commands @anaconda-ai', () => {
  // Test to verify that all models are listed correctly
  test('Install Anaconda AI package', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runInstallAiPackageCommand();
    anacondaAiCli.verifyInstallAiPackageCommand(result);
  });

});
