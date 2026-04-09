import { test } from '@fixture';
import { DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION } from '@testdata/model-api';

test.describe('Anaconda AI CLI Commands @anaconda-ai', () => {
  test('anaconda ai --help', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiHelpCommand();
    anacondaAiCli.verifyAnacondaAiHelpCommand(result);
  });
  test('anaconda ai models list command', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiModelsListCommand();
    anacondaAiCli.verifyAnacondaAiModelsListCommand(result);
  });

  test('anaconda ai models list blocked command', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiBlockedModelsListCommand();
    anacondaAiCli.verifyAnacondaAiBlockedModelsListCommand(result);
  });

  test('anaconda ai download model command', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runDownloadModelCommand(
      DOWNLOAD_TEST_MODEL_NAME,
      DOWNLOAD_TEST_MODEL_QUANTIZATION,
    );
    anacondaAiCli.verifyDownloadModelCommand(result);
  });
});
