import { test } from '@fixture';
import {
  DOWNLOAD_TEST_MODEL_NAME,
  DOWNLOAD_TEST_MODEL_QUANTIZATION,
  INVALID_MODEL_NAME,
  INVALID_MODEL_QUANTIZATION,
} from '@testdata/model-api';

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

  test('anaconda ai download invalid model command', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runDownloadModelCommand(
      INVALID_MODEL_NAME,
      INVALID_MODEL_QUANTIZATION,
    );
    anacondaAiCli.verifyInvalidDownloadModelCommand(result);
  });

  test('anaconda ai servers list command', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiServersListCommand();
    anacondaAiCli.verifyAnacondaAiServersListCommand(result);
  });

  test('anaconda ai launch, stop model command', async ({ anacondaAiCli }) => {
    await test.step('launch model server', async () => {
      const result = await anacondaAiCli.runLaunchModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyLaunchModelCommand(result, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });

    await test.step('stop model server', async () => {
      const stopResult = await anacondaAiCli.runStopModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyStopModelCommand(stopResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });

    await test.step('delete model server', async () => {
      const deleteResult = await anacondaAiCli.runStopAndRemoveModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyStopAndRemoveModelCommand(deleteResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });
  });

  test('anaconda ai launch - invalid format returns ValueError', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runLaunchModelInvalidFormatCommand();
    anacondaAiCli.verifyLaunchModelInvalidFormatCommand(result);
  });

  test('anaconda ai launch - unknown model returns ModelNotFound', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runLaunchModelNotFoundCommand();
    anacondaAiCli.verifyLaunchModelNotFoundCommand(result);
  });
});
