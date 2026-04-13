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

  test('anaconda ai server lifecycle: launch, verify, stop, and delete a model server', async ({ anacondaAiCli }) => {
    await test.step('step 1: launch model server and verify it is running', async () => {
      const launchResult = await anacondaAiCli.runLaunchModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyLaunchModelCommand(launchResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });

    await test.step('step 2: verify server appears in servers list with status "running"', async () => {
      const serversResult = await anacondaAiCli.runAnacondaAiServersListCommand();
      anacondaAiCli.verifyRunningServerInList(serversResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });

    await test.step('step 3: launching the same server again returns AnacondaAIException', async () => {
      const duplicateResult = await anacondaAiCli.runLaunchModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyDuplicateLaunchModelCommand(duplicateResult);
    });

    await test.step('step 4: stop the model server and verify it is stopped', async () => {
      const stopResult = await anacondaAiCli.runStopModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyStopModelCommand(stopResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });

    await test.step('step 5: delete the model server and verify it is removed', async () => {
      const deleteResult = await anacondaAiCli.runStopAndRemoveModelCommand(
        DOWNLOAD_TEST_MODEL_NAME,
        DOWNLOAD_TEST_MODEL_QUANTIZATION,
      );
      anacondaAiCli.verifyStopAndRemoveModelCommand(deleteResult, DOWNLOAD_TEST_MODEL_NAME, DOWNLOAD_TEST_MODEL_QUANTIZATION);
    });
  });

  test('anaconda ai launch - invalid format returns ValueError', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runLaunchModelCommand(
      INVALID_MODEL_NAME,
      INVALID_MODEL_QUANTIZATION,
    );
    anacondaAiCli.verifyLaunchModelInvalidFormatCommand(result);
  });

  test('anaconda ai launch - unknown model returns ModelNotFound', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runLaunchModelCommand(
      INVALID_MODEL_NAME,
      DOWNLOAD_TEST_MODEL_QUANTIZATION,
    );
    anacondaAiCli.verifyLaunchModelNotFoundCommand(result);
  });
});
