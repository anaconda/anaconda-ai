import { test } from '@fixture';

test.describe('Anaconda AI CLI Commands @anaconda-ai', () => {
  test('anaconda ai --help', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiHelpCommand();
    anacondaAiCli.verifyAnacondaAiHelpCommand(result);
  });
  test('anaconda ai --help1', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiHelpCommand();
    anacondaAiCli.verifyAnacondaAiHelpCommand(result);
  });
  test('anaconda ai --help2', async ({ anacondaAiCli }) => {
    const result = await anacondaAiCli.runAnacondaAiHelpCommand();
    anacondaAiCli.verifyAnacondaAiHelpCommand(result);
  });

});
