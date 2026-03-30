import { test as baseTest } from '@anaconda/playwright-utils';
import { AnacondaAiCli } from '@pages/cli/anaconda-ai-cmds';

export const test = baseTest.extend<{
  anacondaAiCli: AnacondaAiCli;
}>({
  anacondaAiCli: async ({ }, use) => {
    await use(new AnacondaAiCli());
  },
});

export const expect = test.expect;
