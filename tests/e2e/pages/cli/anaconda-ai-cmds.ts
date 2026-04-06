import { shellCommand, verifyShellExitCode, type ShellResult } from 'tests/utils/CliUtils';
import * as cliCommands from './cliCommands';
import { expect } from 'tests/test-setup/page-setup';
import { ModelApi } from '@testdata/model-api';

// CLI helpers for Anaconda AI package CLI commands.
export class AnacondaAiCli {
  public async runAnacondaAiHelpCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.anacondaAiHelpCmd);
  }

  public verifyAnacondaAiHelpCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai --help');
  }

  public async runAnacondaAiModelsListCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.anacondaAiModelsListCmd);
  }

  public async runAnacondaAiBlockedModelsListCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.anacondaAiBlockedModelsListCmd);
  }

  // Verify the output of the `anaconda ai models --json` command
  public verifyAnacondaAiModelsListCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai models --json');

    const models = JSON.parse(result.output) as ModelApi[];
    expect(
      Array.isArray(models) && models.length,
      'models output should be a non-empty array',
    ).toBeGreaterThan(0);
    this.assertModelResponseData(models);
  }

  // Verify the output of `anaconda ai models --show-blocked --json` (array may be empty).
  public verifyAnacondaAiBlockedModelsListCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai models --show-blocked --json');

    const models = JSON.parse(result.output) as ModelApi[];
    expect(Array.isArray(models), 'blocked models output should be an array').toBe(true);
    this.assertModelResponseData(models);
  }

  private assertModelResponseData(models: ModelApi[]): void {
    models.forEach((model, modelIndex) => {
      expect(model.model, `models[${modelIndex}].model should be defined`).toBeDefined();
      expect(model.model.trim().length, `models[${modelIndex}].model should not be empty`).toBeGreaterThan(0);
      expect(model.parameters, `models[${modelIndex}].parameters should be defined`).toBeDefined();
      expect(model.quantizations, `models[${modelIndex}].quantizations should be an array`).toBeDefined();
      expect(model.quantizations.length, `models[${modelIndex}].quantizations should not be empty`).toBeGreaterThan(0);
      expect(model.trained_for, `models[${modelIndex}].trained_for should be defined`).toBeDefined();

      model.quantizations.forEach((quant, quantIndex) => {
        expect(quant.method, `models[${modelIndex}].quantizations[${quantIndex}].method should be defined`).toBeDefined();
        expect(quant.running, `models[${modelIndex}].quantizations[${quantIndex}].running should be defined`).not.toBeUndefined();
        expect(quant.downloaded, `models[${modelIndex}].quantizations[${quantIndex}].downloaded should be defined`).not.toBeUndefined();
        expect(quant.blocked, `models[${modelIndex}].quantizations[${quantIndex}].blocked should be defined`).not.toBeUndefined();
      });
    });
  }
}