import { shellCommand, stripAnsiSgrAndTrim, verifyShellExitCode, type ShellResult } from 'tests/utils/CliUtils';
import * as cliCommands from './cliCommands';
import { expect } from 'tests/test-setup/page-setup';
import { INVALID_MODEL_ERROR_MESSAGE, ModelApi, ServerApi } from '@testdata/model-api';

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

    const models = JSON.parse(stripAnsiSgrAndTrim(result.output)) as ModelApi[];
    expect(
      Array.isArray(models) && models.length,
      'models output should be a non-empty array',
    ).toBeGreaterThan(0);
    this.assertModelResponseData(models);
  }

  // Verify the output of `anaconda ai models --show-blocked --json` (array may be empty).
  public verifyAnacondaAiBlockedModelsListCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai models --show-blocked --json');

    const models = JSON.parse(stripAnsiSgrAndTrim(result.output)) as ModelApi[];
    expect(Array.isArray(models), 'blocked models output should be an array').toBe(true);
    this.assertModelResponseData(models);
  }

  public async runAnacondaAiServersListCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.anacondaAiServersListCmd);
  }

  public verifyAnacondaAiServersListCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai servers --json');

    const servers = JSON.parse(stripAnsiSgrAndTrim(result.output)) as ServerApi[];
    expect(Array.isArray(servers), 'servers output should be an array').toBe(true);
  }

  // Verifies that a server for the given model appears in the servers list with status "running"
  public verifyRunningServerInList(result: ShellResult, modelName: string, modelQuantization: string): void {
    verifyShellExitCode(result, 'anaconda ai servers --json');

    const servers = JSON.parse(stripAnsiSgrAndTrim(result.output)) as ServerApi[];
    const expectedModelFile = `${modelName}_${modelQuantization}.gguf`;
    const server = servers.find((s) => s.model === expectedModelFile);

    expect(server, `Expected server for model "${expectedModelFile}" to appear in servers list`).toBeDefined();
    expect(server!.server_id, `Expected server_id to contain model name "${modelName}"`).toContain(modelName);
    expect(server!.model, `Expected model to be "${expectedModelFile}"`).toBe(expectedModelFile);
    expect(server!.status, `Expected server status to be "running"`).toBe('running');
  }

  // Executes `anaconda ai download <model>/<quant>` (positional; no --model)
  public async runDownloadModelCommand(modelName: string, modelQuantization: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.downloadModelCmd(modelName, modelQuantization));
  }

  // Validates model download started or completed successfully
  public verifyDownloadModelCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'anaconda ai download <model>/<quant>');

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    const isDownloadedNow =
      output.includes('downloading') &&
      (output.includes('mb') || output.includes('.gguf')) &&
      output.includes('success');
    const isAlreadyDownloaded = output.includes('success');

    expect(
      isDownloadedNow || isAlreadyDownloaded,
      'Expected the model to be either newly downloaded or already downloaded',
    ).toBeTruthy();
  }

  public verifyInvalidDownloadModelCommand(result: ShellResult): void {
    expect(
      result.exitCode,
      `Expected invalid download command to fail, but got exit code ${result.exitCode}`,
    ).not.toBe(0);
    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    expect(output, 'Expected the model to be invalid').toContain('error');
    expect(output, 'Expected output should contain invalid model error message').toContain(INVALID_MODEL_ERROR_MESSAGE);
  }

  // Executes `anaconda ai launch <model>/<quant> --detach`
  public async runLaunchModelCommand(modelName: string, modelQuantization: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.launchModelCmd(modelName, modelQuantization));
  }

  // Validates that launching a model that already has a running server fails with AnacondaAIException
  public verifyDuplicateLaunchModelCommand(result: ShellResult): void {
    expect(result.exitCode, 'Expected non-zero exit code for duplicate launch').not.toBe(0);

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    expect(
      output.includes('anacondaaiexception') || output.includes('already exists'),
      'Expected AnacondaAIException about duplicate server',
    ).toBeTruthy();
  }

  // Validates model server launched successfully
  public verifyLaunchModelCommand(result: ShellResult, modelName: string, modelQuantization: string): void {
    verifyShellExitCode(result, 'anaconda ai launch <model>/<quant>');

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    const expectedModelFile = `${modelName}_${modelQuantization}.gguf`.toLowerCase();
    expect(output.includes('running'), 'Expected status to be "running"').toBeTruthy();
    expect(
      output.includes('inference/serve/'),
      'Expected "inference/serve/" is running',
    ).toBeTruthy();
    expect(
      output.includes(expectedModelFile),
      `Expected model "${expectedModelFile}" to appear in output`,
    ).toBeTruthy();
  }

  // Executes `anaconda ai stop <modelName>_<modelQuantization>.gguf`
  public async runStopModelCommand(modelName: string, modelQuantization: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.stopModelCmd(modelName, modelQuantization));
  }

  // Validates model server stopped successfully
  public verifyStopModelCommand(result: ShellResult, modelName: string, modelQuantization: string): void {
    verifyShellExitCode(result, 'anaconda ai stop <model>.gguf');

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    const expectedModelFile = `${modelName}_${modelQuantization}.gguf`.toLowerCase();
    expect(output.includes('stopped'), 'Expected status to be "stopped"').toBeTruthy();
    expect(output.includes('success'), 'Expected "Success" in output').toBeTruthy();
    expect(
      output.includes(expectedModelFile),
      `Expected model "${expectedModelFile}" to appear in output`,
    ).toBeTruthy();
  }

  // Executes `anaconda ai stop <modelName>_<modelQuantization>.gguf --rm`
  public async runStopAndRemoveModelCommand(modelName: string, modelQuantization: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.stopAndRemoveModelCmd(modelName, modelQuantization));
  }

  // Validates model server deleted successfully
  public verifyStopAndRemoveModelCommand(result: ShellResult, modelName: string, modelQuantization: string): void {
    verifyShellExitCode(result, 'anaconda ai stop <model>.gguf --rm');

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    const expectedModelFile = `${modelName}_${modelQuantization}.gguf`.toLowerCase();
    expect(output.includes('deleted'), 'Expected status to be "deleted"').toBeTruthy();
    expect(output.includes('success'), 'Expected "Success" in output').toBeTruthy();
    expect(
      output.includes(expectedModelFile),
      `Expected model "${expectedModelFile}" to appear in output`,
    ).toBeTruthy();
  }

  // Validates that launching with an invalid format exits non-zero with a ValueError
  public verifyLaunchModelInvalidFormatCommand(result: ShellResult): void {
    expect(result.exitCode, 'Expected non-zero exit code for invalid format').not.toBe(0);

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    expect(
      output.includes('valueerror') && output.includes('does not look like a quantized model name'),
      'Expected ValueError about invalid model name format',
    ).toBeTruthy();
  }

  // Validates that launching an unknown model exits non-zero with a ModelNotFound error
  public verifyLaunchModelNotFoundCommand(result: ShellResult): void {
    expect(result.exitCode, 'Expected non-zero exit code for unknown model').not.toBe(0);

    const output = stripAnsiSgrAndTrim(result.output).toLowerCase();
    expect(
      output.includes('modelnotfound') || output.includes('was not found'),
      'Expected ModelNotFound error for unknown model',
    ).toBeTruthy();
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
