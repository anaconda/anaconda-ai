import { expect } from '@playwright/test';
import { shellCommand, type ShellResult } from 'tests/utils/CliUtils';

import * as cliCommands from './cliCommands';

export class AnacondaAiCli {
  private verifyExitCode(result: ShellResult, commandName: string): void {
    if (result.exitCode !== 0) {
      console.log(`${commandName} stderr:`, result.stderrOutput || result.output);
    }
    expect(
      result.exitCode,
      `Expected ${commandName} to exit with code 0, but got ${result.exitCode}`,
    ).toBe(0);
  }

  // Runs CLI command to install the package
  public async runInstallAiPackageCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.installAiPackageCmd);
  }

  // Verifies expected install output
  public verifyInstallAiPackageCommand(result: ShellResult) {
    this.verifyExitCode(result, 'install command');
    const output = result.output.toLowerCase();
    const isInstalledNow =
      output.includes('executing transaction') &&
      output.includes('anaconda-ai');
    const isAlreadyInstalled = output.includes(
      'all requested packages already installed',
    );

    expect(
      isInstalledNow || isAlreadyInstalled,
      'Expected anaconda-ai to be either newly installed or already installed',
    ).toBeTruthy();
  }

  // Runs CLI command to activate the Anaconda AI package environment
  public async runActivateAiPackageEnvCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.activateAiPackageEnvCmd);
  }

  // Verifies expected activate output
  public verifyActivateAiPackageEnvCommand(result: ShellResult) {
    this.verifyExitCode(result, 'activate command');
  }

  // Runs CLI command to add the Anaconda AI package environment to the Anaconda Sandbox
  public async runAddAiPackageEnvToSandboxCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.addAiPackageEnvToSandboxCmd);
  }

  // Verifies expected add output
  public verifyAddAiPackageEnvToSandboxCommand(result: ShellResult) {
    this.verifyExitCode(result, 'add site command');
  }

  // Runs CLI command to configure the Anaconda AI package environment to use the Anaconda Sandbox backend
  public async runConfigureAiPackageEnvToUseSandboxCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.configureAiPackageEnvToUseSandboxCmd);
  }

  // Verifies expected configure output
  public verifyConfigureAiPackageEnvToUseSandboxCommand(result: ShellResult) {
    this.verifyExitCode(result, 'configure command');
  }

  // Runs CLI command to verify auth (ANACONDA_AUTH_API_KEY from env via CliUtils)
  public async runAuthWhoamiCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.authWhoamiCmd);
  }

  public verifyAuthWhoamiCommand(result: ShellResult) {
    this.verifyExitCode(result, 'auth whoami');
  }

  // Runs CLI command: anaconda ai --help
  public async runAnacondaAiHelpCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.anacondaAiHelpCmd);
  }

  // Verifies expected anaconda ai --help output
  public verifyAnacondaAiHelpCommand(result: ShellResult) {
    this.verifyExitCode(result, 'anaconda ai --help');
  }
}
