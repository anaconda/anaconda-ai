import { expect } from '@playwright/test';
import { type ShellResult, shellCommand, verifyShellExitCode } from 'tests/utils/CliUtils';

import * as cliCommands from './cliCommands';

/** CLI helpers used only in global-setup (env install, sites, configure, auth). */
export class AnacondaAiSetupCli {
  public async runInstallAiPackageCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.installAiPackageCmd);
  }

  public verifyInstallAiPackageCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'install command');
    const output = result.output.toLowerCase();
    const isInstalledNow = output.includes('executing transaction') && output.includes('anaconda-ai');
    const isAlreadyInstalled = output.includes('all requested packages already installed');

    expect(
      isInstalledNow || isAlreadyInstalled,
      'Expected anaconda-ai to be either newly installed or already installed',
    ).toBe(true);
  }

  public async runActivateAiPackageEnvCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.activateAiPackageEnvCmd);
  }

  public verifyActivateAiPackageEnvCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'conda list anaconda-ai');
  }

  public async runSitesListCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.sitesListCmd);
  }

  public verifySitesListCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'sites list command');
  }

  // True if `anaconda sites list` output contains the given site name.
  public isSiteNameListed(listResult: ShellResult, name: string): boolean {
    const output = `${listResult.output}\n${listResult.stderrOutput}`;
    return output.includes(`│ ${name} `);
  }

  public async runAddSiteCommand(domain: string, name: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.addSiteCmd(domain, name));
  }

  public verifyAddSiteCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'add site command');
  }

  public async runModifySiteCommand(domain: string, name: string): Promise<ShellResult> {
    return await shellCommand(cliCommands.modifySiteCmd(domain, name));
  }

  public verifyModifySiteCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'modify site command');
  }

  public async runConfigureAiPackageEnvToUseSandboxCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.configureAiPackageEnvToUseSandboxCmd);
  }

  public verifyConfigureAiPackageEnvToUseSandboxCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'configure command');
  }

  public async runAuthWhoamiCommand(): Promise<ShellResult> {
    return await shellCommand(cliCommands.authWhoamiCmd);
  }

  public verifyAuthWhoamiCommand(result: ShellResult): void {
    verifyShellExitCode(result, 'auth whoami');
  }
}
