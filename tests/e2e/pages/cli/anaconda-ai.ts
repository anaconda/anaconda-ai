import { expect } from '@playwright/test';
import { shellCommand, type ShellResult } from 'tests/utils/CliUtils';

import {
  installAiPackageCmd,
} from './cliCommands';

export class AnacondaAiCli {
  // Runs CLI command to install the package
  public async runInstallAiPackageCommand(): Promise<ShellResult> {
    return await shellCommand(installAiPackageCmd);
  }

  // Verifies expected install output
  public verifyInstallAiPackageCommand(result: ShellResult) {
    expect(
      result.exitCode,
      `Expected install command to exit with code 0, but got ${result.exitCode}`,
    ).toBe(0);

    const output = result.output.toLowerCase();
    const isInstalledNow =
      output.includes('executing transaction') &&
      output.includes('anaconda-ai');
    const isAlreadyInstalled = output.includes(
      'all requested packages already installed',
    );

    // Soft assertion: Model should be either installed now or already installed
    expect
      .soft(
        isInstalledNow || isAlreadyInstalled,
        'Expected anaconda-ai to be either newly installed or already installed',
      )
      .toBeTruthy();
  }

}
