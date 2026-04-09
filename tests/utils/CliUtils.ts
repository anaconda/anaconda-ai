import { spawn } from 'child_process';

import { expect } from '@playwright/test';
import { logger } from '@anaconda/playwright-utils';

export interface ShellResult {
  exitCode: number;
  output: string;
  stderrOutput: string;
}

// Strip ANSI SGR color sequences (ESC [ … m) and trim — use before parsing CLI stdout/stderr.
export function stripAnsiSgrAndTrim(output: string): string {
  return output.replace(/\x1B\[[0-9;]*m/g, '').trim();
}

/** Asserts a shell command exited 0; logs stderr/output on failure. */
export function verifyShellExitCode(result: ShellResult, commandName: string): void {
  if (result.exitCode !== 0) {
    logger.error(`${commandName} stderr:`, result.stderrOutput || result.output);
  }
  expect(
    result.exitCode,
    `Expected ${commandName} to exit with code 0, but got ${result.exitCode}`,
  ).toBe(0);
}

export const shellCommand = async (input: string) => {
  const commandToExecute = input;
  // Initialize output variables
  let output = '';
  let stderrOutput = '';

  // Spawn with env so Anaconda CLI sees ANACONDA_AUTH_API_KEY 
  const env = {
    ...process.env,
    ANACONDA_AUTH_API_KEY:
      process.env.ANACONDA_AUTH_API_KEY ?? '',
  };
  const commandProcess = spawn(commandToExecute, { shell: true, env });

  commandProcess.stdout.on('data', (data: string) => {
    output += data;
  });

  commandProcess.stderr.on('data', (data: string) => {
    // Store stderr data separately, don't overwrite stdout
    const chunk = data;
    stderrOutput += chunk;
  });

  const exitCode = await new Promise<number>((resolve) => {
    commandProcess.on('close', (code) => {
      resolve(typeof code === 'number' ? code : 1); // Provide fallback for undefined/null
    });
  });

  return { output, exitCode, stderrOutput };
};
