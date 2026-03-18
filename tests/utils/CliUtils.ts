import { spawn } from 'child_process';

export interface ShellResult {
  exitCode: number;
  output: string;
  stderrOutput: string;
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
