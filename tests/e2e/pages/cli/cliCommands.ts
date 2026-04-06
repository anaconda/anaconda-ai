// Anaconda AI package CLI commands

const anacondaAiChannel = process.env.ANACONDA_AI_CHANNEL ?? 'anaconda-cloud/label/dev';
const anacondaAiVersion = process.env.ANACONDA_AI_VERSION ?? '0.5.0';

// Run all command in the anaconda-cli environment
const condaRun = (inner: string): string =>
  `conda run -n anaconda-cli --no-capture-output ${inner}`;

export const installAiPackageCmd =
  `conda create -c defaults -c conda-forge ${anacondaAiChannel}::anaconda-ai=${anacondaAiVersion} -n anaconda-cli --yes`;

// Verify the Anaconda AI package environment is runnable
export const activateAiPackageEnvCmd = condaRun('conda list anaconda-ai');

export const sitesListCmd = condaRun('anaconda sites list');

export const addSiteCmd = (domain: string, name: string): string =>
  condaRun(`anaconda sites add --domain ${domain} --name ${name} --default --yes`);

export const modifySiteCmd = (domain: string, name: string): string =>
  condaRun(`anaconda sites modify --domain ${domain} --name ${name} --default --yes`);

export const configureAiPackageEnvToUseSandboxCmd = condaRun(
  'anaconda ai config --backend ai-catalyst --yes',
);

export const authWhoamiCmd = condaRun('anaconda auth whoami');

export const anacondaAiHelpCmd = condaRun('anaconda ai --help');

export const anacondaAiModelsListCmd = condaRun('anaconda ai models --json');

export const anacondaAiBlockedModelsListCmd = condaRun(
  'anaconda ai models --show-blocked --json',
);
