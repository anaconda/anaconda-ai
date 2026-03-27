// Anaconda AI package CLI commands
// Command to install the Anaconda AI package using conda
export const installAiPackageCmd = `conda create -c defaults -c conda-forge anaconda-cloud/label/dev::anaconda-ai=0.5.0 python=3.12 notebook ipywidgets llm dspy -n anaconda-cli --y`;

// Verify the Anaconda AI package environment is runnable 
export const activateAiPackageEnvCmd = `conda run -n anaconda-cli --no-capture-output true`;

// Command to list sites
export const sitesListCmd = `conda run -n anaconda-cli --no-capture-output anaconda sites list`;

// Command to add a site
export const addSiteCmd = (domain: string, name: string): string => {
  return `conda run -n anaconda-cli --no-capture-output anaconda sites add --domain ${domain} --name ${name} --default --yes`;
};

// Command to modify a site
export const modifySiteCmd = (domain: string, name: string): string => {
  return `conda run -n anaconda-cli --no-capture-output anaconda sites modify --domain ${domain} --name ${name} --default --yes`;
};

// Command to configure the Anaconda AI package environment to use the AI Catalyst backend
export const configureAiPackageEnvToUseSandboxCmd = `conda run -n anaconda-cli --no-capture-output anaconda ai config --backend ai-catalyst --yes`;

// Verify authentication
export const authWhoamiCmd = `conda run -n anaconda-cli --no-capture-output anaconda auth whoami`;

// Command to show Anaconda AI package help
export const anacondaAiHelpCmd = `conda run -n anaconda-cli --no-capture-output anaconda ai --help`;

// Command to list Anaconda AI models
export const anacondaAiModelsListCmd = `conda run -n anaconda-cli --no-capture-output anaconda ai models --json`;
