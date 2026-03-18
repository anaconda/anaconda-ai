// Base URL from .env (used for CLI commands e.g. sites modify --domain)
const baseUrl = process.env.BASE_URL ?? 'https://qa.anaconda-sandbox.com';
const baseDomain = new URL(baseUrl).hostname;

// Anaconda AI package CLI commands
// Command to install the Anaconda AI package using conda
export const installAiPackageCmd = `conda create -c defaults -c conda-forge anaconda-cloud/label/dev::anaconda-ai=0.5.0 python=3.12 notebook ipywidgets llm dspy -n anaconda-cli --y`;

// Verify the Anaconda AI package environment is runnable 
export const activateAiPackageEnvCmd = `conda run -n anaconda-cli --no-capture-output true`;

// Command to set AI Catalyst site as default (Self-Hosted site)
export const addAiPackageEnvToSandboxCmd = `conda run -n anaconda-cli --no-capture-output anaconda sites modify --domain ${baseDomain} --name self-hosted --default --yes`;

// Command to configure the Anaconda AI package environment to use the AI Catalyst backend
export const configureAiPackageEnvToUseSandboxCmd = `conda run -n anaconda-cli --no-capture-output anaconda ai config --backend ai-catalyst --yes`;

// Verify authentication
export const authWhoamiCmd = `conda run -n anaconda-cli --no-capture-output anaconda auth whoami`;

// Command to show Anaconda AI package help
export const anacondaAiHelpCmd = `conda run -n anaconda-cli --no-capture-output anaconda ai --help`;