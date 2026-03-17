
// Command to install the Anaconda AI package using conda
export const installAiPackageCmd = `conda create -c defaults -c conda-forge anaconda-cloud/label/dev::anaconda-ai=0.5.0rc python=3.12 notebook ipywidgets llm dspy -n anaconda-cli --y`;
