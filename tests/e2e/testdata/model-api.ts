// Types for `anaconda ai models --json` CLI output
export type ModelQuantization = {
  method: string;
  running: boolean;
  downloaded: boolean;
  blocked: boolean;
};

export type ModelApi = {
  model: string;
  parameters: number;
  quantizations: ModelQuantization[];
  trained_for: string;
};

// Data to verify the Download Model Command
export const DOWNLOAD_TEST_MODEL_NAME = 'bge-base-en-v1.5';
export const DOWNLOAD_TEST_MODEL_QUANTIZATION = 'q4_k_m';

