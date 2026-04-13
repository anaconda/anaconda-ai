// Types for `anaconda ai models --json` CLI output
export type ModelQuantization = {
  method: string;
  running: boolean;
  downloaded: boolean;
  blocked: boolean;
};

// Types for `anaconda ai models --json` CLI output
export type ModelApi = {
  model: string;
  parameters: number;
  quantizations: ModelQuantization[];
  trained_for: string;
};

// Types for `anaconda ai servers --json` CLI output
export type ServerApi = {
  server_id: string;
  model: string;
  status: string;
};

// Data to verify the Download Model Command
export const DOWNLOAD_TEST_MODEL_NAME = 'bge-base-en-v1.5';
export const DOWNLOAD_TEST_MODEL_QUANTIZATION = 'q4_k_m';

// invalid model data
export const INVALID_MODEL_NAME = 'invalid-model';
export const INVALID_MODEL_QUANTIZATION = 'invalid-quantization';
export const INVALID_MODEL_ERROR_MESSAGE = 'you must include the quantization method in the model';

// Invalid server name for negative stop tests
export const INVALID_SERVER_NAME = 'invalid-server';
