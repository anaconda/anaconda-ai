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

