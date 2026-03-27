// Types for `anaconda ai models --json` CLI output
export type ModelQuantization = {
  method: string;
  running: boolean;
  downloaded: boolean;
  blocked: boolean;
};

export type Model = {
  model: string;
  parameters: number;
  quantizations: ModelQuantization[];
  trained_for: string;
};

