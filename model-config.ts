/**
 * model-config.ts
 * ───────────────
 * Single source of truth for the KannadaCNN ONNX model configuration.
 * Keep this in sync with model_info.json produced by kannada_cnn_train.py
 */

export const MODEL_CONFIG = {
  /** Path to the ONNX file relative to /public (e.g. /public/models/) */
  modelPath: "/models/kannada_cnn.onnx",

  /** Model identity */
  modelName: "KannadaCNN",
  version: "1.0.0",

  /** Input tensor: single greyscale 28×28 image */
  inputName: "input",          // must match torch.onnx.export input_names
  outputName: "output",        // must match torch.onnx.export output_names

  /** Spatial dimensions */
  imageWidth: 28,
  imageHeight: 28,
  channels: 1,                 // greyscale

  /**
   * ONNX Runtime expects layout: NCHW  →  [batch, channels, height, width]
   * Shape for a single image inference: [1, 1, 28, 28]
   */
  inputShape: [1, 1, 28, 28] as const,

  /** Pixel normalisation applied BEFORE inference: pixel / 255 */
  normDivide: 255,
  normMean: 0.0,
  normStd: 1.0,

  /** Classification head */
  numClasses: 10,
  classLabels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] as const,

  /** Architecture info (informational) */
  totalParams: 296_650,
  trainedValAccuracy: 97.3,    // update after training

  /** ONNX export settings */
  onnxOpset: 17,
} as const;

export type ClassLabel = typeof MODEL_CONFIG.classLabels[number];
export type ClassIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;
