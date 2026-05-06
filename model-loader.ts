/**
 * model-loader.ts
 * ───────────────
 * Loads the KannadaCNN ONNX model via ONNX Runtime Web and runs inference.
 *
 * Prerequisites (in your project):
 *   npm install onnxruntime-web
 *
 * In your HTML / Vite config, copy the ONNX Runtime WASM files to /public:
 *   node_modules/onnxruntime-web/dist/*.wasm  →  /public/
 *
 * Place the model file at:
 *   /public/models/kannada_cnn.onnx
 */

import * as ort from "onnxruntime-web";
import { MODEL_CONFIG, type ClassIndex, type ClassLabel } from "./model-config";
import { preprocessCanvas } from "./preprocessing";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface PredictionResult {
  /** Predicted class index 0-9 */
  classIndex: ClassIndex;
  /** Human-readable Kannada digit label */
  label: ClassLabel;
  /** Confidence for the predicted class [0, 1] */
  confidence: number;
  /** Softmax probabilities for all 10 classes */
  probabilities: number[];
  /** Raw logits from the model */
  logits: number[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Session management (singleton)
// ─────────────────────────────────────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;
let _loadPromise: Promise<ort.InferenceSession> | null = null;

/**
 * Load (or return cached) ONNX inference session.
 * Call this once on app startup to avoid cold-start latency at inference time.
 */
export async function loadModel(): Promise<ort.InferenceSession> {
  if (_session) return _session;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    // Prefer WebGL backend for GPU acceleration in-browser, fall back to WASM
    ort.env.wasm.wasmPaths = "/";          // serve .wasm files from /public/
    ort.env.wasm.numThreads = 1;           // safest default for cross-origin

    const session = await ort.InferenceSession.create(MODEL_CONFIG.modelPath, {
      executionProviders: ["webgl", "wasm"],
      graphOptimizationLevel: "all",
    });

    console.log(
      `[model-loader] KannadaCNN loaded ✓  ` +
      `(inputs: ${session.inputNames}, outputs: ${session.outputNames})`
    );

    _session = session;
    return session;
  })();

  return _loadPromise;
}

/** Release the ONNX session and free GPU/WASM memory. */
export async function unloadModel(): Promise<void> {
  if (_session) {
    await _session.release();
    _session    = null;
    _loadPromise = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run the model on a canvas element drawn by the user.
 *
 * @param canvas  HTMLCanvasElement containing the user's drawing.
 * @returns       PredictionResult with class, confidence, and full probabilities.
 */
export async function predict(canvas: HTMLCanvasElement): Promise<PredictionResult> {
  const session = await loadModel();

  // 1. Preprocess canvas → Float32Array [1, 1, 28, 28]
  const inputData  = preprocessCanvas(canvas);
  const inputShape = MODEL_CONFIG.inputShape;          // [1, 1, 28, 28]

  // 2. Build ORT tensor
  const tensor = new ort.Tensor("float32", inputData, [...inputShape]);
  const feeds  = { [MODEL_CONFIG.inputName]: tensor };

  // 3. Run inference
  const results = await session.run(feeds);
  const logits  = Array.from(results[MODEL_CONFIG.outputName].data as Float32Array);

  // 4. Softmax
  const probabilities = softmax(logits);

  // 5. Argmax
  const classIndex = argmax(probabilities) as ClassIndex;
  const label      = MODEL_CONFIG.classLabels[classIndex];
  const confidence = probabilities[classIndex];

  return { classIndex, label, confidence, probabilities, logits };
}

/**
 * Convenience: run inference and return only the top-N predictions.
 */
export async function predictTopN(
  canvas: HTMLCanvasElement,
  n = 3
): Promise<Array<{ classIndex: ClassIndex; label: ClassLabel; confidence: number }>> {
  const result = await predict(canvas);

  return result.probabilities
    .map((conf, idx) => ({
      classIndex: idx as ClassIndex,
      label: MODEL_CONFIG.classLabels[idx],
      confidence: conf,
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps     = logits.map((x) => Math.exp(x - maxLogit));
  const sum      = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function argmax(arr: number[]): number {
  return arr.reduce((best, val, idx, a) => (val > a[best] ? idx : best), 0);
}
