import * as ort from 'onnxruntime-web';
import { MODEL_CONFIG } from './model-config';

export interface PredictionResult {
  predicted: number;
  confidence: number;
  label: string;
  probabilities: number[];
}

let session: ort.InferenceSession | null = null;

export async function loadModel() {
  if (session) return;
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/";
    session = await ort.InferenceSession.create(MODEL_CONFIG.modelPath, {
      executionProviders: ['wasm']
    });
    console.log("ONNX Model loaded successfully.");
  } catch (e) {
    console.error("Failed to load ONNX model", e);
    throw e;
  }
}

function softmax(arr: Float32Array): number[] {
  const max = Math.max(...Array.from(arr));
  const exps = Array.from(arr).map(x => Math.exp(x - max));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sumExps);
}

export async function predict(data: Float32Array): Promise<PredictionResult> {
  if (!session) {
    throw new Error("Model is not loaded yet.");
  }

  const tensor = new ort.Tensor('float32', data, [1, 1, 28, 28]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;

  const results = await session.run(feeds);
  const outputData = results[session.outputNames[0]].data as Float32Array;

  const probabilities = softmax(outputData);
  
  let maxVal = -Infinity;
  let maxIdx = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxVal) {
      maxVal = probabilities[i];
      maxIdx = i;
    }
  }

  return {
    predicted: maxIdx,
    confidence: maxVal,
    label: MODEL_CONFIG.labels[maxIdx],
    probabilities: probabilities
  };
}
