/**
 * preprocessing.ts
 * ────────────────
 * Converts a raw canvas / ImageData into a Float32Array ready to feed
 * into the ONNX model.  Mirrors the exact pipeline used during training:
 *
 *   1. Resize to 28 × 28 (nearest-neighbour via OffscreenCanvas)
 *   2. Convert to greyscale
 *   3. Normalise: pixel / 255  →  [0, 1]
 *   4. Layout: NCHW  →  Float32Array of length 784 (1 × 1 × 28 × 28)
 */

import { MODEL_CONFIG } from "./model-config";

const { imageWidth, imageHeight, channels, normDivide } = MODEL_CONFIG;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Preprocess an HTMLCanvasElement drawn by the user.
 *
 * @param canvas  The canvas the user drew on (any size).
 * @returns       Float32Array of shape [1, 1, 28, 28] suitable for ORT.
 */
export function preprocessCanvas(canvas: HTMLCanvasElement): Float32Array {
  const imageData = getResizedImageData(canvas, imageWidth, imageHeight);
  return imageDataToTensor(imageData);
}

/**
 * Preprocess a raw ImageData object (e.g. from a webcam frame).
 * The ImageData MUST already be 28 × 28; use resizeImageData if not.
 */
export function preprocessImageData(imageData: ImageData): Float32Array {
  if (imageData.width !== imageWidth || imageData.height !== imageHeight) {
    throw new Error(
      `Expected ${imageWidth}×${imageHeight} ImageData, got ${imageData.width}×${imageData.height}. ` +
      `Use resizeImageData() first.`
    );
  }
  return imageDataToTensor(imageData);
}

/**
 * Resize any ImageData to 28 × 28 using a temporary canvas.
 */
export function resizeImageData(
  source: ImageData,
  targetW = imageWidth,
  targetH = imageHeight
): ImageData {
  // Draw source onto a temp canvas then scale it
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width  = source.width;
  srcCanvas.height = source.height;
  srcCanvas.getContext("2d")!.putImageData(source, 0, 0);

  const dstCanvas = document.createElement("canvas");
  dstCanvas.width  = targetW;
  dstCanvas.height = targetH;
  const ctx = dstCanvas.getContext("2d")!;
  ctx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  return ctx.getImageData(0, 0, targetW, targetH);
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Draw canvas → resize → return 28×28 ImageData */
function getResizedImageData(
  canvas: HTMLCanvasElement,
  w: number,
  h: number
): ImageData {
  const tmp  = document.createElement("canvas");
  tmp.width  = w;
  tmp.height = h;
  const ctx  = tmp.getContext("2d")!;

  // White background so blank areas aren't transparent / black by accident
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(canvas, 0, 0, w, h);

  return ctx.getImageData(0, 0, w, h);
}

/**
 * RGBA ImageData  →  Float32Array NCHW [1, 1, 28, 28]
 *
 * Greyscale conversion: Y = 0.299R + 0.587G + 0.114B  (BT.601 luma)
 * Normalisation       : y / 255  →  [0, 1]
 *
 * Note: The model was trained on dark digits on light backgrounds.
 * We INVERT the image when the canvas background is white (typical drawing
 * app convention) so that strokes → high values, matching the training set.
 */
export function imageDataToTensor(
  imageData: ImageData,
  invertColors = true
): Float32Array {
  const { data, width, height } = imageData;
  const n = width * height;
  const tensor = new Float32Array(channels * n); // channels === 1

  for (let i = 0; i < n; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // Luma (greyscale)
    let grey = 0.299 * r + 0.587 * g + 0.114 * b;

    // Invert: white bg → 0, dark stroke → ~255
    if (invertColors) grey = 255 - grey;

    tensor[i] = grey / normDivide;
  }

  return tensor; // shape logically: [1, 1, 28, 28]
}

/**
 * Helper: Given the flat Float32Array produced above, return a debug
 * representation as a 28×28 number[][] (useful for visualisation).
 */
export function tensorTo2D(tensor: Float32Array): number[][] {
  const grid: number[][] = [];
  for (let row = 0; row < imageHeight; row++) {
    const rowArr: number[] = [];
    for (let col = 0; col < imageWidth; col++) {
      rowArr.push(tensor[row * imageWidth + col]);
    }
    grid.push(rowArr);
  }
  return grid;
}
