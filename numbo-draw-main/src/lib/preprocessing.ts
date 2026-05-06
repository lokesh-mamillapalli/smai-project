export function preprocess(source: HTMLCanvasElement | HTMLImageElement): { data: Float32Array; previewCanvas: HTMLCanvasElement } {
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  ctx.drawImage(source, 0, 0, 28, 28);

  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = imageData.data;
  
  const tensor = new Float32Array(28 * 28);
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    
    const gray = (r * 0.299 + g * 0.587 + b * 0.114);
    
    let val = 0;
    if (a > 0) {
      val = 255 - gray;
    }
    
    tensor[i / 4] = val / 255.0;
  }
  
  return { data: tensor, previewCanvas: canvas };
}
