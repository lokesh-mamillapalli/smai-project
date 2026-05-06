export const MODEL_CONFIG = {
  name: "KannadaCNN",
  version: "1.0.0",
  inputShape: [1, 28, 28] as const,
  labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  accuracy: "99.59%",
  modelPath: "/kannada_cnn.onnx"
};
