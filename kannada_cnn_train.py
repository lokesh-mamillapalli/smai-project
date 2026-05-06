!pip install -q onnx onnxscript

# Kannada-MNIST CNN Training & TS File Generation
# Run on Kaggle with GPU T4 accelerator enabled

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

t_global_start = time.time()

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
DATA_DIR = "/kaggle/input/competitions/Kannada-MNIST"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
dig_df   = pd.read_csv(os.path.join(DATA_DIR, "Dig-MNIST.csv"))

def df_to_tensors(df, has_label=True):
    pixel_cols = [c for c in df.columns if c.startswith("pixel")]
    X = df[pixel_cols].values.astype(np.float32) / 255.0
    X = X.reshape(-1, 1, 28, 28)
    X_t = torch.tensor(X, dtype=torch.float32)
    if has_label:
        y = df["label"].values.astype(np.int64)
        return X_t, torch.tensor(y, dtype=torch.long)
    return X_t

# Merge train + Dig-MNIST for more training data (~70 k samples)
combined = pd.concat([train_df, dig_df], ignore_index=True)
X_all, y_all = df_to_tensors(combined, has_label=True)
X_test       = df_to_tensors(test_df,  has_label=False)

# 90/10 split
n_val  = int(0.10 * len(X_all))
n_train= len(X_all) - n_val
perm   = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(SEED))
idx_tr, idx_val = perm[:n_train], perm[n_train:]

X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
X_val, y_val = X_all[idx_val], y_all[idx_val]

# ── 2. DATA AUGMENTATION & DATALOADERS ────────────────────────────────────────
train_transform = nn.Sequential(
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
)

def make_loader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

train_loader = make_loader(X_tr,  y_tr,  batch_size=256, shuffle=True)
val_loader   = make_loader(X_val, y_val, batch_size=512, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ── 3. MODEL DEFINITION ───────────────────────────────────────────────────────
class KannadaCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d(3)

        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(128 * 3 * 3, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x, augment=False):
        if augment and self.training:
            x = train_transform(x)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

model = KannadaCNN(dropout=0.3).to(DEVICE)

# ── 4. TRAINING ───────────────────────────────────────────────────────────────
EPOCHS   = 20
LR       = 3e-3
WD       = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2,
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

best_val_acc = 0.0
best_state   = None

print("Training started...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_b, augment=True), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            preds = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += len(y_b)

    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

print(f"Best val accuracy : {best_val_acc*100:.3f}%")
model.load_state_dict(best_state)
model.eval()

# ── 5. SUBMISSION & EXPORT ────────────────────────────────────────────────────
test_preds = []
with torch.no_grad():
    for (X_b,) in test_loader:
        X_b = X_b.to(DEVICE)
        test_preds.extend(model(X_b).argmax(1).cpu().numpy())

pd.DataFrame({"id": test_df["id"], "label": test_preds}).to_csv("submission.csv", index=False)

dummy_input = torch.randn(1, 1, 28, 28, device=DEVICE)
torch.onnx.export(
    model, dummy_input, "kannada_cnn.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18, do_constant_folding=True,
)
print("ONNX model saved!")

# ── 6. GENERATE .TS FILES FOR LOVABLE APP ─────────────────────────────────────

model_config_ts = f"""export const MODEL_CONFIG = {{
  name: "KannadaCNN",
  version: "1.0.0",
  inputShape: [1, 28, 28] as const,
  labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  accuracy: "{best_val_acc * 100:.2f}%",
  modelPath: "/kannada_cnn.onnx"
}};
"""

preprocessing_ts = """export function preprocess(source: HTMLCanvasElement | HTMLImageElement): { data: Float32Array; previewCanvas: HTMLCanvasElement } {
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
"""

model_loader_ts = """import * as ort from 'onnxruntime-web';
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
"""

utils_ts = """import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
"""

with open("model-config.ts", "w") as f: f.write(model_config_ts)
with open("preprocessing.ts", "w") as f: f.write(preprocessing_ts)
with open("model-loader.ts", "w") as f: f.write(model_loader_ts)
with open("utils.ts", "w") as f: f.write(utils_ts)

print("TS files (model-config.ts, preprocessing.ts, model-loader.ts, utils.ts) generated successfully!")
print(f"Total notebook time: {(time.time()-t_global_start)/60:.1f} min")
