# Kannada MNIST Drawing Classifier

This project is an interactive web application that allows users to draw Kannada digits and uses a Convolutional Neural Network (CNN) to classify the drawn digit in real-time.

The project is split into two main parts:
1. **Model Training**: A Python script to train a CNN on the Kannada-MNIST dataset and export it to ONNX format.
2. **Frontend Web App**: A React + Vite web application that loads the ONNX model using ONNX Runtime Web and infers the digit drawn on a canvas.

## Prerequisites

- [Node.js and npm](https://nodejs.org/) (for running the frontend)
- [Bun](https://bun.sh/) (optional alternative for running the frontend)
- Python 3.8+ (if you want to retrain the model)

## 1. Running the Frontend Web App

The frontend is a Vite project built with React and uses `Bun` as the package manager and runtime.

To get started with the frontend:

1. Navigate to the web application directory:
   ```bash
   cd numbo-draw-main
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```
   *(Alternatively, you can use `bun install` or `pnpm install`.)*

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).

## Tech Stack

- **Model:** PyTorch, ONNX
- **Frontend:** React, Vite, Tailwind CSS, shadcn/ui
- **Inference:** ONNX Runtime Web (`onnxruntime-web`)
