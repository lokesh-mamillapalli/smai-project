# Final Project Report: Handwritten Indic Script Recognition (Kannada MNIST)

**Team Members:** 
- [Name 1], [Roll Number 1], [Email 1]
- [Name 2], [Roll Number 2], [Email 2]

**GitHub Repository:** [Insert Link Here]

---

## 1. Introduction
The objective of this project (Topic T3) is to build an end-to-end Handwritten Indic Script Recognition system. Specifically, we focus on Kannada digits (0-9). The project encompasses training a custom lightweight Convolutional Neural Network (CNN) from scratch and deploying it in a modern, interactive web application. This application allows users to draw characters with their mouse or upload images, providing instant, live unicode predictions completely in the browser. 

This technology has profound implications for:
- **Educational Apps:** Allowing students to practice writing native scripts and receiving immediate feedback.
- **Form Digitization:** Automating the extraction of handwritten numeric data from regional Indian forms.
- **Accessibility:** Providing alternative, localized input methods for users across India.

## 2. Dataset and Preprocessing
We utilized the Kaggle **Kannada-MNIST** dataset. To ensure a robust model, we combined the primary `train.csv` and the supplementary `Dig-MNIST.csv` datasets, resulting in approximately 70,000 training samples.

- **Data Split:** A 90/10 split was applied to separate the dataset into training and validation sets.
- **Preprocessing:** Pixel values were normalized to the `[0, 1]` range and reshaped to `1x28x28` tensors.
- **Augmentation:** To prevent overfitting and improve generalization on real-world canvas drawings, we utilized dynamic data augmentation including:
  - `RandomAffine` (rotation up to 10 degrees, slight scaling and translation)
  - `RandomErasing` (to simulate broken strokes or noise)

## 3. Model Architecture
We designed a lightweight, highly efficient 3-layer CNN (~300k parameters) built entirely from scratch using PyTorch. No transfer learning was required.

**Architecture Details:**
1. **Conv Block 1:** `Conv2d(1 -> 32, kernel=3, padding=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`
2. **Conv Block 2:** `Conv2d(32 -> 64, kernel=3, padding=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`
3. **Conv Block 3:** `Conv2d(64 -> 128, kernel=3, padding=1)` → `BatchNorm2d` → `ReLU` → `AdaptiveAvgPool2d(3)`
4. **Classifier Head:** `Flatten` → `Dropout(0.3)` → `Linear(128*3*3 -> 256)` → `ReLU` → `Linear(256 -> 10)`

## 4. Training Methodology
The model was trained on a Kaggle T4 GPU, reaching state-of-the-art accuracy (>97%) in under 5 minutes.
- **Optimizer:** `AdamW` (LR = 3e-3, Weight Decay = 1e-4)
- **Scheduler:** `OneCycleLR` to dynamically adjust learning rates, accelerating convergence.
- **Loss Function:** `CrossEntropyLoss` with label smoothing (0.05) to penalize overconfidence.
- **Export:** The best model state was exported to the ONNX format (`kannada_cnn.onnx`) with dynamic batching to allow cross-platform inference in the browser.

## 5. Web Application Integration
Instead of a standard Streamlit app, we elevated the user experience by building a **modern React application** using Vite, Tailwind CSS, and TanStack Router. This allows the model to run *entirely client-side* via `onnxruntime-web` and `TensorFlow.js`, eliminating server latency and drastically improving privacy and speed.

### Key Features
- **Drawable Canvas:** Users can draw directly on a responsive canvas. The app intercepts the pointer events, scales the strokes, and maps them to a 28x28 grayscale tensor.
- **Drag & Drop Uploads:** Users can easily drag and drop cropped images of digits for instant prediction.
- **Practice Mode:** A gamified feature that challenges users to draw a specific random Kannada digit. It displays a faint watermark of the target character in the background of the canvas to guide the user's strokes, provides immediate "Correct/Try Again" feedback, and tracks the session score.
- **Celebration Animations:** If the model predicts a digit with >97% confidence, a celebratory confetti animation is triggered, reinforcing positive user interaction.
- **Probability Distribution:** Visualizes the model's confidence across all 10 classes using a dynamic bar chart.

## 6. Real-World Applications & Use Cases
- **Educational Apps:** The dedicated *Practice Mode* functions directly as a learning tool. Students learning the Kannada script can be prompted to draw specific digits and immediately know if their stroke order and shape are legible.
- **Form Digitization:** Government and regional bank forms in Karnataka frequently contain handwritten Kannada numbers. By extending our canvas/upload logic to bounding-box extractors, our lightweight ONNX model can instantly digitize these localized paper records without expensive cloud GPU API calls.
- **Accessibility:** Users who are not comfortable with standard QWERTY numeric pads can use a stylus or touch screen to natively input numeric data in their mother tongue, seamlessly bridging the digital divide.

## 7. Conclusion
By fusing a custom-trained, highly optimized CNN with a modern edge-inference web stack, we successfully built a robust Indic Script Recognizer. The system processes handwriting with zero network latency, hits >97% accuracy, and packages the solution into an engaging, educational, and highly accessible user interface.
