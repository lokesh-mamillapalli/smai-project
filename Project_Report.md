# Final Project Report: Handwritten Indic Script Recognition (Kannada MNIST)

**Team Members:** 
- Lokesh Mamillapalli, 2023101115, lokesh.mamillapalli@students.iiit.ac.in
- Mannem Bheema siddartha, 2023101009, mannem.siddartha@students.iiit.ac.in
- Chaitanya Thogata, 2023113018, chaitanya.thogata@research.iiit.ac.in

**GitHub Repository:** https://github.com/lokesh-mamillapalli/smai-project

---

## 1. Introduction
The objective of this project (Topic T3) is to build an end-to-end Handwritten Indic Script Recognition system. Specifically, we focus on Kannada digits (0-9). The project encompasses training a custom lightweight Convolutional Neural Network (CNN) from scratch and deploying it in a modern, interactive web application. This application allows users to draw characters with their mouse or upload images, providing instant, live unicode predictions completely in the browser. 

This technology has profound implications for:
- **Educational Apps:** Allowing students to practice writing native scripts and receiving immediate feedback.
- **Form Digitization:** Automating the extraction of handwritten numeric data from regional Indian forms.
- **Accessibility:** Providing alternative, localized input methods for users across India.

## 2. Dataset and Preprocessing
We utilized the Kaggle **Kannada-MNIST** dataset. To ensure a robust model, the primary `train.csv` resulting in approximately 65,000 training samples.

- **Data Split:** An 80/20 split was applied to separate the 60,000 training samples into training and validation sets.
- **Preprocessing:** Pixel values were normalized to the `[0, 1]` range and reshaped to `(28, 28, 1)` tensors.
- **Augmentation:** To prevent overfitting and improve generalization on real-world canvas drawings, we utilized on-the-fly data augmentation using TensorFlow's `ImageDataGenerator`, including:
  - Rotation (up to 10 degrees)
  - Width and Height shifts (10%)
  - Shear and Zoom transformations (10%)

## 3. Model Architecture
We designed a lightweight, highly efficient 3-block CNN (~300k parameters) built entirely from scratch using TensorFlow/Keras. No transfer learning was required.

**Architecture Details:**
1. **Conv Block 1:** `Conv2D(32, 3x3)` → `BatchNorm` → `ReLU` → `Conv2D(32, 3x3)` → `BatchNorm` → `ReLU` → `MaxPool2D(2)` → `Dropout(0.25)`
2. **Conv Block 2:** `Conv2D(64, 3x3)` → `BatchNorm` → `ReLU` → `Conv2D(64, 3x3)` → `BatchNorm` → `ReLU` → `MaxPool2D(2)` → `Dropout(0.25)`
3. **Conv Block 3:** `Conv2D(128, 3x3)` → `BatchNorm` → `ReLU` → `MaxPool2D(2)` → `Dropout(0.25)`
4. **Classifier Head:** `Flatten` → `Dense(128)` → `BatchNorm` → `ReLU` → `Dropout(0.5)` → `Dense(10, Softmax)`

## 4. Training Methodology
The model was trained on a Kaggle T4 GPU, reaching high accuracy (>97%) in under 5 minutes.
- **Optimizer:** `Adam` (Learning Rate = 1e-3)
- **Callbacks:** `ReduceLROnPlateau` (to dynamically reduce LR on validation plateaus) and `EarlyStopping` (patience of 10 epochs, restoring best weights) to accelerate convergence and prevent overfitting.
- **Loss Function:** `categorical_crossentropy` to handle the one-hot encoded multi-class distributions.
- **Export:** The model function trace was converted to the ONNX format (`kannada_cnn.onnx`) using `tf2onnx` to allow cross-platform inference natively in the browser.

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
