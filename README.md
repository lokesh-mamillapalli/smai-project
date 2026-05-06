# Handwritten Kannada Digit Recognizer

This repository contains the front-end application for the **Handwritten Indic Script Recognition** project (Topic T3). It is a modern React-based web app that uses a custom-trained CNN to classify Kannada handwritten digits (0-9) entirely within the user's browser.

## Features
- **Live Canvas Drawing:** Draw a Kannada digit using your mouse or touch device and receive an instant prediction with zero network latency.
- **Drag & Drop Uploads:** Alternatively, drag and drop an existing image of a handwritten digit into the upload zone for rapid classification.
- **Practice Mode:** A built-in gamified mode that challenges users to draw a randomly generated Kannada digit. A faint watermark of the target character is displayed in the background of the canvas to guide the user's strokes. Get immediate "Correct/Try Again" feedback and track your session score!
- **High-Accuracy Animations:** The model achieves >97% accuracy. Whenever the model hits >97% confidence on a prediction, the app celebrates with a custom confetti animation!
- **In-Browser Inference:** Powered by `onnxruntime-web`, the neural network executes locally on your machine, ensuring data privacy and lighting-fast results.

## Real-World Use Cases

Our application natively solves the core problems outlined in the Handwritten Indic Script Recognition task:

1. **Educational Apps:**  
   The interactive canvas and "Practice Mode" act as a perfect educational suite. Students learning the Kannada script can practice writing numerals and instantly see if their stroke order and structure are legible to native readers.
   
2. **Form Digitization:**  
   Regional forms in Karnataka (such as government documents or bank deposit slips) heavily rely on handwritten Kannada digits. The upload and preprocessing pipeline in this app demonstrates how localized script models can be used to scan and digitize paper records seamlessly.
   
3. **Accessibility:**  
   Not all users are comfortable with standard keyboards. This app provides an accessibility layer, allowing users to input digits natively in their mother tongue simply by drawing them on a touch screen.

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed (v22.12.0+ recommended).

### Installation
1. Navigate to the project directory:
   ```bash
   cd numbo-draw-main
   ```
2. Install the dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser to the local URL (usually `http://localhost:5173`) and start drawing!

---
*Built using React, Vite, Tailwind CSS, and ONNX Runtime Web.*
