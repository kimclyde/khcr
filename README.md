# 🔢📝 Khmer Handwriting Recognition

A Streamlit web application for recognizing handwritten Khmer digits (០-៩) and characters using deep learning models.

## Features

- **Khmer Digit Recognition**: Recognizes handwritten Khmer digits (០-៩) using a Deep LSTM model
- **Khmer Character Recognition**: Recognizes handwritten Khmer characters using a Hybrid CNN-RNN model
- **Interactive Canvas**: Draw directly on the web interface using an interactive canvas
- **Real-time Prediction**: Get instant recognition results with confidence scores


## Installation

1. Clone the repository:
```bash
git clone https://github.com/kimclyde/khcr.git
cd khdr
```

2. Install required dependencies:
```bash
pip install streamlit torch streamlit-drawable-canvas
```

3. Ensure model files are present:
   - [`best_lstm_model.pth`](best_lstm_model.pth) for digit recognition
   - [`best-checkpoint.ckpt`](best-checkpoint.ckpt) for character recognition

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Choose between two recognition modes:
   - **🔢 Khmer Digit Recognition**: For recognizing digits ០-៩
   - **📝 Khmer Character Recognition**: For recognizing Khmer characters

4. Draw on the canvas using your mouse or touch input

5. Click "Recognize Character" to get predictions


