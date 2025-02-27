# Human Facial Emotion Detection using Deep Learning (CNN)

## 📌 Overview
This project is a **Facial Emotion Detection** system built using **Deep Learning (CNN)**. It classifies human facial expressions into seven emotions:

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😀 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

The system is trained on the **Face Expression Recognition Dataset** and deployed using **Flask** for real-time emotion detection.


## 🚀 Features
- **CNN Model** trained on the Face Expression Recognition dataset
- **Flask Web Application** for real-time emotion detection
- **Image Preprocessing** using Keras' ImageDataGenerator
- **Model Training** with **Conv2D, MaxPooling2D, BatchNormalization, Dropout**
- **Live Emotion Prediction** on uploaded images
- **Confusion Matrix & Classification Report** for performance analysis

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**(for deep learning model))
- **Flask** (for deployment)
- **NumPy, Matplotlib, Seaborn** (for visualization)
- **OpenCV** (for image processing)
- **Bootstrap 5** (for frontend styling)


## 🏋️‍♂️ Model Training
The `human_face_emotion_recognition.ipynb` file contains the training pipeline:
- Dataset preprocessing
- Data augmentation
- CNN model architecture
- Model training & evaluation
- Performance visualization

## 📸 Usage
1. Run the Flask app.
2. Upload an image with a human face.
3. Get the detected emotion with a confidence score.

## 📊 Model Performance
The model achieves high accuracy in recognizing emotions, validated using:
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** for misclassification analysis



