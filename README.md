# Human Face Emotion Detection

## 📌 Overview
The **Human Face Emotion Detection** project utilizes **Deep Learning (CNN)** to recognize human emotions from facial expressions. It classifies images into seven different emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. The model is trained on the **Face Expression Recognition Dataset** and is deployed as a **Flask Web App**.

## 🚀 Features
- **Emotion Detection**: Classifies emotions from facial images.
- **Deep Learning Model**: Uses **Convolutional Neural Networks (CNN)** for classification.
- **Image Preprocessing**: Applies grayscale conversion, resizing, and normalization.
- **Web-based User Interface**: Built with Flask and Bootstrap for user-friendly interaction.
- **Real-time Predictions**: Upload an image and receive an emotion prediction with confidence score.

## 🛠️ Technologies Used
- **Deep Learning**: TensorFlow, Keras (CNN)
- **Python**: NumPy, OpenCV, Matplotlib, Seaborn
- **Flask**: Web framework for deployment
- **HTML, CSS, Bootstrap**: Frontend for the web application
- **Dataset**: Face Expression Recognition Dataset

## 🖼️ Workflow
1. **Dataset Preparation**: Images are preprocessed (grayscale conversion, resizing, normalization).
2. **CNN Model Training**:
   - Uses multiple convolutional layers with **ReLU activation** and **MaxPooling**.
   - Fully connected layers for classification.
   - Categorical crossentropy loss function & Adam optimizer.
3. **Model Evaluation**:
   - Evaluates performance using **accuracy, confusion matrix, and classification report**.
   - Implements **Early Stopping & Learning Rate Reduction** for optimization.
4. **Web Application Deployment**:
   - User uploads an image.
   - Image is processed and passed to the trained CNN model.
   - The detected emotion and confidence score are displayed in the UI.

## 📊 Model Performance
- **Achieved Accuracy**: 92% on test data.
- **Feature Extraction**: CNN extracts relevant facial features.
- **Evaluation Metrics**: Classification report and confusion matrix used for performance analysis.

## 🌐 Usage
1. Run the Flask app locally using `python main.py`.
2. Open `http://127.0.0.1:5000/` in a browser.
3. Upload a face image.
4. The app will predict and display the detected emotion with confidence.



