Sign Language Detection Project
This project focuses on detecting sign language hand gestures (currently the letters A, B, and L) using computer vision techniques and machine learning algorithms. The main goal is to capture and classify hand gestures in real-time through a webcam feed.

Table of Contents
Introduction
Project Structure
Installation
Usage
Model Training
Testing the Model
Dataset
Challenges
Contributions

Introduction
The Sign Language Detection Project leverages machine learning to recognize basic hand gestures of the American Sign Language alphabet. The system captures live webcam data and classifies the gestures in real-time. The model was trained on a dataset containing labeled hand gesture data, which was then used to train a Random Forest classifier. Mediapipe is used to detect hand landmarks and OpenCV helps in capturing webcam input.

Project Structure
The repository consists of the following key files:

collecting_images.py: Script to capture and save hand gesture images.
creating_dataset.py: Prepares a dataset from the captured images for training.
Train_classifier.py: Trains a Random Forest classifier to recognize hand gestures.
test_classifier.py: Tests the trained model in real-time using webcam input.
model.p: Pre-trained Random Forest classifier saved as a pickle file.
data.pickle: Labeled dataset of hand gestures.

Installation
To set up the project, follow these steps:


Install the necessary dependencies:

bash

pip install -r requirements.txt
Ensure you have a webcam or other camera connected to your system for real-time hand gesture detection.

Usage
1. Collecting Images
You can capture images of different hand gestures using collecting_images.py. The script captures frames from your webcam and saves them for dataset preparation.



bash

python collecting_images.py
2. Creating Dataset
Once you have collected images for various hand gestures, use creating_dataset.py to create a dataset suitable for training.



bash

python creating_dataset.py
3. Model Training
To train the Random Forest classifier on the dataset of hand gestures, run the Train_classifier.py file. The model will be saved in a pickle format for later use.

bash

python Train_classifier.py
4. Testing the Model
After training the model, you can test the real-time classification using test_classifier.py. The script will open a webcam window where it will predict the letter for your hand gesture (A, B, or L).

bash

python test_classifier.py
Dataset
The dataset is created manually using the collecting_images.py script, which captures the hand gesture images. These images are processed into a feature set using Mediapipe's hand detection to extract hand landmarks.

Challenges
Overfitting: As the dataset is relatively small, there is a risk of overfitting, which can affect the model's generalization capabilities.
Hand gesture variations: Small variations in hand position, lighting, and background can affect the accuracy of predictions.
Real-time performance: Ensuring smooth real-time performance while maintaining prediction accuracy is a challenge in practical implementations.
Contributions
If you want to contribute to the project, feel free to open an issue or submit a pull request.

