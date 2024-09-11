Project Report for Sign Language Detection System
1. Introduction
•	Project Title: Sign Language Detection Using Machine Learning and Computer Vision
•	Objective: The aim of this project is to create a system that can detect and recognize sign language letters from a live video feed using computer vision and machine learning techniques.
•	Purpose: To help bridge the communication gap between sign language users and non-sign language users by providing a real-time, accessible sign language recognition tool.
2. Background
•	Background: Sign language is a critical form of communication for individuals who are deaf or hard of hearing. Automatic recognition of sign language through image processing and machine learning is an active area of research with applications in accessibility and education. This project involves building a system that uses a webcam to capture hand gestures and classifies them into sign language characters.
3. Learning Objectives
•	Learn how to collect and preprocess image data for training machine learning models.
•	Implement computer vision techniques using OpenCV and MediaPipe to detect hand gestures.
•	Develop a Random Forest machine learning model for classifying sign language gestures.
•	Understand how to evaluate and improve model performance.
4. Activities and Tasks
1.	Data Collection: Used OpenCV to capture images of hand gestures (A, B, L) for training the classifier.
•	Script: collecting images.py
2.	Data Preprocessing: Preprocessed the images and prepared the dataset for training.
•	Script: creating dataset.py
•	Dataset: data.pickle
3.	Model Training: Trained a Random Forest Classifier on the preprocessed data.
•	Script: Train classifier.py
•	Model: model.p
4.	Model Testing and Prediction: Integrated the model with live webcam input to recognize and classify sign language gestures.
•	Code: import cv2.py
5. Skills and Competencies
•	Technical Skills:
•	Proficiency in Python programming, particularly with libraries such as OpenCV, MediaPipe, and scikit-learn.
•	Data preprocessing and feature scaling.
•	Hyperparameter tuning using GridSearchCV.
•	Model evaluation using cross-validation.
•	Real-time video processing.
•	Soft Skills:
•	Problem-solving: Overcame challenges related to data quality, model accuracy, and system integration.
•	Research: Explored and implemented best practices in gesture recognition and machine learning.
6. Feedback and Evidence
•	Accuracy: After training, the model achieved a cross-validation accuracy of 85 % accuracy and a test accuracy of 90%
•	Challenges Identified: Some gestures (like ‘B’ and ‘L’) were occasionally misclassified due to similarity in hand shape.
•	Feedback: The system worked well with controlled lighting but required fine-tuning for real-world conditions with varying lighting and backgrounds.
7. Challenges and Solutions
•	Challenge 1: Overfitting on certain gestures: The model was performing too well on training data but poorly on real-world input.
•	Solution: Used cross-validation to better generalize the model, and increased the variability in the training dataset by collecting data under different lighting conditions.
•	Challenge 2: Low accuracy in live video detection.
•	Solution: Adjusted the MediaPipe hand detection confidence and tuned the random forest hyperparameters to improve performance.
8. Outcomes and Impact
•	Outcome: The project successfully implemented a real-time sign language recognition system capable of identifying hand gestures for A, B, and L.
•	Impact: This project demonstrates how machine learning and computer vision can be applied to increase accessibility for people who rely on sign language. Further development could expand its usefulness to a broader range of gestures and alphabets.
9. Conclusion
•	Reflection: This project provided valuable hands-on experience with machine learning, computer vision, and real-time systems. While there were challenges, the project successfully achieved its objective of building a basic sign language recognition system. In the future, the project could be expanded by including more sign language letters, improving gesture accuracy, and incorporating additional languages.
•	Next Steps: Explore other machine learning models, such as Convolutional Neural Networks (CNNs), which may yield better results for image classification tasks like sign language detection.
