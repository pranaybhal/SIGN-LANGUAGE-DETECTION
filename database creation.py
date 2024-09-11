import os
import cv2
import mediapipe as mp
import pickle  # Import pickle module

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the MediaPipe Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Path to the main data directory (where folders 0, 1, 2 are located)
DATA_DIR = 'C:/Users/Lenovo/Desktop/SIGN LANGUAGE DETECTION/data'

OUTPUT_DIR = './output'  # Directory to save images with landmarks

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data = []
labels = []

# Iterate through each class folder (0, 1, 2)
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)

    # Ensure we are working with directories
    if not os.path.isdir(class_path):
        continue

    # Iterate through each image in the class directory
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        data_aux = []
        x_ = []
        y_ = []

        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            print(f"Hand landmarks detected in image: {img_path}")

            # Draw landmarks on the image for all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Draw landmarks and connections on the image
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            data.append(data_aux)
            labels.append(class_dir)  # Use the folder name as the label (0, 1, 2)

        else:
            print(f"No hand landmarks detected in image: {img_path}")

        # Save the image with landmarks
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(output_path, img)  # Save the image

        # Display the image with landmarks (optional)
        cv2.imshow('Hand Landmarks', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

hands.close()

# Save data and labels to a file using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)