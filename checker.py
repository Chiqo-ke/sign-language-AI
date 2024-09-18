import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import re  # To handle extracting labels from filenames

# Initialize MediaPipe Hand and Pose tracking
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Path to the directory containing training videos
train_video_dir = 'Training'

# List to store hand landmarks data and labels
data = []

# Function to extract the gesture name from the filename, ignoring numbers
def extract_label(filename):
    match = re.match(r"([a-zA-Z]+)", filename)
    if match:
        return match.group(1)  # Extract the non-numeric part as the label
    return None

# Process all videos in the directory
for filename in os.listdir(train_video_dir):
    if filename.endswith('.mp4'):
        video_path = os.path.join(train_video_dir, filename)
        label = extract_label(filename)  # Extract the label
        if label:
            cap = cv2.VideoCapture(video_path)

            with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, mp_pose.Pose() as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:  # End of video
                        break

                    # Convert the image to RGB
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect hands and pose in the image
                    result_hands = hands.process(rgb_image)
                    result_pose = pose.process(rgb_image)

                    if result_hands.multi_hand_landmarks:
                        for hand_landmarks in result_hands.multi_hand_landmarks:
                            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                            # Calculate velocity and acceleration
                            velocity = np.gradient(landmarks, axis=0)
                            acceleration = np.gradient(velocity, axis=0)

                            # Append the landmarks, velocity, acceleration, and label to data
                            data.append((landmarks, velocity, acceleration, label))

            cap.release()

# Save the hand landmarks, velocity, acceleration, and labels to a file
with open("motion_data_with_equations.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"Processed and saved data from {len(os.listdir(train_video_dir))} videos.")
