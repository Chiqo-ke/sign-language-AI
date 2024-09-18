import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import pickle

# Initialize MediaPipe Hand and Pose tracking
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Load the saved motion data with equations
with open("motion_data_with_equations.pkl", "rb") as f:
    motion_data_with_equations = pickle.load(f)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to calculate velocity and acceleration
def calculate_velocity(landmark_sequence):
    return np.gradient(landmark_sequence, axis=0)

def calculate_acceleration(landmark_sequence):
    return np.gradient(np.gradient(landmark_sequence, axis=0), axis=0)

# Function to compare motion based on landmarks, velocity, and acceleration
def compare_motion(current_landmarks, saved_landmarks, saved_velocity, saved_acceleration, tolerance=0.2):
    current_velocity = calculate_velocity(current_landmarks)
    current_acceleration = calculate_acceleration(current_landmarks)

    position_diff = np.linalg.norm(current_landmarks - saved_landmarks)
    velocity_diff = np.linalg.norm(current_velocity - saved_velocity)
    acceleration_diff = np.linalg.norm(current_acceleration - saved_acceleration)

    total_diff = position_diff + velocity_diff + acceleration_diff
    max_diff = tolerance * 3  # Adjust this value for flexibility

    similarity = max(0, 1 - (total_diff / max_diff))
    return similarity * 100  # Return as percentage similarity

# Initialize webcam for real-time tracking
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, mp_pose.Pose() as pose:
    last_check_time = time.time()
    best_label_display = ""  # Store the best label for display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands and pose
        result_hands = hands.process(rgb_image)
        result_pose = pose.process(rgb_image)

        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                best_similarity = 0
                best_label = None

                for saved_landmarks, saved_velocity, saved_acceleration, label in motion_data_with_equations:
                    similarity = compare_motion(current_landmarks, saved_landmarks, saved_velocity, saved_acceleration)

                    # Track the best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_label = label

                # Update best match if similarity is above a certain threshold
                current_time = time.time()
                if current_time - last_check_time >= 2:
                    if best_similarity >= 15:
                        print(f"Best match: {best_label} with {best_similarity:.2f}% similarity.")
                        engine.say(best_label)  # Speak out the label
                        engine.runAndWait()
                        best_label_display = best_label  # Update the label to display on screen
                    last_check_time = current_time

        # Display the frame with recognized sign
        if best_label_display:
            cv2.putText(frame, f"Recognized: {best_label_display}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
