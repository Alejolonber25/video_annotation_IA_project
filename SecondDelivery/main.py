import cv2
import mediapipe as mp
import os
import csv
import shutil

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Set up main directory
captures_path = './captures'
dataset_path = os.path.join(captures_path, 'dataset')
results_path = os.path.join(captures_path, 'results')

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Copy folder structure from dataset to results if missing
for folder in os.listdir(dataset_path):
    dataset_folder_path = os.path.join(dataset_path, folder)
    results_folder_path = os.path.join(results_path, folder)

    if os.path.isdir(dataset_folder_path):
        if not os.path.exists(results_folder_path):
            # Create the folder in results (empty, no files)
            os.makedirs(results_folder_path)
            print(f"Created missing folder in results: {results_folder_path}")

# List to hold data for CSV
data = []

# Process each label directory inside dataset/
for label in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label)
    if not os.path.isdir(label_folder):
        continue

    # Result folder should already exist now (created above if missing)
    label_result_folder = os.path.join(results_path, label)

    for filename in os.listdir(label_folder):
        image_path = os.path.join(label_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_path} (could not load)")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"No landmarks found for {image_path}")
            continue

        # Extract landmark values
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        landmarks.append(label)
        data.append(landmarks)

        # Draw landmarks
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Save annotated image
        save_path = os.path.join(label_result_folder, filename)
        cv2.imwrite(save_path, annotated_image)

# CSV header: x_0, y_0, z_0, v_0, ..., x_32, y_32, z_32, v_32, label
header = []
for i in range(33):  # 33 landmarks
    header += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
header.append("label")

# Write to CSV
with open("pose_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

# Release MediaPipe resources
pose.close()

print("✅ Processing complete. Annotated images in captures/results/, data in pose_data.csv")