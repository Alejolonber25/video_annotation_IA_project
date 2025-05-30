from glob import glob
import numpy as np
import os
import cv2
import mediapipe as mp
import csv

# Configurar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

video_root = './SecondDelivery/videos'
raw_dataset_root = './SecondDelivery/raw_datasets'

# Create the directory if it doesn't exist
os.makedirs(raw_dataset_root, exist_ok=True)


# Get the names of all the directories in the video root --> Class names
label_folders = glob(os.path.join(video_root, '*'))

for label_folder in label_folders:
    label = os.path.basename(label_folder)
    print(f"\n***Processing label: {label}***")

    #Array to store the landmarks per class
    class_landmarks = []

    video_files = glob(os.path.join(label_folder, '*'))

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"***Extracting frames from: {video_name}***")

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                class_landmarks.append(landmarks)

        cap.release()

    #Save one CSV file with all landmarks per class
    if class_landmarks:
        csv_path = os.path.join(raw_dataset_root, f"{label}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            #CSV header for the landmarks
            header = []
            for i in range(33):
                header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
            header.append("label")

            writer.writerow(header)
            writer.writerows(class_landmarks)

        print(f"✅ Saved {len(class_landmarks)} samples to {csv_path}")
    else:
        print(f"⚠️ No landmarks found for label: {label}")

pose.close()
print("\n🎉 All label raw datasets created successfully.")
