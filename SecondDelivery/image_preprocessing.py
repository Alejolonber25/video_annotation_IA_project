from glob import glob
import numpy as np
import os
import cv2
import mediapipe as mp
import csv

#Read files

#print("Current working directory:", os.getcwd())

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

video_root = 'SecondDelivery/captures/videos'

dataset_root = 'SecondaryDelivery/captures/dataset'

#Make the dir for frames if it doesn't exist
os.makedirs(dataset_root, exist_ok=True)

#get all the available actions stored in the video dir
label_folders = glob(os.path.join(video_root, '*'))

for label_folder in label_folders:
    label = os.path.basename(label_folder)
    print(f"Processing label: {label}")

    #Create the equivalent directory within the dataset dir
    dataset_label_folder = os.path.join(dataset_root, label)
    os.makedirs(dataset_label_folder, exist_ok=True)
    
    #Now get all the videos within the folder
    video_files = glob(os.path.join(label_folder, '*'))

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Extracting frames from: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        frameCount = 0
        all_landmarks =[]

        #Divide the frames into images and mark the landmarks
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.proces(rgb)

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmarks.append(label)  # Add label at the end
                all_landmarks.append(landmarks)

            frame_count += 1

        cap.release()

        #Save te landmarks in a CSV in their respective fir
        if all_landmarks:
            csv_path = os.path.join(dataset_label_folder, f"{video_name}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                header = []
                for i in range(33):
                    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
                header.append("label")

                writer.writerow(header)
                writer.writerows(all_landmarks)

            print(f"Saved landmark data to {csv_path}")
        else:
            print(f"No landmarks found in video: {video_name}")

pose.close()
print("Landmark CSV files saved in dataset folder.")