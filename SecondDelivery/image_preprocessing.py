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

video_root = './videos'
dataset_root = './dataset'

# Crear carpeta destino si no existe
os.makedirs(dataset_root, exist_ok=True)

# Archivo CSV general
csv_path = os.path.join(dataset_root, "landmarks_dataset.csv")

# Lista para guardar todos los landmarks
all_landmarks = []

# Obtener todas las carpetas (labels)
label_folders = glob(os.path.join(video_root, '*'))

for label_folder in label_folders:
    label = os.path.basename(label_folder)
    print(f"Processing label: {label}")

    video_files = glob(os.path.join(label_folder, '*'))

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Extracting frames from: {video_name}")

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
                landmarks.append(label)  # Agregar la etiqueta al final
                all_landmarks.append(landmarks)

        cap.release()

pose.close()

# Guardar todos los landmarks en un solo CSV
if all_landmarks:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Crear encabezado
        header = []
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        header.append("label")

        writer.writerow(header)
        writer.writerows(all_landmarks)

    print(f"Saved all landmark data to {csv_path}")
else:
    print("No landmarks found in any video.")

print("✅ Landmark CSV dataset created successfully.")
