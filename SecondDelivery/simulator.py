import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque

# Cargar pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

# Cargar modelo y transformadores
model = joblib.load('./SecondDelivery/pose_classifier_model.pkl')
scaler = joblib.load('./SecondDelivery/scaler.pkl')
pca = joblib.load('./SecondDelivery/pca.pkl')

# Buffer de frames
N = 2
frame_buffer = deque(maxlen=N)

# Cámara
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks if len(landmarks) == 132 else None

def compute_avg_diff(window):
    diffs = []
    for i in range(len(window) - 1):
        diff = np.array(window[i + 1]) - np.array(window[i])
        diffs.append(diff)
    return np.mean(diffs, axis=0)

def normalize_landmarks(landmarks):
    # landmarks: lista de 132 valores [x0, y0, z0, v0, ..., x32, y32, z32, v32]
    coords = np.array(landmarks).reshape((33, 4))
    
    # Centro el cuerpo: usar el punto de la cadera (ej. landmark 23 o 24 o promedio)
    center = (coords[23][:3] + coords[24][:3]) / 2  # solo x,y,z
    coords[:, :3] -= center  # centrar

    # Escalar por distancia entre hombros (landmark 11 y 12)
    shoulder_dist = np.linalg.norm(coords[11][:3] - coords[12][:3])
    if shoulder_dist > 0:
        coords[:, :3] /= shoulder_dist  # escalar
    return coords.flatten()

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    lm = extract_landmarks(results)
    if lm:
        lm = normalize_landmarks(lm)
        frame_buffer.append(lm)

    label = "Detecting..."
    if len(frame_buffer) == N:
        avg_diff = compute_avg_diff(list(frame_buffer))
        x_scaled = scaler.transform([avg_diff])  # Aplicar scaler
        x_pca = pca.transform(x_scaled)          # Aplicar PCA
        prediction = model.predict(x_pca)
        label = prediction[0]

    cv2.putText(frame, f"Prediction: {label}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
