import cv2 
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
import time

# Cargar pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

# Cargar modelo y transformadores
model = joblib.load('./SecondDelivery/pose_svm.pkl')
scaler = joblib.load('./SecondDelivery/scaler.pkl')
pca = joblib.load('./SecondDelivery/pca.pkl')
le = joblib.load('./SecondDelivery/label_encoder.pkl')  # 👈 Cargar el LabelEncoder

# Buffer de frames y predicciones
N = 2
frame_buffer = deque(maxlen=N)
prediction_history = deque(maxlen=5)

# Cámara
cap = cv2.VideoCapture(0)

# Control de predicción
last_prediction = "Detecting..."
last_prediction_time = 0
cooldown_seconds = 1.0

# Clases transitorias que deben mostrarse si aparecen aunque no ganen la votación
TRANSIENT_CLASSES = {"sitting down", "sitting up", "towards"}

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
    coords = np.array(landmarks).reshape((33, 4))
    
    center = (coords[23][:3] + coords[24][:3]) / 2
    coords[:, :3] -= center

    left_ankle = coords[27][:3]
    right_ankle = coords[28][:3]
    ankle_center = (left_ankle + right_ankle) / 2

    head = coords[0][:3]
    body_height = np.linalg.norm(head - ankle_center)

    if body_height == 0:
        shoulder_dist = np.linalg.norm(coords[11][:3] - coords[12][:3])
        body_height = shoulder_dist if shoulder_dist > 0 else 1.0

    coords[:, :3] /= body_height
    return coords.flatten()

def weighted_majority(history):
    weights = np.linspace(1, 2, len(history))  # más peso a predicciones recientes
    counts = {}
    for label, weight in zip(history, weights):
        counts[label] = counts.get(label, 0) + weight
    return max(counts, key=counts.get)

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

    current_time = time.time()
    if len(frame_buffer) == N and current_time - last_prediction_time > cooldown_seconds:
        avg_diff = compute_avg_diff(list(frame_buffer))
        x_scaled = scaler.transform([avg_diff])
        x_pca = pca.transform(x_scaled)
        pred_encoded = model.predict(x_pca)[0]  # entero
        pred_label = le.inverse_transform([pred_encoded])[0]  # 👈 convertir a string

        prediction_history.append(pred_label)

        # Mostrar predicción transitoria inmediatamente si aparece
        if pred_label in TRANSIENT_CLASSES:
            last_prediction = pred_label
        else:
            last_prediction = weighted_majority(list(prediction_history))

        last_prediction_time = current_time

    # Mostrar predicción en pantalla
    cv2.putText(frame, f"Prediction: {last_prediction}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # (Opcional) Mostrar historial de predicciones para depuración
    hist_text = ', '.join(list(prediction_history))
    cv2.putText(frame, f"History: {hist_text}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
