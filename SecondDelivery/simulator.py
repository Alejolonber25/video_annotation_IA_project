import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

model = joblib.load('./SecondDelivery/best_pose_classifier_model.pkl')

N = 5
frame_buffer = deque(maxlen=N)

cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    """Extrae landmarks planos [x1, y1, z1, v1, ..., x33, y33, z33, v33]"""
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks if len(landmarks) == 132 else None  # 33 * 4 = 132

def compute_avg_diff(window):
    """Calcula la media de diferencias entre frames consecutivos"""
    diffs = []
    for i in range(len(window) - 1):
        diff = np.array(window[i + 1]) - np.array(window[i])
        diffs.append(diff)
    return np.mean(diffs, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    lm = extract_landmarks(results)
    if lm:
        frame_buffer.append(lm)

    label = "Detecting..."
    if len(frame_buffer) == N:
        avg_diff = compute_avg_diff(list(frame_buffer))
        prediction = model.predict([avg_diff])
        label = prediction[0]

    cv2.putText(frame, f"Prediction: {label}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
