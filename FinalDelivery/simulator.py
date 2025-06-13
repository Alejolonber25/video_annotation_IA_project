#import all the necessary libraries
import cv2 
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
import time

# load the pose detector with the same parameters that were used to make the dataset
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

# Load all the necessary models and transformers
model = joblib.load('./FinalDelivery/pose_knn.pkl') #This can be changed to any other model that is exported
scaler = joblib.load('./FinalDelivery/scaler.pkl')
pca = joblib.load('./FinalDelivery/pca.pkl')
le = joblib.load('./FinalDelivery/label_encoder.pkl') 

# Use the same N for average of differences between frames
N = 2
#Make a frame buffer and a prediction history to make slower and cleaner predictions
frame_buffer = deque(maxlen=N)
prediction_history = deque(maxlen=5)

# connect to the camara
cap = cv2.VideoCapture(0)

# This shows the first message when executing the file
last_prediction = "Detecting..."

#This two control the prediction speed
last_prediction_time = 0
cooldown_seconds = 1.0

# This transient classes are the ones that had the most trouble showing, 
# so they are prioritized in the voting process
TRANSIENT_CLASSES = {"sitting down", "sitting up", "towards"}

#Use the same method to extract the land marks
def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks if len(landmarks) == 132 else None

# This calculates the differences between frames and averages N frames
def compute_avg_diff(window):
    diffs = []
    for i in range(len(window) - 1):
        diff = np.array(window[i + 1]) - np.array(window[i])
        diffs.append(diff)
    return np.mean(diffs, axis=0)

#Normalizes the landmarks the same as in the process of making the data set
#I left more comments in that raw_to_preprocessed.py
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

#Make a voting method to take more into account the transient classes
def weighted_majority(history):
    weights = np.linspace(1, 2, len(history))  # Add more weight to the recent predictions
    counts = {}
    for label, weight in zip(history, weights):
        counts[label] = counts.get(label, 0) + weight
    return max(counts, key=counts.get)

#Main loop fo r execution
while True:
    #Make use of the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    #Get the frames from the camera
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Process the results from the frames
    results = pose.process(rgb)

    #Extract the landmarks
    lm = extract_landmarks(results)
    if lm:
        #Normalize the landmards
        lm = normalize_landmarks(lm)
        #Add the landmarks for the voting queue
        frame_buffer.append(lm)

    #Measure the current time to time to slow down the predictions
    current_time = time.time()
    if len(frame_buffer) == N and current_time - last_prediction_time > cooldown_seconds:
        #Calculate the differences between the frames within the buffer
        avg_diff = compute_avg_diff(list(frame_buffer))
        #normalize the data with the same scaler
        x_scaled = scaler.transform([avg_diff])
        #Reduce the dimentionality of the data
        x_pca = pca.transform(x_scaled)
        #Predict the actions using the imported model
        pred_encoded = model.predict(x_pca)[0] 
        #Change the integer categorical values back tos tring
        pred_label = le.inverse_transform([pred_encoded])[0]
        #Add the label to the prediction history
        prediction_history.append(pred_label)

        # Show a the prediction of a transient class as soon it shows up
        if pred_label in TRANSIENT_CLASSES:
            last_prediction = pred_label
        else:
            #If there is no transient one, use the most voted one
            last_prediction = weighted_majority(list(prediction_history))

        last_prediction_time = current_time

    # Show the prediction on screen
    cv2.putText(frame, f"Prediction: {last_prediction}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the prediction history
    hist_text = ', '.join(list(prediction_history))
    cv2.putText(frame, f"History: {hist_text}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # show on screen the prediction
    cv2.imshow('Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the resources when the execution ends
pose.close()
cap.release()
cv2.destroyAllWindows()
