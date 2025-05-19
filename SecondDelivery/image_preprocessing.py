from glob import glob
import numpy as np
import os
import cv2

#Read files

#print("Current working directory:", os.getcwd())

root = 'SecondDelivery/captures'
#fileNames = glob(os.path.join(root, 'dataset', '**', '*.jpg'), recursive=True)
#print(f"Found {len(fileNames)} images:")
#print(fileNames)

videoRoot = os.path.join(root, 'videos')
frameRoot = os.path.join(root, 'frames')

#Make the dir for frames if it doesn't exist
os.makedirs(frameRoot, exist_ok=True)

labelFolders = glob(os.path.join(videoRoot, '*'))

for labelFolder in labelFolders:
    label = os.path.basename(labelFolder)
    print(f"Processing label: {label}")

    saveLabelFolder = os.path.join(frameRoot, label)
    os.makedirs(saveLabelFolder, exist_ok=True)

    videoFiles = glob(os.path.join(labelFolder, '*'))

    for videoPath in videoFiles:
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        print(f"Extracting frames from: {videoName}")
        
        cap = cv2.VideoCapture(videoPath)
        frameCount = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frameCount += 1
            frameFilename = f"{videoName}_{frameCount:04d}.jpg"
            framePath = os.path.join(saveLabelFolder, frameFilename)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save the grayscale frame
            cv2.imwrite(framePath, gray)
        
        cap.release()
        print(f"Saved {frameCount} frames from {videoName}")