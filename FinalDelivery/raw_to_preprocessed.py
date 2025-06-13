import os
import csv
import numpy as np
from glob import glob

#Path to the raw datasets folder and output CSV file that comebines all the data
raw_data_folder = './FinalDelivery/raw_datasets'
output_csv = './DinalDelivery/preprocessed_dataset.csv'
# Number of frames per window
# This numer will be used to determine the number of differences between frames using average
N = int(input("Enter the number of frames to group by: ")) 

#This process is used to normalize the landmarks
# So the data is centered and scaled and the size of the person doesn't affect the model
def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape((33, 4))
    
    #This method to determine what each landmark meant was Asked to an AI
    # Center of the body: Average between the hips (landmark 23 y 24)
    center = (coords[23][:3] + coords[24][:3]) / 2
    coords[:, :3] -= center

    # Calculate the "virtual height" of the body between the ankles (27,28) and the head (0)
    left_ankle = coords[27][:3]
    right_ankle = coords[28][:3]
    ankle_center = (left_ankle + right_ankle) / 2

    head = coords[0][:3]
    body_height = np.linalg.norm(head - ankle_center)

    # Backup: if the body height is zeo, then use the distance between shoulders (11,12)
    if body_height == 0:
        shoulder_dist = np.linalg.norm(coords[11][:3] - coords[12][:3])
        body_height = shoulder_dist if shoulder_dist > 0 else 1.0

    coords[:, :3] /= body_height

    return coords.flatten()

#Function to make the average of differences between an N number of frames
def compute_average_differences(landmark_array, window):
    samples = []
    #For every N number of frames calculate the average difference between them
    for i in range(len(landmark_array) - window):
        # This list stores the differences between frames
        diffs = []
        for j in range(window - 1):
            vec1 = np.array(landmark_array[i + j])
            vec2 = np.array(landmark_array[i + j + 1])
            diff = vec2 - vec1
            diffs.append(diff)
        #After adding all the differences, calculate the average
        avg_diff = np.mean(diffs, axis=0)
        #Append the average differences to the samples list
        samples.append(avg_diff)
    return samples

#This list is to store all the data from all the CSV files
all_feature_rows = []

#Get all the CSV files in the raw data folder
csv_files = glob(os.path.join(raw_data_folder, '*.csv'))

#For loop that process each CSV file
for csv_path in csv_files:
    label = os.path.splitext(os.path.basename(csv_path))[0]  # Use filename as label
    print(f"***Processing: {label}***")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Read all rows as floats
        data = [normalize_landmarks(list(map(float, row))) for row in reader]

    #Verify that therea re enough frames to process
    if len(data) < N:
        print(f"Not enough frames in {label} to process window of N={N}")
        continue
    
    #Call the method to calculate average differences
    averaged_differences = compute_average_differences(data, N)

    #Append the averaged differences to the general feature list
    for average_vector in averaged_differences:
        #Append the average vector but also label each row with the label of the CSV file
        all_feature_rows.append(list(average_vector) + [label])

# Check if there are any feature rows to write on
if all_feature_rows:
    print(f"***Writing preprocessed dataset to {output_csv}***")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Generate header for the CSV file
        header = []
        for i in range(33):  # 33 landmarks
            header += [f"dx{i}", f"dy{i}", f"dz{i}", f"dv{i}"]
        header.append("label")
        # Write the header to the CSV file
        writer.writerow(header)
        #Write all the feature rows to the CSV file
        writer.writerows(all_feature_rows)

    print("Preprocessing complete.")
else:
    print("No data to write.")