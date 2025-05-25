import os
import csv
import numpy as np
from glob import glob

raw_data_folder = './SecondDelivery/raw_datasets'
output_csv = './SecondDelivery/preprocessed_dataset.csv'
N = int(input("Enter the number of frames to group by: "))  # Number of frames per window

#Function to make the average of differences between an N number of frames
def compute_average_differences(landmark_array, window):
    samples = []
    for i in range(len(landmark_array) - window):
        diffs = []
        for j in range(window - 1):
            vec1 = np.array(landmark_array[i + j])
            vec2 = np.array(landmark_array[i + j + 1])
            diff = vec2 - vec1
            diffs.append(diff)
        avg_diff = np.mean(diffs, axis=0)
        samples.append(avg_diff)
    return samples

all_feature_rows = []

#Get all the CSV files in the raw data folder
csv_files = glob(os.path.join(raw_data_folder, '*.csv'))

for csv_path in csv_files:
    label = os.path.splitext(os.path.basename(csv_path))[0]  # Use filename as label
    print(f"***Processing: {label}***")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Read all rows as floats
        data = [list(map(float, row)) for row in reader]

    if len(data) < N:
        print(f"⚠️ Not enough frames in {label} to process window of N={N}")
        continue

    averaged_differences = compute_average_differences(data, N)

    for average_vector in averaged_differences:
        all_feature_rows.append(list(average_vector) + [label])

if all_feature_rows:
    print(f"***Writing preprocessed dataset to {output_csv}***")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Generate header
        header = []
        for i in range(33):  # 33 landmarks
            header += [f"dx{i}", f"dy{i}", f"dz{i}", f"dv{i}"]
        header.append("label")

        writer.writerow(header)
        writer.writerows(all_feature_rows)

    print("✅ Preprocessing complete.")
else:
    print("⚠️ No data to write.")