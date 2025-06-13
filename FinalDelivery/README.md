# Requirements

* Python 3.11.0 (required for MediaPipe compatibility)
* pip package manager

# Install Python dependences
```bash
python3.11 -m pip install -r requirements.txt
```

First, by recommendation, manage your own Python environment
```bash
python3.11 -m venv venv
cd venv/Scripts
./activate
```

# Landmarks Processing

This process extracts pose landmarks from images and saves them in a CSV file for further project use.

### How Landmarks are Saved:

* Each processed image that contains detected pose landmarks corresponds to one row in the CSV file.
* Each row contains a flattened list of landmark coordinates and visibility scores, followed by the image’s class label.

### Landmark Format per Image:

* There are 33 pose landmarks detected by MediaPipe.
* Each landmark consists of 4 values:

  * `x`: normalized horizontal coordinate (float, approx. 0 to 1)
  * `y`: normalized vertical coordinate (float, approx. 0 to 1)
  * `z`: normalized depth coordinate (float, approx. 0 to 1)
  * `visibility`: confidence score indicating landmark visibility (float, 0 to 1)
* The landmarks are stored in a flattened order as:
  `[x_0, y_0, z_0, v_0, x_1, y_1, z_1, v_1, ..., x_32, y_32, z_32, v_32]`
* The class label (string) is appended at the end of each row.

### CSV Structure:

* Total columns: 33 landmarks × 4 values each = 132 numeric columns + 1 label column = **133 columns**.
* Header example:
  `["x_0", "y_0", "z_0", "v_0", ..., "x_32", "y_32", "z_32", "v_32", "label"]`
* Each row corresponds to a single image’s landmark data and label.