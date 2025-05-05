# Project Title: Real-Time Human Activity Recognition

## Group Members:
- Alejandro Londoño
- Juan Diego Lora
- Simón García

## Research Questions:
- How can a basic machine learning model classify human activities in real time using joint position data?
- What are the best models for this real-time classification process?
- How would different individuals and perspectives affect the estimation process?
- What is the highest possible accuracy achievable with the selected model?
- How can we improve generalization when taking into account speed, body type, clothing and others?

## Problem type: Supervised Classification

## Methodology: CRISP-DM Adaptation
- **Business Understanding**: This project aims to build a model that identifies movements and specific positions a person is performing. The model will process a continuous stream of data using a camera. The model will learn to classify actions based on joint angles and relative distances between them.
- **Data Understanding**: The data will come from a continuous stream of images captured by the camera, which will be transformed into numerical data corresponding to pixel-like information for each capture.
- **Data Preparation**: The raw data will be converted into grayscale, then transformed using PCA (Principal Component Analysis) to reduce dimensionality. This will allow the model to process meaningful data with fewer features. Additionally, the data will be normalized to reduce the impact of scale differences and outliers.
- **Modeling**: We will use machine learning models such as XGBoost and RandomForest, and employ ensemble methods to ensure reliability and reduce variance and bias. Hyperparameter tuning will be done using Grid Search to find the optimal settings.
- **Evaluation Strategy**: The performance of the model will be measured using accuracy, recall, and F1-score. We will also test if certain actions and positions are easier to detect than others.
- **Next Steps**: The data needs to be collected in a controlled manner, with attention to homogeneity. Afterward, models will be trained and tested, fine-tuned for accurate predictions, and a pipeline will be built to process the real-time data stream. Finally, a simple GUI will be designed to display the model's results.

## Block Diagram

![Block Diagram](images/Classification%20Model.jpg)

## Data Collection Strategy:
### General:
- Data will be recorded using a preinstalled laptop camera to guarantee standard conditions.
- The computer must be placed over a table, with the screen at a 90° angle pointing toward an open area for the actions to take place.
- Recordings will vary in length and will be categorized by action type.

### Considerations:
- Only one person should be visible in the camera's frame at a time.
- Lighting conditions should be homogeneous throughout the data collection.
- The background should vary to allow the model to adapt to different scenarios.
- Multiple people with varying types of clothing should be recorded to increase variability.
- A plain background (e.g., wall or static background) should be used to reduce noise and improve prediction accuracy.

## Annotation Strategy:
- Semi-manual annotation will be performed using LabelStudio.
- Key movements will be tracked using MediaPipe or OpenPose.

## Exploratory Data Analysis:
- Exploratory analysis will be performed to assess data quality and determine how the pipeline should be structured. Visualizations like histograms and boxplots will be used to identify outliers and missing data.

## Performance Metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Recall**: Measures the completeness of each action and posture classification.
- **F1-score**: Balances precision and recall to evaluate the model's correctness.

## Ethical Considerations:
- **Privacy**: Ensure that collected videos are only used for educational purposes within the scope of the project.
- **Consent**: All participants must be informed and consent to having their videos recorded and used for the project.
- **Fairness**: A diverse group of participants will be used to ensure generalization and fairness in the model.

## Tools and Resources:
- **Python**, **NumPy**, **Pandas**, **scikit-learn**, **Matplotlib**: Used for data handling, model building, visualizations, and metric evaluations.
- **Label Studio**: For annotating video recordings.
- **MediaPipe**: To capture and analyze specific joint movements.
- **GitHub**: For version control and collaboration.

## References:
[1] C. Lugaresi *et al.*, "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint arXiv:1906.08172*, 2019. [Online]. Available: [https://arxiv.org/abs/1906.08172](https://arxiv.org/abs/1906.08172)

[2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.
