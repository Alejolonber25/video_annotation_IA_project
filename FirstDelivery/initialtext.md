Proyect title: Real-Time Human Activity Recognition (?)

Research Questions:
- How can a basic machine learning model classify human activities in real time using joint position data?
- What are the best models for this real-time classification process?
- How would different individuals and perspectives affect the estimation process?
- What is the highest possible accuracy achievable with the selected model?
- How can we improve generalization when taking into account speed, body type, clothing, etc.

Problem type: Supervised classification (With time series analysis?)

Methodology:¨Crisp-DM Adaptation
- Business understanding
- Data understanding
- Data preparation
- Modeling
- Evaluation stategy
- Next Steps
Esto lo completo luego, les pregunto una vaina antes.

Data Collection Strategy:
General:
- Data will be recorded using a preinstalled laptop camera to guarantee standard conditions
- The computer must be placed over a table, with the screen on a 90° angle pointing towards an open area where the actions are going to take place.
- The recordings will be of different length and categorized by the type of actions.
Considerations:
- There can only be one person on the camera's point of view at any given time.
- The lighting should try to be homogenous throughout the data gathering and when running the model itself.
- Background should vary, to allow the model to learn different scenarios.
- Record multiple people with different types of clothing to add variability
- There should be a plain background, wall or static background so the noice is reduced during the data collection and the preditions to be more accurate.
Anotation Strategy:
- Semimanual annotation will be done using LabelStudio
- Follow keymovements using mediapip or OpenPose.

Exploratory Data Analysis:
- Lleno despues, pregunto

Performance Metrics:
- Accuracy: To measure the overall correctness of the model.
- Recall: Measure the completeness of every action and posture classification.
- F1-score: To include the balance between precision and recall to also measure the correctness of the model.

Ethical Considerations:
- Privacy: Ensure the collected videous of people are only used for education purposes and the extents explored within this project.
- Consent: All participants must be informed before recording the videos and know they will be used on a project.
- Fairness: Generalization must be ensured by using a diverse group of individuals.

Tools and Resources:
- Python, Numpy, Pandas, sckit-learn and Matplotlib: Used to handle all the data, models, visutalizations and measuring metrics.
- Label Studio for anotations on the recordings.
- MediaPipe to capture and measure specific joint movements.
- Github as the online repository for version control and collabotarion.

Group Members:
- Alejando
- Juan
- Simón García (SimonGarcia01

Tools and Resources:
- Python, Numpy, Pandas, sckit-learn and Matplotlib: Used to handle all the data, models, visutalizations and measuring metrics.
- Label Studio for anotations on the recordings.
- MediaPipe to capture and measure specific joint movements.
- Github as the online repository for version control and collabotarion.

Group Members:
- Alejando
- Juan
- Simón García (SimonGarcia0)
