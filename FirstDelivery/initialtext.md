Project title: Real-Time Human Activity Recognition

Group Members:
- Alejando Londoño
- Juan Diego Lora
- Simón García

Research Questions:
- How can a basic machine learning model classify human activities in real time using joint position data?
- What are the best models for this real-time classification process?
- How would different individuals and perspectives affect the estimation process?
- What is the highest possible accuracy achievable with the selected model?
- How can we improve generalization when taking into account speed, body type, clothing and others?

Problem type: Supervised Classification

Methodology:¨Crisp-DM Adaptation
- Business understanding: For this proyect it involves understanding the task iteself. This project is looking to make a model that allows to identify movements and some specific positions that a person is doing. This involves feeding the model a continous stream of information using a camera. The model should gain information from previous recordings on which it will learn to classify this actions based on the angle of joints and the relative distance between them.
- Data understanding: The data will include a lot of information that comes as the raw data that is gathered by the camera. The information comes as a stream of numbers that display the pixel-like information associated to the images in every capture. 
- Data preparation: The data must be prepared in a way so the model is not only able to process the data correctly (so the answer makes sense) but also that the results can be calculated efficiently. The data will be transformed into an gray scale and then everything will be transformed through PCA to reduce the dimentionality of the dataset which allow less but more meaningful information to be processed (reducing the amount of pixels). The data will also be normalized to reduce any problems with scales and the effects of outliers.
- Modeling: For the model an XGBoost/RandomForest/etc will be used to process the live data and then make the appropiate predictions. For this situation a robust ensamble models would be more realiable to reduce variance and bias. The model's hyperparameters will be tuned using Grid Search to find the optimal values from a set of preestablish ones. 
- Evaluation stategy: For this project we are planning to use accuracy, recall and F1-score to measure the performance of the model. There will be tests not only for general stream of information, but we will also check if some activities and body positions are easier to detect than others.
- Next Steps: We must collect the data in an orderly manner, making sure to control as much the conditions in which the data is gathered (striving for homogeneity). We must train and thest the models and adjust them so the predictions are accurate. A pipeline must be implemented so the stream of information is processed seamlessly through the model and it can make work efficiently. Finally there must be a basic GUI design to display the results of the model.

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
- Follow keymovements using mediapipe or OpenPose.

Exploratory Data Analysis:
- The exploratory analysis of the data will be done in order to determine the general quality of the data and how how the pipeline must be implemented in order to work with the data. For this process the appropiate visualizations like histograms and boxplot diagrams will be useful. Additionaly, checking the data for outliers and missing data must be done in order to determine the quality of the recorded videos.

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
