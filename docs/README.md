# Technical Documentation

Feel free to add any number of markdown files in this folder that can help us better understand your solution.
You can include the following (not an exhaustive list, feel free to add more):

- Your approach to solve this problem and what makes it unique

**Problem Statement 8:Classify User Application Traffic at the Network in a Multi-UE Connected Scenario**
**Project Documentation**
Our Approach to Solve the Problem Statement 8:
Our solution focuses on real-time network traffic classification.
We trained a Random Forest Classifier on pre-collected traffic data, scaled the features using StandardScaler, and encoded categorical labels with LabelEncoder.
During live capture, we extract the same features, align them with training features, scale them, and feed them into the trained model for prediction.
What makes our solution unique:
Consistency: The feature alignment ensures live data matches training data exactly.
Explainability: Predictions are mapped back to human-readable traffic categories.
Extensibility: Easy to plug in other ML models or additional features.

- Technical Stack - List of OSS libraries/projects used along with their links
Python 3.10+
scikit-learn
 – ML models, preprocessing
pandas
 – Data handling
numpy
 – Numerical operations
joblib
 – Model persistence
pyshark
 – Live packet capture

- The technical architecture of your solution

- Implementation details
    Data Collection
Dataset prepared with network flow features.
Each row labeled as web, dns, p2p, etc.
Model Training
Preprocess features (scaling, encoding).
Train RandomForestClassifier.
Save traffic_classifier.pkl, scaler.pkl, and label_encoder.pkl.
Live Prediction (a2172554...py)
Capture packets → Extract features.
Align with training columns.
Apply scaler.
Persistence: Models and encoders stored with joblib.
Prediction Output: Converted from numbers → class names → human-friendly descriptions.

- Installation instructions
- User guide
- Salient features of the project

Note: Kindly add screenshots wherever possible.
