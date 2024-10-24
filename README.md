# Predict Diseases using Blood Samples

This project demonstrates how to build a machine learning model to predict a person's health status (e.g., healthy, disease like diabetes, thalassemia, etc..) based on their blood sample data. The model is trained on a balanced dataset using Random Forest Classifier. The project also includes saving the model and label encoder for future predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [License](#license)

## Introduction

This repository contains code for a machine learning model that predicts whether a person is healthy or has a specific disease based on blood sample data. The model is saved as `predict_disease_model_with_encoder.pkl` which contains both the trained model and the `LabelEncoder` to map numerical labels to class names.

## Dataset

The datasets used in this project are included in the repository:
- `Blood_samples_dataset_balanced.csv`: The training dataset with balanced classes.
- `people_blood_records.csv`: A sample dataset with blood records to predict health status.

## Project Structure
``` bash
Predict_Diseases_Using_Blood_Samples/
├── datasets/                                          # Folder containing datasets
│   ├── Blood_samples_dataset_balanced.csv
│   └── people_blood_records.csv
│
├── model/                                            # Folder containing saved models
│   └── predict_disease_based_on_blood_samples.pkl
│
├── Predict_diseases_using_blood_samples.ipynb         # Jupyter notebook containing code
├── README.md                                          # Project documentation
└── requirements.txt                                   # Python dependencies for the project
```

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/SinethB/predict-diseases-using-blood-samples.git
2. Install the required dependencies by running the following command:
   ```bash
   pip install -r requirements.txt

## Usage

1. **Training the Model:**  
   The model is trained on `Blood_samples_dataset_balanced.csv`. The Random Forest classifier is used, and both the model and the `LabelEncoder` are saved in `predict_disease_based_on_blood_samples.pkl`
2. **Making Predictions: ** 
   To make predictions on new data, load the saved model and use it to predict the health status of people from a new dataset (`people_blood_records.csv`). Here's how you can run it:

  - Predict from CSV
```python
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a Pandas DataFrame
people_df = pd.read_csv('people_blood_records.csv')

# Display the first few rows to verify
print("Uploaded data preview:")
print(people_df.head(7))

# Separate the 'Name' column and the blood record features
names = people_df['Name']
X_new = people_df.drop(columns=['Name'])  # features

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)  # Scale the features using the same scaler

# Load the saved model and label encoder together
with open('predict_disease_based_on_blood_samples.pkl', 'rb') as model_file:
    model_and_encoder = pickle.load(model_file)

# Extract the model and the label encoder from the dictionary
model = model_and_encoder['model']
le = model_and_encoder['label_encoder']

# Make predictions using the loaded model
y_pred_numeric = model.predict(X_new_scaled)

# Convert numeric predictions back to original labels using the LabelEncoder
y_pred_labels = le.inverse_transform(y_pred_numeric)

# Print the result (replace n with actual number of rows. here 7)
results_df = pd.DataFrame({
'Name': names,
'Predicted Health Status': y_pred_labels
})
results_df.head(n)

# Display the counts for each health status
health_status_counts = results_df['Predicted Health Status'].value_counts()
print("\nHealth Status Counts:")
print(health_status_counts)

# Save the results to a CSV file 
people_df.to_csv('predicted_health_status.csv', index=False)
```
3. **Model Perfomance:**
   The model was evaluated using a confusion matrix, and the following are some of the performance metrics:
    ```bash
    Accuracy: 100%
    Precision: 100%
    Recall: 100%

## Model Performance
You can evaluate the model’s performance using a confusion matrix and calculate other metrics such as precision, recall, and accuracy.

To visualize the confusion matrix:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
```
## License
This project is licensed under the MIT License.
