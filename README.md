# Disease Prediction Using Blood Samples

## Overview
An advanced machine learning system that predicts various health conditions (including diabetes and thalassemia) using blood sample analysis. The model achieves 93% accuracy using a Random Forest Classifier trained on a balanced dataset, with integrated label encoding for seamless prediction deployment.

## Key Features
- Multi-disease classification capability
- Automated data preprocessing and scaling
- Integrated label encoding for categorical predictions
- Real-time prediction interface
- Batch prediction support for multiple records
- Results export functionality

## Technical Implementation

### Machine Learning Pipeline
- **Model**: Random Forest Classifier
- **Preprocessing**: StandardScaler for feature normalization
- **Data Encoding**: LabelEncoder for disease categories
- **Serialization**: Pickle for model persistence
- **Performance**: 100% accuracy on test set

### Data Processing
```python
# Feature scaling implementation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model and encoder packaging
model_package = {
    'model': trained_model,
    'label_encoder': label_encoder
}
```

## Project Structure
```
Predict_Diseases_Using_Blood_Samples/
├── datasets/                                 
│   ├── Blood_samples_dataset_balanced.csv    # Training dataset
│   └── people_blood_records.csv              # Test records
├── model/                                    
│   └── predict_disease_based_on_blood_samples.pkl  # Serialized model
├── Predict_diseases_using_blood_samples.ipynb
├── README.md                                 
└── requirements.txt                          
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/SinethB/predict-diseases-using-blood-samples.git
cd predict-diseases-using-blood-samples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

### Single Prediction
```python
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and encoder
with open('predict_disease_based_on_blood_samples.pkl', 'rb') as file:
    model_package = pickle.load(file)

model = model_package['model']
encoder = model_package['label_encoder']

# Prepare input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(input_data)

# Get prediction
prediction = model.predict(X_scaled)
disease = encoder.inverse_transform(prediction)[0]
```

### Batch Prediction
```python
# Read CSV data
df = pd.read_csv('people_blood_records.csv')
names = df['Name']
features = df.drop(columns=['Name'])

# Scale features
X_scaled = scaler.fit_transform(features)

# Predict
predictions = model.predict(X_scaled)
results = pd.DataFrame({
    'Name': names,
    'Predicted Health Status': encoder.inverse_transform(predictions)
})
```

## Model Performance Metrics
- **Accuracy**: 93%
- **Precision**: 95%
- **Recall**: 92%

### Confusion Matrix Visualization
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
```

## Dataset Description
- **Training Data**: `Blood_samples_dataset_balanced.csv`
  - Balanced class distribution
  - Multiple disease categories
- **Test Data**: `people_blood_records.csv`
  - Real-world blood sample records
  - Named entries for individual tracking

## Future Enhancements
- [ ] Web API implementation
- [ ] Additional ML algorithms comparison
- [ ] Feature importance analysis
- [ ] Cross-validation implementation
- [ ] Interactive dashboard for predictions

## Technical Requirements
- Python 3.10
- scikit-learn
- pandas
- numpy
- seaborn (for visualization)

## License
MIT License
