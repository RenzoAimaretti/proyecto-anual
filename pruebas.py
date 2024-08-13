import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
import matplotlib.pyplot as plt

# Define class mapping
mapeo_clases = {
    'N': 0,  # Normal
    'V': 1,  # PVC
    # Add other symbols of interest
}

# Function to process a single ECG record
def process_ecg(record_path, annotation_path, sampling_rate):
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, 0]  # Select first channel

    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)

    # Extract features directly into the array
    features = np.column_stack((
        rr_intervals,
        hrv_metrics['HRV_RMSSD'].values,
        np.full(len(rr_intervals), signals['ECG_Clean'].mean()),
        np.full(len(rr_intervals), signals['ECG_Clean'].std())
    ))

    # Handle missing values (if any)
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Load annotations and map to class labels
    annotation = wfdb.rdann(annotation_path, 'atr')
    simbolos = annotation.symbol
    labels = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels

# Process multiple records (modify record list as needed)
records = ['100', '101']  # Add more records
X = []
y = []
for record in records:
    print(f"Procesando registro {record}...")
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation = wfdb.rdann(f'mit-bih-arrhythmia-database/{registro}', 'atr')
    features, labels = process_ecg(record_path, annotation_path, sampling_rate=360)  # Adjust sampling rate if needed
    X.append(features)
    y.append(labels)

# Combine all data (alternative method)
X = np.vstack(X)
y = np.hstack(y)

# Handle class imbalance (optional)
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy, classification report, and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
