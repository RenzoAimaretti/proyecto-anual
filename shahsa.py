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

mapeo_clases = {
    'N': 0,  # Normal
    'V': 1,  # PVC
    
}

def procesar_ecg(r, a):
    record = wfdb.rdrecord(r)
    anotation = wfdb.rdann(a, 'atr')
    simbolos = anotation.symbol
    labels = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])
    ecg_signal = record.p_signal[:, 0]  # Seleccionar el primer canal

    sampling_rate = record.fs
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=record.fs)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
    rmssd = hrv_metrics['HRV_RMSSD'].values[0]

    # Extract features
    features = np.column_stack((
        rr_intervals,
        np.full(len(rr_intervals), rmssd),
        np.full(len(rr_intervals), signals['ECG_Clean'].mean()),
        np.full(len(rr_intervals), signals['ECG_Clean'].std())
    ))

    # Manejar valores faltantes (si los hay)
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # Estandarizar características
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Cargar anotaciones y mapear a etiquetas de clase
    simbolos = anotation.symbol
    labels = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels

records = ['100', '101','102','103','104','105','106','107','108','109']
X_total, y_total = [], []
for record in records:
    print(f"Procesando registro {record}...")
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation = f'mit-bih-arrhythmia-database/{record}'
    features, labels = procesar_ecg(record_path, annotation)
    X_total.append(features)
    y_total.append(labels)

X = np.vstack(X_total)
y = np.hstack(y_total)

min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

smote=SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

clasificador = GaussianNB()
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,target_names=['Normal', 'PVC']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()