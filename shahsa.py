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
    'N': 0,  # Grupo 0: Latidos normales
    'L': 1,  # Grupo 1: Bloqueo de rama (izquierda y derecha)
    'R': 1,
    'A': 2,  # Grupo 2: Contracciones prematuras (auricular y ventricular)
    'V': 2,
    'F': 3,  # Grupo 3: Latidos de fusión
    'Q': 4,  # Grupo 4: Latidos QRS aberrantes
    'P': 4,  # Grupo 4: Latidos por marcapasos (englobado con QRS aberrante)
}

def procesar_ecg(r, a):
    record = wfdb.rdrecord(r)
    anotation = wfdb.rdann(a, 'atr')
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

    labels_agrupados = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels_agrupados

records = ['100', '101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
X_total, y_total = [], []
for record in records:
    print(f"Procesando registro {record}...")
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation = f'mit-bih-arrhythmia-database/{record}'
    features, labels_agrupados = procesar_ecg(record_path, annotation)
    X_total.append(features)
    y_total.append(labels_agrupados)

X = np.vstack(X_total)
y = np.hstack(y_total)

print("Length of X:", len(X))
print("Length of y:", len(y))

min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len] 

print("Length of X:", len(X))
print("Length of y:", len(y))

smote=SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

clasificador = GaussianNB()
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)

reindex_mapeo = {
    0: 'Latido Normal',
    1: 'Bloqueo de Rama',
    2: 'Contracciones Prematuras',
    3: 'Latido de Fusion',
    4: 'Latido QRS Aberrante'
    # Asegúrate de incluir todas las etiquetas posibles
}
# Proporcionar un valor predeterminado para las claves que faltan
default_value = 'Otro'


'''LAS ETIQUETAS ESTAN MAL REINDEXADAS DAN DIFERENTE A LO QUE DEBERIA SER


y_test_reindex: ['Contracciones Prematuras' 'Latido Normal' 'Latido QRS Aberrante' ...
 'Latido QRS Aberrante' 'Contracciones Prematuras'
 'Contracciones Prematuras']

y_pred_reindex: ['Latido QRS Aberrante' 'Latido QRS Aberrante' 'Latido QRS Aberrante' ...
 'Latido QRS Aberrante' 'Latido QRS Aberrante' 'Contracciones Prematuras']


'''
y_test_reindex = np.array([reindex_mapeo.get(label, default_value) for label in y_test])
y_pred_reindex = np.array([reindex_mapeo.get(label, default_value) for label in y_pred])
# Verificar las etiquetas reindexadas
print("y_test_reindex:", y_test_reindex)
print("y_pred_reindex:", y_pred_reindex)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test_reindex, y_pred_reindex, labels=list(reindex_mapeo.values()),zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test_reindex, y_pred_reindex))

# Visualize confusion matrix
nombres_clases_agrupadas = ['Normal', 'Bloqueo de Rama', 'Prematura', 'Otros']

cm = confusion_matrix(y_test_reindex, y_pred_reindex)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=nombres_clases_agrupadas, yticklabels=nombres_clases_agrupadas)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Clases Agrupadas")
plt.show()