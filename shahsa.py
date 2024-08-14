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
    # Latidos normales
    'N': 0,  # Latido normal

    # Contracciones Prematuras (valor común: 1)
    'V': 1,  # Contracción ventricular prematura (PVC)
    'E': 1,  # Latido de escape ventricular
    'F': 1,  # Latido de fusión (Fusion of ventricular and normal beat)
    'J': 1,  # Latido de escape nodal (Nodal (junctional) escape beat)
    'A': 1,  # Contracción auricular prematura (Atrial premature beat)
    'a': 1,  # Contracción auricular aberrante (Aberrated atrial premature beat)
    
    # Latido de Fusión (valor común: 2)
    '/': 2,  # Latido de fusión de latido normal y PVC (Fusion of paced and normal beat)
    'f': 2,  # Latido de fusión de latido normal y latido aberrante (Fusion of paced and ventricular beat)
    
    # Contracciones Auriculares (valor común: 3)
    'L': 3,  # Latido de escape del nodo SA (Left bundle branch block beat)
    'R': 3,  # Latido de escape ventricular (Right bundle branch block beat)
    'S': 3,  # Contracción supraventricular prematura (Supraventricular premature beat)
    'P': 3,  # Latido por marcapasos (Paced beat)
    
    # Otros (valor común: 4)
    'Q': 4,  # Latido QRS aberrante (Unclassifiable beat)
    'e': 4,  # Latido de escape ventricular retardado (Ventricular escape beat)
    '!': 4,  # Latido ectópico nodal (Ventricular flutter wave)
    'I': 4,  # Latido idioventricular (Ventricular flutter wave)
    'i': 4,  # Latido de escape idioventricular (Ventricular flutter wave)
    '+': 4,  # Ritmo cambiante (Rhythm change)
    '~': 4,  # Cambio en la frecuencia cardiaca (Signal quality change)
    '|': 4,  # Comienzo de segmento de segmento (Isolated QRS-like artifact)
    's': 4,  # Latido sistólico (Systole)
    'T': 4,  # Latido ventricular no capturado (T-wave peak)
    '*': 4,  # Artefacto (Systole)
    'x': 4,  # Latido aberrante (Waveform onset)
    '[': 4,  # Comienzo de una pausa (P-wave peak)
    ']': 4,  # Fin de una pausa (Waveform end)
    'p': 4,  # Potencial del marcapasos (Non-conducted pacer spike)
    'B': 4,  # Bloqueo de rama (Left bundle branch block)
    'b': 4,  # Bloqueo de rama incompleto (Right bundle branch block)
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

records = ['100', '101','102','103','104']
X_total, y_total = [], []
for record in records:
    print(f"Procesando registro {record}...")
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation = f'mit-bih-arrhythmia-database/{record}'
    features, labels_agrupados = procesar_ecg(record_path, annotation)

    print(f"features length: {len(features)}")
    print(f"labels length: {len(labels_agrupados)}")

    min_len = min(len(features), len(labels_agrupados))
    features = features[:min_len]
    labels_agrupados = labels_agrupados[:min_len]

    print(f"features length: {len(features)}")
    print(f"labels length: {len(labels_agrupados)}")

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
    1: 'Contracciones Prematuras',
    2: 'Latido de Fusión',
    3: 'Contracciones Auriculares',
    4: 'Otros'
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
y_test_counts = Counter(y_test_reindex)
y_pred_counts = Counter(y_pred_reindex)


print("Counts in y_test:")
for label, count in y_test_counts.items():
    print(f"{label}: {count}")

print("Counts in y_pred:")
for label, count in y_pred_counts.items():
    print(f"{label}: {count}")
print("y_test_reindex:", y_test)
print("y_pred_reindex:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
nombres_clases_agrupadas = ['Normal', 'Prematuras','Fusion','Contraccion Auricular', 'Otros']

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=nombres_clases_agrupadas, yticklabels=nombres_clases_agrupadas)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Clases Agrupadas")
plt.show()