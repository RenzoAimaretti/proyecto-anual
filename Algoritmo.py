import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter


# Por ejemplo, puedes procesar varios registros y combinarlos
registros = ['100', '101']  # Añadir más registros aquí
X_total = []
y_total = []
mapeo_clases = {
    'N': 0,  # Latido normal
    'V': 1,  # PVC
    # Añadir otros símbolos que te interesen
}
for registro in registros:
    print(f"Procesando registro {registro}...")
    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database/{registro}')
    annotation = wfdb.rdann(f'mit-bih-arrhythmia-database/{registro}', 'atr')
    ecg_signal = record.p_signal[:, 0]  # Seleccionar el primer canal
    sampling_rate = record.fs

    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
    rmssd = hrv_metrics['HRV_RMSSD'].values[0]

    # Extraer las características y anotaciones
    X = np.column_stack((
        rr_intervals,
        np.full(len(rr_intervals), rmssd),
        np.full(len(rr_intervals), signals['ECG_Clean'].mean()),
        np.full(len(rr_intervals), signals['ECG_Clean'].std())
    ))

    simbolos = annotation.symbol
    y = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    # Alinea X e y y agrégalo a las listas totales
    min_len = min(len(X), len(y))
    X_total.append(X[:min_len])
    y_total.append(y[:min_len])

# Combinar todos los datos
X = np.vstack(X_total)
y = np.hstack(y_total)


# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clasificador = GaussianNB()
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión: {precision:.2f}")

# Reporte de clasificación
reporte = classification_report(y_test, y_pred, target_names=['Normal', 'PVC'])
print("Reporte de Clasificación:\n", reporte)

# Matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", matriz_confusion)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'PVC'], yticklabels=['Normal', 'PVC'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.show()