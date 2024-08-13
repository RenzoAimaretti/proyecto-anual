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

# Definición de los registros y mapeo de clases
registros = ['100', '101']  # Añadir más registros aquí
X_total = []
y_total = []
mapeo_clases = {
    'N': 0,  # Latido normal
    'V': 1,  # PVC
    # Añadir otros símbolos que te interesen
}

for registro in registros:
    # Lectura de los registros y anotaciones
    print(f"Procesando registro {registro}...")
    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database/{registro}')
    annotation = wfdb.rdann(f'mit-bih-arrhythmia-database/{registro}', 'atr')
    ecg_signal = record.p_signal[:, 0]  # Seleccionar el primer canal
    sampling_rate = record.fs

    # Procesamiento del ECG
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]
    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
    
    # Verificar que las métricas HRV están presentes
    rmssd = hrv_metrics.get('HRV_RMSSD', pd.Series([np.nan])).values[0]
    sdnn = hrv_metrics.get('HRV_SDNN', pd.Series([np.nan])).values[0]
    pnn50 = hrv_metrics.get('HRV_pNN50', pd.Series([np.nan])).values[0]
    
    ecg_clean_mean = signals['ECG_Clean'].mean()
    ecg_clean_std = signals['ECG_Clean'].std()

    # Análisis de la morfología del ECG
    delineation, waves_info = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method="dwt")
    
    p_wave_duration = np.nan
    if 'ECG_P_Offset' in delineation.columns and 'ECG_P_Onset' in delineation.columns:
        p_wave_duration = np.mean(delineation['ECG_P_Offset'] - delineation['ECG_P_Onset'])
        
    qrs_duration = np.nan
    if 'ECG_R_Offset' in delineation.columns and 'ECG_Q_Onset' in delineation.columns:
        qrs_duration = np.mean(delineation['ECG_R_Offset'] - delineation['ECG_Q_Onset'])
    
    t_wave_amplitude = np.nan
    if 'ECG_T_Peak' in delineation.columns:
        t_wave_amplitude = np.mean(ecg_signal[delineation['ECG_T_Peak']])
    
    qrs_amplitude = np.nan
    if 'ECG_R_Peak' in delineation.columns and 'ECG_Q_Peak' in delineation.columns:
        qrs_amplitude = np.mean(ecg_signal[delineation['ECG_R_Peak']] - ecg_signal[delineation['ECG_Q_Peak']])
    
    # Análisis de frecuencias (Fourier)
    filtered_signal = nk.signal_filter(ecg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=50)
    power_spectrum = nk.signal_power(filtered_signal, sampling_rate=sampling_rate, frequency_band=[0.5, 50])
    
    dominant_freq = np.nan
    if 'Frequency' in power_spectrum and 'Power' in power_spectrum:
        dominant_freq = power_spectrum['Frequency'][np.argmax(power_spectrum['Power'])]  # Frecuencia dominante
    
    X = np.array([rr_intervals.mean(), rmssd, sdnn, pnn50, ecg_clean_mean, ecg_clean_std, 
                  p_wave_duration, qrs_duration, t_wave_amplitude, qrs_amplitude, dominant_freq])
    
    X_total.append(X)

    # Extraer las etiquetas
    simbolos = annotation.symbol
    y = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    # Asegurarse de que las longitudes de X y y coinciden
    min_len = min(len(X), len(y))
    if min_len > 0:
        X_total.append(X[:min_len])
        y_total.append(y[:min_len])
    else:
        print(f"Longitud desalineada en el registro {registro}")

# Combinar todos los datos
if X_total and y_total:  # Verificar si las listas no están vacías
    X = np.array(X_total)
    y = np.hstack(y_total)

    min_len = min(len(X), len(y))
    if min_len > 0:
        X = X[:min_len]
        y = y[:min_len]
    else:
        print("No hay datos suficientes para alinear X y y.")
    
    print(f"Longitud de X: {len(X)}")
    print(f"Longitud de y: {len(y)}")

    # Escalado de características
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # Imputar los valores NaN en X con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_normalized)
    # Verificar las clases en y
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Clases en y: {unique_classes}, con conteos: {counts}")
    
    # Comprobar si hay más de una clase en y
    if len(unique_classes) > 1:
        # Aplicar SMOTE si hay más de una clase
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
        # Entrenar el modelo
        clasificador = GaussianNB()
        clasificador.fit(X_train, y_train)
        y_pred = clasificador.predict(X_test)
    
        # Calcular la precisión del modelo
        precision = accuracy_score(y_test, y_pred)
        print(f"Precisión: {precision:.2f}")
        unique_classes = np.unique(y_test)
        print(f"Clases en y_test: {unique_classes}")
    
        # Reporte de clasificación y matriz de confusión
        reporte = classification_report(y_test, y_pred, target_names=['Normal', 'PVC'])
        print("Reporte de Clasificación:\n", reporte)
        matriz_confusion = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusión:\n", matriz_confusion)
    
        # Graficar la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'PVC'], yticklabels=['Normal', 'PVC'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Matriz de Confusión")
        plt.show()
    else:
        print("No hay suficientes clases en y para aplicar SMOTE.")
    