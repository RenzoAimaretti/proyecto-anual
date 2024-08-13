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

# Inicializar listas para almacenar los datos
X_total = []
y_total = []

# Supongamos que estas variables ya están definidas en el código anterior
# rr_intervals, rmssd, sdnn, pnn50, ecg_clean_mean, ecg_clean_std, delineation, ecg_signal, sampling_rate, annotation, mapeo_clases

# Inicializar variables con valores por defecto
p_wave_duration = np.nan
qrs_duration = np.nan
t_wave_amplitude = np.nan
qrs_amplitude = np.nan

# Verificar la existencia de las columnas y calcular las características
if 'ECG_P_Offset' in delineation.columns and 'ECG_P_Onset' in delineation.columns:
    p_wave_duration = np.mean(delineation['ECG_P_Offset'] - delineation['ECG_P_Onset'])
    print(f"Duración de la onda P: {p_wave_duration:.2f}")
else:
    print("Las columnas 'ECG_P_Offset' y/o 'ECG_P_Onset' no existen en el DataFrame.")

if 'ECG_R_Offset' in delineation.columns and 'ECG_Q_Onset' in delineation.columns:
    qrs_duration = np.mean(delineation['ECG_R_Offset'] - delineation['ECG_Q_Onset'])
    print(f"Duración del QRS: {qrs_duration:.2f}")
else:
    print("Las columnas 'ECG_R_Offset' y/o 'ECG_Q_Onset' no existen en el DataFrame.")

if 'ECG_T_Peak' in delineation.columns:
    t_wave_amplitude = np.mean(ecg_signal[delineation['ECG_T_Peak']])
    print(f"Amplitud de la onda T: {t_wave_amplitude:.2f}")
else:
    print("La columna 'ECG_T_Peak' no existe en el DataFrame.")

if 'ECG_R_Peak' in delineation.columns and 'ECG_Q_Peak' in delineation.columns:
    qrs_amplitude = np.mean(ecg_signal[delineation['ECG_R_Peak']] - ecg_signal[delineation['ECG_Q_Peak']])
    print(f"Amplitud del QRS: {qrs_amplitude:.2f}")
else:
    print("Las columnas 'ECG_R_Peak' y/o 'ECG_Q_Peak' no existen en el DataFrame.")

# Análisis de frecuencias (Fourier)
filtered_signal = nk.signal_filter(ecg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=50)
power_spectrum = nk.signal_power(filtered_signal, sampling_rate=sampling_rate, frequency_band=[0.5, 50])

# Verificar si 'Frequency' y 'Power' están en el diccionario
if 'Frequency' in power_spectrum and 'Power' in power_spectrum:
    dominant_freq = power_spectrum['Frequency'][np.argmax(power_spectrum['Power'])]  # Frecuencia dominante
else:
    dominant_freq = np.nan
    print("Las claves 'Frequency' y/o 'Power' no existen en el resultado de signal_power.")    

# Vector de características
X = np.array([rr_intervals.mean(), rmssd, sdnn, pnn50, ecg_clean_mean, ecg_clean_std, 
              p_wave_duration, qrs_duration, t_wave_amplitude, qrs_amplitude, dominant_freq])

# Escalado de características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(1, -1)).flatten()

# Extraer las etiquetas
simbolos = annotation.symbol
y = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

# Asegurarse de que las longitudes de X y y coinciden
if len(X_normalized) == len(y):
    X_total.append(X_normalized)
    y_total.append(y)
else:
    print(f"Longitud desalineada en el registro {registro}")

# Combinar todos los datos
if X_total and y_total:  # Verificar que las listas no estén vacías
    X = np.array(X_total)
    y = np.hstack(y_total)
    min_len = min(len(X), len(y))

    if min_len > 0:
        X = X[:min_len]
        y = y[:min_len]
    else:
        print("No hay datos suficientes para alinear X y y.")
else:
    print("No se han añadido datos a X_total o y_total.")

print(f"Longitud de X: {len(X)}")
print(f"Longitud de y: {len(y)}")


