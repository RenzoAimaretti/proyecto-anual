import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


#1 PROCESAMIENTO DE LOS DATOS DEL ECG

# Descargar un registro específico desde PhysioNet
# wfdb.dl_database('mitdb', dl_dir='mit-bih-arrhythmia-database')


record = wfdb.rdrecord('mit-bih-arrhythmia-database/102')

# Convertir el registro a un array de numpy
ecg_signal = record.p_signal[:, 0]  # Seleccionar el primer canal (Lead I)
sampling_rate = record.fs  # Obtener la frecuencia de muestreo

# Procesar la señal de ECG con NeuroKit2
signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
signals_subset = signals.iloc[:3000]


nk.ecg_plot(signals_subset, info)

# Mostrar una parte de la señal de ECG filtrada
plt.figure(figsize=(12, 6))
plt.plot(signals['ECG_Clean'][:3000])  # Graficar solo los primeros 3000 puntos
plt.title("Señal de ECG Filtrada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

# Extraer los picos R
rpeaks = info["ECG_R_Peaks"]

# Calcular los intervalos RR en milisegundos
rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # Convertir a milisegundos

# Calcular la HRV RMSSD
hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
rmssd = hrv_metrics['HRV_RMSSD'].values[0]  # Obtener el valor de RMSSD

# Graficar la señal original y las características extraídas
'''
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(signals['ECG_Clean'][:3000])  # Graficar solo los primeros 3000 puntos
plt.title("Señal de ECG Filtrada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")

plt.subplot(212)
plt.plot(rr_intervals, label='RR Intervals (ms)')
plt.axhline(y=rmssd, color='r', linestyle='--', label=f'RMSSD = {rmssd:.2f} ms')
plt.title("Variabilidad de la Frecuencia Cardíaca (HRV)")
plt.xlabel("Intervalos RR")
plt.ylabel("Duración (ms)")
plt.legend()
plt.tight_layout()
plt.show()
'''



#2 CLASIFICACIÓN DE LOS DATOS DEL ECG

#Algoritmo genetico

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def calcular_aptitud_nb(individuo, X, y, alpha=0.01):
    # Seleccionar características basadas en el individuo
    X_seleccionado = X[:, individuo == 1]
    
    # Penalización fuerte si no se seleccionan características
    if X_seleccionado.shape[1] == 0:
        return 0
    
    # Entrenar Naive Bayes
    clasificador = GaussianNB()
    scores = cross_val_score(clasificador, X_seleccionado, y, cv=5)
    
    # Calcular precisión media
    accuracy = scores.mean()
    
    # Penalización por número de características
    penalizacion = alpha * (individuo.sum() / len(individuo))
    
    # Calcular aptitud final
    aptitud = accuracy - penalizacion
    
    return aptitud




