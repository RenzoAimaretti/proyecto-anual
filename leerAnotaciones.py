import wfdb
import pandas as pd
import neurokit2 as nk
import numpy as np
# Ruta al archivo .atr sin la extensión
file_base = 'mit-bih-arrhythmia-database/103'

# Leer el archivo .atr
record = wfdb.rdrecord(file_base)
annotations = wfdb.rdann(file_base, 'atr')
ecg_signal = record.p_signal[:, 0]
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

# Print the features
print("Features:")
df_features = pd.DataFrame(features, columns=['RR Intervals', 'RMSSD', 'Mean ECG', 'Std ECG'])
print(df_features)


# Mostrar información sobre el archivo
print(f"Anotaciones para el archivo {file_base}.atr:")
print(f"Duración: {record.sig_len / record.fs} segundos")
print(f"Señales: {record.n_sig}")
# Crear un DataFrame con las anotaciones
df = pd.DataFrame({
    'Tipo': annotations.symbol
})

# Realizar el recuento de cada tipo de símbolo
count = df['Tipo'].value_counts()
print(f"features length: {len(features)}")
# Mostrar el recuento
print(count)
# Imprimir la sumatoria de anotaciones
print(f"Sumatoria de anotaciones: {count.sum()}")


