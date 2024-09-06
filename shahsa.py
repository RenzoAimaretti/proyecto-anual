import multiprocessing
import wfdb
import neurokit2 as nk
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Mapeo de clases
mapeo_clases = {
    'N': 0, 'V': 1, 'E': 1, 'F': 1, 'J': 1, 'A': 1, 'a': 1,
    '/': 2, 'f': 2, 'L': 3, 'R': 3, 'S': 3, 'P': 3,
    'Q': 4, 'e': 4, '!': 4, 'I': 4, 'i': 4, '+': 4, '~': 4,
    '|': 4, 's': 4, 'T': 4, '*': 4, 'x': 4, '[': 4, ']': 4,
    'p': 4, 'B': 4, 'b': 4,
}
# Definir la función para procesar ECG
def procesar_ecg(record):
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation_path = f'mit-bih-arrhythmia-database/{record}'

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(annotation_path, 'atr')
    ecg_signal = record.p_signal[:, 0]

    sampling_rate = record.fs
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
    rmssd = hrv_metrics['HRV_RMSSD'].values[0]

    # Extraer características
    features = np.column_stack((
        rr_intervals,
        np.full(len(rr_intervals), rmssd),
        np.full(len(rr_intervals), signals['ECG_Clean'].mean()),
        np.full(len(rr_intervals), signals['ECG_Clean'].std())
    ))

    # Manejar valores faltantes y estandarizar
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Cargar anotaciones y mapear a etiquetas de clase
    simbolos = annotation.symbol
    labels_agrupados = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels_agrupados

# Definir la función para el multiprocesamiento
def procesar_registro(record):
    print(f"Procesando registro {record}...")
    features, labels_agrupados = procesar_ecg(record)
    
    # Ajustar longitudes
    min_len = min(len(features), len(labels_agrupados))
    features = features[:min_len]
    labels_agrupados = labels_agrupados[:min_len]
    
    return features, labels_agrupados

# Listado de registros
records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', 
           '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210',
           '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']

# Multiprocesamiento
if __name__ == '__main__':
    with multiprocessing.Pool(processes=(multiprocessing.cpu_count())//2) as pool:
        results = pool.map(procesar_registro, records)

    # Unir resultados
    X_total, y_total = zip(*results)

    # Convertir listas de listas en arrays concatenados
    X_total = np.concatenate(X_total)
    y_total = np.concatenate(y_total)

    print(f"Total de muestras procesadas: {len(X_total)}")
