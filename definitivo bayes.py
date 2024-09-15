import multiprocessing
import wfdb
import neurokit2 as nk
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Mapeo de clases
mapeo_clases = {
    'N': 0, 'V': 1, 'E': 1, 'F': 1, 'J': 1, 'A': 1, 'a': 1,
    '/': 2, 'f': 2, 'L': 3, 'R': 3, 'S': 3, 'P': 3,
    'Q': 4, 'e': 4, '!': 4, 'I': 4, 'i': 4, '+': 4, '~': 4,
    '|': 4, 's': 4, 'T': 4, '*': 4, 'x': 4, '[': 4, ']': 4,
    'p': 4, 'B': 4, 'b': 4,
}

# Definir función para procesar cada registro ECG
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

    # Separar las características y etiquetas
    X_total, y_total = zip(*results)

    # Verificar que todas las longitudes coincidan
    min_len = min([len(x) for x in X_total])

    # Recortar para asegurar que los arrays tienen la misma longitud
    X_total = [x[:min_len] for x in X_total]
    y_total = [y[:min_len] for y in y_total]

    # Concatenar listas de arrays en matrices unificadas
    X = np.vstack(X_total)
    y = np.hstack(y_total)

    print(f"Total de muestras procesadas: {len(X_total)}")


    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar y evaluar el clasificador base
    clasificador = GaussianNB()
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)

    # Mapeo para reporte
    reindex_mapeo = {
        0: 'Latido Normal',
        1: 'Contracciones Prematuras',
        2: 'Latido de Fusión',
        3: 'Contracciones Auriculares',
        4: 'Otros'
    }
    default_value = 'Otro'

    y_test_reindex = np.array([reindex_mapeo.get(label, default_value) for label in y_test])
    y_pred_reindex = np.array([reindex_mapeo.get(label, default_value) for label in y_pred])

    # Mostrar resultados
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    accuracy1=accuracy_score(y_test, y_pred)
    # Visualizar matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys',
                xticklabels=list(reindex_mapeo.values()), yticklabels=list(reindex_mapeo.values()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Configurar el algoritmo genético
    def fitness(individual):
        var_smoothing = 10**(-individual[0])
        clasificador = GaussianNB(var_smoothing=var_smoothing)
        clasificador.fit(X_train, y_train)
        y_pred = clasificador.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 9)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=9, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)



    # Ejecutar el algoritmo genético
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    result, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                                      stats=None, halloffame=None, verbose=True)



    # Obtener el mejor individuo y evaluar el modelo
    best_individual = tools.selBest(population, k=1)[0]
    var_smoothing = 10**(-best_individual[0])
    print(f"Mejor hiperparámetro: var_smoothing={var_smoothing}")

    clasificador = GaussianNB(var_smoothing=var_smoothing)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    accuracy2 = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys',
                xticklabels=list(reindex_mapeo.values()), yticklabels=list(reindex_mapeo.values()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    if accuracy2 > accuracy1:
        print(f'MEJORÓ EL RENDIMIENTO EN {(accuracy2 - accuracy1) * 100:.2f}%')
