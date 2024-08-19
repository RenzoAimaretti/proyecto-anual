import wfdb
import neurokit2 as nk
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from deap import base, creator, tools, algorithms
from sklearn.impute import SimpleImputer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Mapeo de clases
mapeo_clases = {
    'N': 0,  # Latido normal
    'V': 1,  # Contracción ventricular prematura (PVC)
    'E': 1,  # Latido de escape ventricular
    'F': 1,  # Latido de fusión (Fusion of ventricular and normal beat)
    'J': 1,  # Latido de escape nodal (Nodal (junctional) escape beat)
    'A': 1,  # Contracción auricular prematura (Atrial premature beat)
    'a': 1,  # Contracción auricular aberrante (Aberrated atrial premature beat)
    '/': 2,  # Latido de fusión de latido normal y PVC (Fusion of paced and normal beat)
    'f': 2,  # Latido de fusión de latido normal y latido aberrante (Fusion of paced and ventricular beat)
    'L': 3,  # Latido de escape del nodo SA (Left bundle branch block beat)
    'R': 3,  # Latido de escape ventricular (Right bundle branch block beat)
    'S': 3,  # Contracción supraventricular prematura (Supraventricular premature beat)
    'P': 3,  # Latido por marcapasos (Paced beat)
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
    annotation = wfdb.rdann(a, 'atr')
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
    simbolos = annotation.symbol
    labels_agrupados = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels_agrupados

# Procesar los registros
records = ['100', '101', '102', '103']
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

# Aplicar SMOTE para manejar desbalanceo de clases
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Configurar el clasificador k-NN
k = 5  # Número de vecinos
clasificador = KNeighborsClassifier(n_neighbors=k)
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)

# Reindexar etiquetas para reportes
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
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
accuracy1=accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(reindex_mapeo.values()), yticklabels=list(reindex_mapeo.values()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Configurar y ejecutar el algoritmo genético (si deseas ajustar hiperparámetros del k-NN)
def fitness(individual):
    k = int(individual[0])
    k = max(1, min(k, 20))  # Asegurar que k esté en el rango [1, 20]
    clasificador = KNeighborsClassifier(n_neighbors=k)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return (accuracy,)

def random_int(low, up):
    return np.random.randint(low, up + 1)
# Configurar el algoritmo genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random_int, low=1, up=20)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=1, up=20, eta=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

# Ejecutar el algoritmo genético
population = toolbox.population(n=50)
ngen = 40
cxpb = 0.5
mutpb = 0.2

result, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                                  stats=None, halloffame=None, verbose=True)

# Obtener el mejor individuo
best_individual = tools.selBest(population, k=1)[0]
k_best = int(best_individual[0])
print(f"Mejor hiperparámetro: k={k_best}")

# Entrenar el modelo con el mejor hiperparámetro
clasificador = KNeighborsClassifier(n_neighbors=k_best)
clasificador.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clasificador.predict(X_test)

# Generar el reporte
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

accuracy2 = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(reindex_mapeo.values()), yticklabels=list(reindex_mapeo.values()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
if accuracy2 > accuracy1:
    print(f'MEJORÓ EL RENDIMIENTO EN {(accuracy2 - accuracy1) * 100}%')

