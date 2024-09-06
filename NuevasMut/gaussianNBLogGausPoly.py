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

def procesar_ecg(r, a):
    record = wfdb.rdrecord(r)
    annotation = wfdb.rdann(a, 'atr')
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

    # Manejar valores faltantes y estandarizar
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Cargar anotaciones y mapear a etiquetas de clase
    simbolos = annotation.symbol
    labels_agrupados = np.array([mapeo_clases[s] for s in simbolos if s in mapeo_clases])

    return features, labels_agrupados

# Procesar registros
records = ['100', '101', '102']
X_total, y_total = [], []
for record in records:
    print(f"Procesando registro {record}...")
    record_path = f'mit-bih-arrhythmia-database/{record}'
    annotation = f'mit-bih-arrhythmia-database/{record}'
    features, labels_agrupados = procesar_ecg(record_path, annotation)

    min_len = min(len(features), len(labels_agrupados))
    features = features[:min_len]
    labels_agrupados = labels_agrupados[:min_len]

    X_total.append(features)
    y_total.append(labels_agrupados)

X = np.vstack(X_total)
y = np.hstack(y_total)

min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

# Balancear el conjunto de datos
smote = SMOTE(sampling_strategy='not majority',k_neighbors=7)
X_smote, y_smote = smote.fit_resample(X, y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

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
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(reindex_mapeo.values()), yticklabels=list(reindex_mapeo.values()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Crear el tipo de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Definir la función de evaluación (fitness) para GaussianNB
def fitness(individual):
    var_smoothing = 10 ** individual[0]  # Convertir de logaritmo a valor original
    clf = GaussianNB(var_smoothing=var_smoothing)
    scores = cross_val_score(clf, X, y, cv=5)  # Supone que X e y están definidos
    return np.mean(scores),

# Configuración de DEAP
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -9, -1)  # Logaritmo de var_smoothing entre 10^-9 y 10^-1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Mutación Logarítmica
def mut_logarithmic(individual):
    individual[0] += np.random.uniform(-1, 1) * 0.5
    return individual,

# Mutación Gaussiana
def mut_gaussian(individual, mu=0, sigma=0.3):
    individual[0] += np.random.normal(mu, sigma)
    return individual,

# Mutación Polinomial Acotada
def mut_polynomial_bounded(individual, low=-9, up=-1, eta=1.0):
    delta_1 = (individual[0] - low) / (up - low)
    delta_2 = (up - individual[0]) / (up - low)
    rand = np.random.random()
    mut_pow = 1.0 / (eta + 1.0)

    if rand <= 0.5:
        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
        delta_q = (val ** mut_pow) - 1.0
    else:
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
        delta_q = 1.0 - (val ** mut_pow)

    individual[0] += delta_q * (up - low)
    return individual,

# Registrar las mutaciones en el toolbox
toolbox.register("mutate_log", mut_logarithmic)
toolbox.register("mutate_gauss", mut_gaussian, mu=0, sigma=0.3)
toolbox.register("mutate_poly", mut_polynomial_bounded, low=-9, up=-1, eta=1.0)

# Selección y evaluación
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

# Ejemplo de ejecución del algoritmo genético
population = toolbox.population(n=100)

# Proceso evolutivo
NGEN = 40
for gen in range(NGEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Aplicar la mutación logarítmica en la primera fase
    if gen < NGEN // 3:
        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate_log(mutant)
                del mutant.fitness.values

    # Aplicar la mutación gaussiana en la segunda fase
    elif gen < 2 * NGEN // 3:
        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate_gauss(mutant)
                del mutant.fitness.values

    # Aplicar la mutación polinomial acotada en la fase final
    else:
        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate_poly(mutant)
                del mutant.fitness.values

    # Evaluar los nuevos individuos
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# Mejor individuo encontrado
best_ind = tools.selBest(population, 1)[0]
print(f"Mejor individuo: {best_ind[0]}, con fitness: {best_ind.fitness.values[0]}")

clasificador = GaussianNB(var_smoothing=best_ind[0])
clasificador.fit(X_train, y_train)
y_pred = clasificador.predict(X_test)

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
    print(f'MEJORÓ EL RENDIMIENTO EN {(accuracy2 - accuracy1) * 100:.2f}%')
