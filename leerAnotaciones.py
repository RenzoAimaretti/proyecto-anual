import wfdb
import pandas as pd

# Ruta al archivo .atr sin la extensión
file_base = 'mit-bih-arrhythmia-database/101'

# Leer el archivo .atr
record = wfdb.rdrecord(file_base)
annotations = wfdb.rdann(file_base, 'atr')

# Mostrar información sobre el archivo
print(f"Anotaciones para el archivo {file_base}.atr:")

# Mostrar las anotaciones
for i in range(len(annotations.sample)):
    print(f"Tiempo: {annotations.sample[i]/1000}, Tipo: {annotations.symbol[i]}")

# Crear un DataFrame con las anotaciones
df = pd.DataFrame({
    'Tipo': annotations.symbol
})

# Realizar el recuento de cada tipo de símbolo
count = df['Tipo'].value_counts()

# Mostrar el recuento
print(count)


