import neurokit2 as nk
import matplotlib.pyplot as plt
# Simular una señal de ECG de ejemplo
ecg_signal = nk.ecg_simulate(duration=10, noise=0.1)

# Procesar la señal de ECG
signals, info = nk.ecg_process(ecg_signal, sampling_rate=1000)

# Mostrar los resultados
nk.ecg_plot(signals)

# Extraer características específicas
hrv = nk.hrv_time(signals, sampling_rate=1000)

# Graficar la señal original y las características extraídas
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(signals['ECG_Clean'])
plt.title("Señal de ECG Filtrada")
plt.subplot(212)
plt.plot(hrv['HRV_RMSSD'])
plt.title("Variabilidad de la Frecuencia Cardíaca (RMSSD)")
plt.tight_layout()
plt.show()