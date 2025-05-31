from scipy.io.wavfile import write
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

def guardar_wav(nombre_archivo, señal, fs=44100):
    """
    Guarda la señal como archivo WAV
    
    Args:
        nombre_archivo (str): Ruta del archivo de salida
        señal (np.array): Señal de audio
        fs (int): Frecuencia de muestreo
    """
    
    # Guardar archivo WAV
    
    write(nombre_archivo, fs, (señal * 32767).astype(np.int16))
    
    #write(nombre_archivo, fs, señal)

def reproducir_audio(señal, fs=44100):
    """Reproduce la señal usando sounddevice"""
    import sounddevice as sd  # Import local para mejor portabilidad
    sd.play(señal, fs)
    sd.wait()

def reproducir_y_grabar(senal, fs=44100):
    """
    Reproduce una señal y graba al mismo tiempo.
    
    :param senal: Señal a reproducir (array NumPy)
    :param fs: Frecuencia de muestreo
    :return: Señal grabada
    """
    duracion = len(senal) / fs
    print(f"Reproduciendo y grabando por {duracion:.2f} segundos...")

    # Reproducir y grabar en simultáneo
    grabado = sd.playrec(senal, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    print("Grabación terminada.")
    return grabado.flatten()

def medir_latencia():
    """
    Estima la latencia del sistema de audio usando una señal de impulso.

    :return: Latencia estimada en segundos y en muestras
    :rtype: tuple[float, int]
    """
    fs = 44100
    dur = 1.0  # duración de 1 segundo

    # Crear señal de prueba: impulso en la muestra 100
    senal = np.zeros(int(fs * dur))
    senal[100] = 1.0

    # Reproducir y grabar usando función ya definida
    grabado = reproducir_y_grabar(senal, fs)

    # Aplanar por si quedó con forma (N,1)
    grabado = grabado.flatten()

    # Calcular correlación cruzada
    corr = correlate(grabado, senal)
    delay_samples = np.argmax(corr) - len(senal) + 1
    latencia_segundos = delay_samples / fs

    # Mostrar resultado
    print(f"Latencia estimada: {latencia_segundos:.4f} segundos ({delay_samples} muestras)")

    # Graficar
    t = np.linspace(0, dur, len(senal))
    plt.figure(figsize=(12, 4))
    plt.plot(t, senal, label="Señal emitida")
    plt.plot(t, grabado, label="Señal grabada")
    plt.title(f"Latencia estimada: {latencia_segundos*1000:.2f} ms")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Escuchar señal grabada
    reproducir_audio(grabado, fs)

    return latencia_segundos, delay_samples

# Ejecutar
medir_latencia()