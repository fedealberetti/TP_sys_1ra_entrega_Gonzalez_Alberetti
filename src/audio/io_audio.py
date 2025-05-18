from scipy.io.wavfile import write
import numpy as np

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