from scipy.io.wavfile import write

def guardar_wav(nombre_archivo, señal, fs=44100):
    """
    Guarda la señal como archivo WAV
    
    Args:
        nombre_archivo (str): Ruta del archivo de salida
        señal (np.array): Señal de audio
        fs (int): Frecuencia de muestreo
    """
    write(nombre_archivo, fs, señal)

def reproducir_audio(señal, fs=44100):
    """Reproduce la señal usando sounddevice"""
    import sounddevice as sd  # Import local para mejor portabilidad
    sd.play(señal, fs)
    sd.wait()