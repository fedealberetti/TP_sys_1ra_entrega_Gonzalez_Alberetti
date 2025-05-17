import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import sounddevice as sd

def generar_ruido_rosa(t, ncols=16, fs=44100):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney.
    
    Parámetros
    ----------
    t : float
        Duración en segundos del ruido generado.
    ncols : int, opcional
        Número de columnas (fuentes aleatorias). Por defecto 16.
    fs : int, opcional
        Frecuencia de muestreo en Hz. Por defecto 44100.
    
    Retorna
    -------
    numpy.ndarray
        Array con la señal de ruido rosa normalizada en formato int16.
    """
    nrows = int(t * fs)  # Muestras totales
    
    # Inicialización de matriz con valores aleatorios
    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # Generación de cambios aleatorios
    cols = np.random.geometric(0.5, nrows)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=nrows)
    array[rows, cols] = np.random.random(nrows)
    
    # Relleno hacia adelante y suma
    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1).values

    # Normalización y conversión a formato WAV
    """
PREPARACIÓN DE LA SEÑAL PARA ARCHIVO WAV
1. Centrado en cero: Remueve el componente DC (frecuencia 0 Hz) que puede causar
   clicks en la reproducción y distorsión en equipos de audio
2. Normalización: Asegura que la señal ocupe todo el rango dinámico [-1, 1]
   - Maximiza la relación señal/ruido (SNR)
   - Previene saturación (clipping) al usar máximo rango sin recorte
3. Conversión a 16 bits: Formato estándar para audio WAV de calidad CD
   - 32767 = Valor máximo positivo en enteros con signo de 2 bytes (2^15 - 1)
   - Permite compatibilidad universal con reproductores y software de audio
    """

    total = total - np.mean(total)  # Centrado en 0
    total = total / np.max(np.abs(total))  # Rango [-1, 1]
    total = (total * 32767).astype(np.int16)  # Conversión a int16 para WAV
    
 
    return total