import numpy as np
import pandas as pd
from scipy import signal
from audio.generadores import *
import soundfile as sf
import matplotlib.pyplot as plt
from .io_audio import guardar_wav, reproducir_audio


def sintetizacionRI(banda8vas, tiempox8va, fs, tiempoT60):
    # se toma la amplitud Ai=1 para todas frecuencias de banda
    t = np.arange(0,tiempoT60, 1/fs)  # Vector de tiempo para la duración del T60
    listaYi = []
    for i in range(len(banda8vas)):
        taui = -(np.log(10**(-3))/ tiempox8va[i])
        Yi = np.exp(-taui * t) * np.cos(2 * np.pi * banda8vas[i] * t)  # Respuesta al impulso de cada banda
        listaYi.append(Yi)

    sumaYi = np.sum(listaYi, axis=0)  # Suma de la respuesta al impulso
    # Normalización de la señal o llamar a la f normalizacion_senial
    normsumaYi = normalizar_senial(sumaYi, t)[0] # Normalizar la señal
    # Graficar la respuesta al impulso   
    # Guardar la respuesta al impulso como un archivo WAV
    #  print(type(sumaYi, "sumaYi"))  # Verificar el tipo de sumaYi,
    return t, normsumaYi




def funcRI(fi, sine):
    datosfi, fs = sf.read(fi)  # Cargar el archivo de respuesta al impulso
    datossine, fs = sf.read(sine)  # Cargar el archivo de señal de entrada (sine sweep)
    # Realizar la convolución
    RI = signal.fftconvolve(datossine, datosfi, mode='full')
    # Normalizar la señal resultante y grabarla como un archivo WAV   
    return RI




def filtronorma(senial, fs, tipo='octava', order=4):
    audiodata, fs = sf.read(senial)
    """
    Aplica filtros de octava o tercio de octava según norma IEC 61260.
    
    Parámetros:
        audiodata (array): Señal de audio.
        fs (float): Frecuencia de muestreo (Hz).
        band_type (str): 'octave' o 'third-octave'.
        order (int): Orden del filtro (recomendado: 4).
        zero_phase (bool): Si True, usa filtrado sin fase (filtfilt).
    
    Retorna:
        dict: {frecuencia_central: señal_filtrada}
    """
    # Definir frecuencias centrales según IEC 61260
    if tipo == 'octava':
        centers = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        G = 1.0  # Ancho de banda para octava
    elif tipo == 'third-octave':
        centers = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
                   250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        G = 1.0 / 3.0  # Ancho de banda para tercio
    else:
        raise ValueError("band_type debe ser 'octave' o 'third-octave'")

    factor = 2 ** (G / 2)  # Cálculo del factor de banda
    nyquist = fs / 2
    filtroporbandas = {}

    for fc in centers:
        # Calcular frecuencias de corte
        lower = fc / factor
        upper = fc * factor
        
        # Saltar bandas que exceden Nyquist
        if upper > nyquist:
            continue
        
        # Diseñar filtro Butterworth en formato SOS
        sos = signal.iirfilter(
            N=order, 
            Wn=[lower, upper], 
            btype='band', 
            analog=False, 
            ftype='butter', 
            fs=fs, 
            output='sos'
        )
        
 
        filtered_data = signal.sosfilt(sos, audiodata)
        
        filtroporbandas[fc] = filtered_data

    return filtroporbandas

def ir_a_log(archivo):
    fs=44100  # Frecuencia de muestreo por defecto, se puede ajustar según el archivo
    """
    Convierte una respuesta al impulso (RI) a una escala logarítmica normalizada.
    
    La función aplica la siguiente fórmula:
        R(t) = 20 * log10( A(t) / A(t)_max )
    
    Parámetros:
        impulse_response : array-like
            Arreglo que contiene la respuesta al impulso.

    Retorna:
        np.array: Arreglo con la señal transformada a escala logarítmica (dB).
    """
    # Se asegura que impulse_response sea un arreglo de NumPy
    audiodata, fs = sf.read(archivo)

    # Se obtiene el valor máximo absoluto para la normalización
    A_max = np.max(np.abs(audiodata))

    # Evitar división por cero: si A_max es 0, se devuelve un array con -inf (logaritmo indefinido)
    if A_max == 0:
        return np.full_like(audiodata, -np.inf)

    # Se aplica la fórmula de conversión a escala logarítmica
    irlog = 20 * np.log10(np.abs(audiodata) / A_max)


    return irlog