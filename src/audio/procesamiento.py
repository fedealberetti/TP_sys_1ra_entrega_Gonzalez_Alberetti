import numpy as np
import pandas as pd
from scipy import signal
from generadores import normalizacion_senial
import soundfile as sf

def sintetizacionRI(banda8vas, tiempox8va, fs, tiempoT60):
    # se toma la amplitud Ai=1 para todas frecuencias de banda
    t = np.arange(0,fs*tiempoT60, 1/fs)  # Vector de tiempo para la duración del T60
    listaYi = []
    for i in range(len(banda8vas)):
        taui = np.log(10**(-3))/ tiempox8va[i]
        Yi = (np.exp(taui*t))*np.cos(2*np.pi*banda8vas[i]*t)  # Respuesta al impulso de cada banda
        listaYi.append(Yi)

    sumaYi = np.sum(listaYi)  # Suma de la respuesta al impulso
    # Normalización de la señal o llamar a la f normalizacion_senial
    normsumaYi = normalizacion_senial(listaYi, t) # Normalizar la señal
    # Graficar la respuesta al impulso   
    # Guardar la respuesta al impulso como un archivo WAV
    print(type(sumaYi, "sumaYi"))  # Verificar el tipo de sumaYi,
    return normsumaYi

def funcaRI(fi, sine):
    datosfi, fs = sf.read(fi)  # Cargar el archivo de respuesta al impulso
    datossine, fs = sf.read(sine)  # Cargar el archivo de señal de entrada (sine sweep)
    # Realizar la convolución
    RI = signal.fftconvolve(datossine, datosfi, mode='full')
    # Normalizar la señal resultante y grabarla como un archivo WAV   
    return RI

def main():
    # Definir parámetros segun R1 Nuclear Reactor Hall https://www.openair.hosted.york.ac.uk/?page_id=626
    banda8vas = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])  # Frecuencias de banda
    tiempox8va = np.array([6.98, 5.45, 4.49, 5.15, 5.43, 4.78, 3.6, 2.51, 1.45, 0.98])  # Duraciones de cada banda en segundos
    fs = 44100  # Frecuencia de muestreo
    tiempoT60 = 1.5  # Tiempo de reverberación en segundos

    sintetizacionRI(banda8vas, tiempox8va, fs, tiempoT60)
    # Reproducir senial resultante



if __name__ == "__main__":
    main()   