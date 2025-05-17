from audio.generadores import generar_ruido_rosa
from audio.io_audio import guardar_wav, reproducir_audio
from utils.graficotemporal import graficar_dominio_temporal
from utils.graficofrecuencia import graficar_espectro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd


def main():
    # Generar 5 segundos de ruido rosa
    t = 5
    señal = generar_ruido_rosa(t)
    
    # Guardar archivo
    guardar_wav("ruido_rosa.wav", señal)

    # Genera grafico en el dominio temporal de la señal
    graficar_dominio_temporal(señal, frecuencia_muestreo=44100)

    # Genera grafico en el dominio de la frecuencia de la señal
    graficar_espectro()

      # Reproducir audio
    print("Reproduciendo ruido rosa durante", t, "segundos...")
    reproducir_audio(señal)

if __name__ == "__main__":
    main()