from audio.generadores import *
from audio.io_audio import guardar_wav, reproducir_audio
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

from utils import espectro
from utils import espectroSine_sweep
from utils.graficofrecuencia import graficar_espectro, graficar_espectro_inv, graficar_espectro_sweep
from utils.graficotemporal import graficar_dominio_temporal


def main():
    # Generar 5 segundos de ruido rosa
    t = 5
    pink, vt = generar_ruido_rosa(t)
    normalizar_senial(pink, vt)
    
    # Guardar archivo
    guardar_wav("ruido_rosa.wav", pink)

    # Genera grafico en el dominio temporal de la señal
    graficar_dominio_temporal(pink, frecuencia_muestreo=44100)

    # Genera grafico en el dominio de la frecuencia de la señal
    graficar_espectro()

      # Reproducir audio
    print("Reproduciendo ruido rosa durante", t, "segundos...")
    reproducir_audio(pink)

    # Genera sine sweep de 20 a 20000 Hz en 5 segundos
  
   # normalizar_senial(sine_sweep,t)

    sine, vec_t, R, K, L, w1 = generar_sine_sweep(20, 20000, t,44100)
    normalizar_senial(sine, vec_t)

    # Graficar en el dominio frecuencia
    graficar_espectro_sweep()



    # Guardar archivo sine sweep
    guardar_wav("sine_sweep.wav", sine)

    # Guardar archivo filtro inverso
    filtinv = generar_fi(sine,vec_t, R, K, L, w1)
    normalizar_senial(filtinv, vec_t)
    
    guardar_wav("filtro_inverso.wav", filtinv)

    # graficar en el dominio de la frecuencia filtro inverso
    graficar_espectro_inv()    

    # Reproducir audio
    reproducir_audio(sine)
    reproducir_audio(filtinv)

if __name__ == "__main__":
    main()