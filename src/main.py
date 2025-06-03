from audio.generadores import *
from audio.io_audio import *
from audio.procesamiento import filtronorma, funcRI, ir_a_log, sintetizar_respuesta_impulso
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
#     # Generar 5 segundos de ruido rosa
#     t = 5
#     pink, vt = generar_ruido_rosa(t)
#     normalizar_senial(pink, vt)
    
#     # Guardar archivo
#     guardar_wav("ruido_rosa.wav", pink)

#     # Genera grafico en el dominio temporal de la señal
#     graficar_dominio_temporal(pink, frecuencia_muestreo=44100)

#     # Genera grafico en el dominio de la frecuencia de la señal
#     graficar_espectro()

#       # Reproducir audio
#     print("Reproduciendo ruido rosa durante", t, "segundos...")
#     reproducir_audio(pink)

#     # Genera sine sweep de 20 a 20000 Hz en 5 segundos
  
#    # normalizar_senial(sine_sweep,t)

#     sine, vec_t, R, K, L, w1 = generar_sine_sweep(20, 20000, t,44100)
#     normalizar_senial(sine, vec_t)

#     # Graficar en el dominio frecuencia
#     graficar_espectro_sweep()



#     # Guardar archivo sine sweep
#     guardar_wav("sine_sweep.wav", sine)

#     # Guardar archivo filtro inverso
#     filtinv = generar_fi(sine,vec_t, R, K, L, w1)
#     normalizar_senial(filtinv, vec_t)
    
#     guardar_wav("filtro_inverso.wav", filtinv)

#     # graficar en el dominio de la frecuencia filtro inverso
#     graficar_espectro_inv()    

#     # Reproducir audio
#     reproducir_audio(sine)
#     reproducir_audio(filtinv)
#  # Ejemplo: señal de prueba
#     fs = 44100
#     t = np.linspace(0, 5, int(5 * fs), endpoint=False)
#     senal = np.random.randn(len(t))  # Generar una señal de ruido blanco como ejemplo

#     # Ejecutar función de reproducción y grabación
#     grabado = reproducir_y_grabar(senal, fs)
#     # Reproduce lo grabado en reproducción y grabacion
#     reproducir_audio(grabado, fs)
#     # Medir latencia del sistema de audio
#     medir_latencia()

  # # Datos del R1 Nuclear Reactor Hall
  # banda8vas = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
  # tiempox8va = [6.98, 5.45, 4.49, 5.15, 5.43, 4.78, 3.6, 2.51, 1.45, 0.98]
  # fs = 44100
  # tiempoT60 = 8  # Duración total de la respuesta al impulso

  # # Sintetizar la respuesta al impulso
  # t, ri = sintetizar_respuesta_impulso(t60_por_banda=tiempox8va,fs=fs,duracion=tiempoT60,frecuencias_centrales=banda8vas)

  # # Graficar dominio temporal
  # graficar_dominio_temporal(ri, fs)

  # # Guardar como archivo WAV
  # guardar_wav("RI_R1_Reactor_Hall.wav", ri, fs)

  # # Reproducir audio
  # print("Reproduciendo respuesta al impulso sintetizada...")
  # reproducir_audio(ri, fs)




# Datos para tercio de octava
  frecuencias_centrales = [
    25, 31, 39, 50, 62, 79, 99, 125, 157, 198,
    250, 315, 397, 500, 630, 794, 1000, 1260, 1587, 2000,
    2520, 3175, 4000, 5040, 6350, 8000, 10079, 12699, 16000
  ]

  t60_por_banda = [
    38.89, 36.02, 29.22, 20.60, 22.32, 23.35, 22.36, 18.38, 18.60, 16.15,
    14.81, 14.29, 13.51, 11.79, 10.66, 10.03, 9.09, 7.85, 6.53, 5.60,
    4.68, 3.66, 2.83, 2.13, 1.56, 1.10, 0.84, 0.64, 0.56
  ]

  # Configuración de síntesis
  fs = 44100  # Frecuencia de muestreo
  duracion_total = 40.0  # Duración total en segundos (suficiente para el T60 más largo)

  # Sintetizar la respuesta al impulso para tercios de octava
  t, ri_tercios = sintetizar_respuesta_impulso(
    t60_por_banda=t60_por_banda,
    fs=fs,
    duracion=duracion_total,
    frecuencias_centrales=frecuencias_centrales
  )

  # Graficar dominio temporal
  graficar_dominio_temporal(ri_tercios, fs)

  # Guardar como archivo WAV
  guardar_wav("RI_Tercios_Octava.wav", ri_tercios, fs)

  # Reproducir audio
  print("Reproduciendo respuesta al impulso sintetizada (tercios de octava)...")
  reproducir_audio(ri_tercios, fs)


      # Parte 2
  convolucionFIconRI = funcRI("filtro_inverso.wav", "sine_sweep.wav")
  normconvFIconsinsweep = normalizar_senial(convolucionFIconRI, t)[0]

  guardar_wav("convolucion_FI_con_sinesweep.wav", normconvFIconsinsweep, fs=fs)
  graficar_dominio_temporal(normconvFIconsinsweep, frecuencia_muestreo=fs)
  reproducir_audio(normconvFIconsinsweep, fs=fs)

  convolucionFIconReactor = funcRI("filtro_inversoDRIVE.wav", "Toma_n3_c-03.wav")
  normconvFIconReactor = normalizar_senial(convolucionFIconReactor, t)[0]
  guardar_wav("convolucion_FI_con_Sinesweepaula.wav", normconvFIconReactor, fs=fs)
  graficar_dominio_temporal(normconvFIconReactor, frecuencia_muestreo=fs)
  reproducir_audio(normconvFIconReactor, fs=fs)

      # Parte 3
  fs = 44100  # Frecuencia de muestreo
  freccentral = 1000  # Frecuencia central para el filtro
  senialporbandas = filtronorma("convolucion_FI_con_Sinesweepaula.wav", fs=fs, tipo='octava', order=4)
  guardar_wav("convolucion_FI_con_Sinesweepaula1000hz.wav", senialporbandas[freccentral], fs=fs)
  graficar_dominio_temporal(senialporbandas[freccentral], frecuencia_muestreo=fs)
  reproducir_audio(senialporbandas[freccentral], fs=fs)

  RIfiltnormlog = ir_a_log("RI_sintetizada.wav")
  graficar_dominio_temporal(RIfiltnormlog, frecuencia_muestreo=fs)

if __name__ == "__main__":
    main()