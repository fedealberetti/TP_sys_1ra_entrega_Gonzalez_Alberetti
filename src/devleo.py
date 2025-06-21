from audio.generadores import generar_ruido_rosa, generar_sine_sweep, generar_fi, normalizar_senial
from audio.io_audio import guardar_wav, reproducir_audio
from audio.io_audio import *
from audio.procesamiento import filtronorma, funcRI, ir_a_log, sintetizar_respuesta_impulso, aplicar_transformada_hilbert, promedio_movil_convolucion
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
#
    

    audio_data, sample_rate = sf.read("RI_R1_Reactor_Hall.wav")
# Analizar transformada de Hilbert
    trafo_hilbert = aplicar_transformada_hilbert(audio_data)
    # norm_ri_trafo_hilbert = normalizar_senial(trafo_hilbert, t)[0]
    graficar_dominio_temporal(trafo_hilbert, sample_rate)
    promedio_movil = promedio_movil_convolucion(audio_data, 5000)
    esc_log = ir_a_log(promedio_movil)
    graficar_dominio_temporal(esc_log, sample_rate)
    reproducir_audio(promedio_movil, sample_rate)
    
if __name__ == "__main__":
    main()